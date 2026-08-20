# coding=utf-8
"""IP102 dataset loader (COCO-format annotations) for the L2P repo.

The dataset is a single Input on Kaggle / a folder locally, containing:
  * train.json / val.json / test.json  (COCO-style: images, annotations,
    categories)  -- val.json is optional and falls back to test.json
  * filtered_class.txt / classes.txt   (the 25-class mapping used everywhere)
  * a JPEGImages / images folder with the actual pictures

This module is deliberately import-light (numpy/os/json only) so the dataset
parser and the pure-metric unit tests can run without a full JAX/TF stack.
TF datasets are built from the path lists in ``libml/input_pipeline.py``.
"""

import json
import os

import numpy as np

DATASET_REGISTRY = {}

SKIP_DIRS = {
    '.git', '.idea', '.ipynb_checkpoints', '.venv', 'venv', '__pycache__',
    'node_modules', 'model_saved_check', 'output', 'output_ip102', 'logs',
    'exps', 'runs', 'checkpoints', 'Weight', 'weights',
}


def register_dataset(name):
  def _deco(cls):
    DATASET_REGISTRY[name] = cls
    return cls
  return _deco


def get_data_manager(name, *args, **kwargs):
  if name not in DATASET_REGISTRY:
    raise KeyError('Dataset %r chua dang ky. Da co: %s'
                   % (name, sorted(DATASET_REGISTRY)))
  return DATASET_REGISTRY[name](*args, **kwargs)


def deep_walk_find(base, is_target, max_depth=5, skip=SKIP_DIRS):
  """Deep-walk `base` and return the first dir matching `is_target(dir)`."""
  base = os.path.abspath(base)
  if not os.path.isdir(base):
    return None
  for dirpath, dirnames, filenames in os.walk(base):
    depth = dirpath[len(base):].count(os.sep)
    if depth > max_depth:
      dirnames[:] = []
      continue
    dirnames[:] = [d for d in dirnames
                   if not d.startswith('.') and d not in skip]
    try:
      if is_target(dirpath):
        return dirpath
    except OSError:
      continue
  return None


def find_data_root(env_var='IP102_DATA_ROOT', marker='train.json'):
  """Auto-locate the dataset folder.

  Priority:
    1. env var (e.g. IP102_DATA_ROOT) pointing straight at the folder
    2. /kaggle/input  (Kaggle notebook, single Dataset Input)
    3. local deep-walk from cwd / this file / its parent / grandparent
  """
  if env_var and os.environ.get(env_var):
    cand = os.environ[env_var].strip()
    if os.path.isdir(cand) and os.path.exists(os.path.join(cand, marker)):
      return cand
  if os.path.isdir('/kaggle/input'):
    found = deep_walk_find(
        '/kaggle/input',
        lambda d: os.path.exists(os.path.join(d, marker)),
        max_depth=6)
    if found:
      return found
  script_dir = os.path.dirname(os.path.abspath(__file__))
  roots = [os.getcwd(), script_dir, os.path.dirname(script_dir)]
  parent = os.path.dirname(script_dir)
  grand = os.path.dirname(parent) if parent else None
  if grand and os.path.isdir(grand):
    roots.append(grand)
  for extra in ('IP102 dataset', 'dataset', 'data'):
    for root in (script_dir, parent):
      cand = os.path.join(root, extra)
      if os.path.isdir(cand):
        roots.append(cand)
  seen = set()
  for root in roots:
    root = os.path.abspath(root)
    if root in seen:
      continue
    seen.add(root)
    found = deep_walk_find(
        root, lambda d: os.path.exists(os.path.join(d, marker)),
        max_depth=5)
    if found:
      return found
  raise FileNotFoundError(
      'Khong tim thay thu muc dataset (chua %s). Dat bien moi truong %s '
      'hoac dat dataset gan code.' % (marker, env_var))


def find_image_dir(data_root):
  """Locate the image directory inside a dataset root (deep-walk)."""
  for rel in ('VOC2007/VOC2007/JPEGImages', 'VOC2007/JPEGImages',
              'JPEGImages', 'images', 'Images'):
    p = os.path.join(data_root, rel)
    if os.path.isdir(p):
      return p

  def _has_images(d):
    if os.path.basename(d).lower() not in ('jpegimages', 'images'):
      return False
    try:
      names = os.listdir(d)[:50]
    except OSError:
      return False
    return any(n.lower().endswith(('.jpg', '.jpeg', '.png')) for n in names)

  found = deep_walk_find(data_root, _has_images, max_depth=5)
  if found:
    return found
  raise FileNotFoundError('Khong tim thay thu muc chua anh trong ' + data_root)


def find_side_file(data_root, name):
  """Find filtered_class.txt / classes.txt inside/around the data root."""
  for cand in (os.path.join(data_root, name),):
    if os.path.isfile(cand):
      return cand
  for base in (data_root, os.path.dirname(os.path.abspath(data_root))):
    found = deep_walk_find(
        base, lambda d, n=name: os.path.isfile(os.path.join(d, n)),
        max_depth=4)
    if found:
      return os.path.join(found, name)
  return None


def _load_coco_split(json_path):
  with open(json_path, 'r', encoding='utf-8') as f:
    d = json.load(f)
  file_name = {im['id']: im['file_name'] for im in d['images']}
  return file_name, d['annotations']


def _build_image_label(anns):
  """One label per image: keep the annotation with the largest area."""
  image_id_to_cat = {}
  for a in anns:
    img = a['image_id']
    cat = a['category_id']
    area = a.get('area', 0)
    if img not in image_id_to_cat or area > image_id_to_cat[img][1]:
      image_id_to_cat[img] = (cat, area)
  return {k: v[0] for k, v in image_id_to_cat.items()}


def _seed_sample(files, cap, seed):
  """Deterministic per-class subsample (used by the memory_size knob)."""
  files = list(files)
  if len(files) <= cap:
    return files
  idx = np.random.RandomState(seed).permutation(len(files))[:cap]
  return [files[i] for i in sorted(idx)]


@register_dataset('ip102')
class IP102DataManager(object):
  """IP102 (COCO-format) incremental data manager.

  The 25 classes are read from ``filtered_class.txt`` (NOT inferred from the
  JSON), and their names from ``classes.txt``. Supports train/val/test splits,
  a dedicated val split with fallback to test, and an optional deterministic
  class-order shuffle (PASS convention: seed 1993).
  """

  def __init__(self, data_root=None, shuffle=True, seed=1993,
               split_val=True, task_sizes=None):
    self.data_root = data_root or find_data_root()
    self.image_dir = find_image_dir(self.data_root)
    self.shuffle = shuffle
    self.seed = seed
    self.split_val = split_val
    self.task_sizes = task_sizes or [7, 6, 6, 6]
    self._load_meta()
    self._apply_class_order()

  # ------------------------------------------------------------------ io
  def _load_meta(self):
    self._meta = {}
    for s in ('train', 'val', 'test'):
      jp = os.path.join(self.data_root, s + '.json')
      self._meta[s] = _load_coco_split(jp) if os.path.exists(jp) \
          else (None, None)
    self.available_splits = [s for s in ('train', 'val', 'test')
                             if self._meta[s][0] is not None]
    assert 'train' in self.available_splits, 'train.json bat buoc phai co'
    self.val_split = 'val' if (self.split_val and 'val' in
                               self.available_splits) else 'test'
    self.test_split = 'test'

    # class ids: strictly from filtered_class.txt when present
    filtered_path = find_side_file(self.data_root, 'filtered_class.txt')
    if filtered_path:
      with open(filtered_path, 'r', encoding='utf-8') as f:
        ids = [int(line.strip()) for line in f
               if line.strip() and not line.strip().startswith('#')]
    else:
      all_ids = set()
      for s in self.available_splits:
        all_ids |= set(a['category_id'] for a in self._meta[s][1])
      ids = sorted(all_ids)
    self.class_ids = ids
    self.num_classes = len(self.class_ids)
    self.class_id_to_idx = {cid: i for i, cid in enumerate(self.class_ids)}

    # class names: from classes.txt ("<id> <name>") when present
    self.class_names = [None] * self.num_classes
    classes_path = find_side_file(self.data_root, 'classes.txt')
    if classes_path:
      id_to_name = {}
      with open(classes_path, 'r', encoding='utf-8') as f:
        for line in f:
          parts = line.strip().split(' ', 1)
          if len(parts) == 2 and parts[0].isdigit():
            id_to_name[int(parts[0])] = parts[1].strip()
      for i, cid in enumerate(self.class_ids):
        self.class_names[i] = id_to_name.get(cid)

    self._build_by_class()

  def _build_by_class(self):
    self._by_cls = {}
    for s in self.available_splits:
      file_name, anns = self._meta[s]
      img_label = _build_image_label(anns)
      by_cls = {i: [] for i in range(self.num_classes)}
      for img_id, cat in img_label.items():
        if cat in self.class_id_to_idx:
          by_cls[self.class_id_to_idx[cat]].append(file_name[img_id])
      self._by_cls[s] = {i: [os.path.join(self.image_dir, f)
                             for f in files]
                         for i, files in by_cls.items()}

  def _apply_class_order(self):
    order = (np.random.RandomState(self.seed).permutation(self.num_classes)
             if self.shuffle else np.arange(self.num_classes)).tolist()
    self.class_order = order
    new_by = {}
    for s in self.available_splits:
      new_by[s] = [self._by_cls[s][orig] for orig in order]
    self._by_cls = new_by

  # -------------------------------------------------------------- public
  def _resolve(self, split):
    if split is None:
      return 'train'
    if split not in self.available_splits:
      if split == 'val' and 'test' in self.available_splits:
        return 'test'
      if split == 'test' and 'val' in self.available_splits:
        return 'val'
      return self.available_splits[0]
    return split

  def split_counts(self, split=None):
    split = self._resolve(split)
    return [len(self._by_cls[split][k]) for k in range(self.num_classes)]

  def get_paths_by_class(self, classes, split='train', memory_size=0,
                         num_seen=None, seed=0):
    """Absolute image paths + labels for a list of class indices."""
    split = self._resolve(split)
    by_cls = self._by_cls[split]
    paths, labels = [], []
    for idx in classes:
      files = by_cls[idx]
      if split == 'train' and memory_size and memory_size > 0:
        n_seen = num_seen if num_seen else max(len(classes), 1)
        cap = max(1, int(memory_size // n_seen))
        files = _seed_sample(files, cap, seed)
      paths += list(files)
      labels += [idx] * len(files)
    return paths, np.asarray(labels, dtype=np.int64)

  def verify(self):
    """Dataset sanity report: 0 missing images, counts match JSON."""
    report = {'num_classes': self.num_classes,
              'class_ids': self.class_ids, 'class_names': self.class_names,
              'val_split': self.val_split,
              'image_dir': self.image_dir, 'data_root': self.data_root}
    for s in self.available_splits:
      file_name, anns = self._meta[s]
      missing = [f for f in file_name.values()
                 if not os.path.exists(os.path.join(self.image_dir, f))]
      img_label = _build_image_label(anns)
      report[s] = {
          'images_in_json': len(file_name),
          'annotations': len(anns),
          'images_labeled': len(img_label),
          'missing_images': len(missing),
          'missing_samples': missing[:5],
      }
    return report