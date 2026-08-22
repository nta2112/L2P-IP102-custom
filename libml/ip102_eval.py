# coding=utf-8
"""Per-task evaluation and result writing (JAX/TF-free).

Kept separate from the training code so the row generation, results.csv and
history.json writing can be unit-tested without a JAX/TF install. It only
needs a "model" object exposing:

  numclass, total_nc, fg_nc, task_size, prototype, mAP_matrix,
  collect_scores(classes, split, numclass) -> {'logits','features','labels'}
"""

import csv
import json
import os

import numpy as np

from libml.ip102_metrics import (lifelong_metrics, nme_top1, ood_metrics,
                                 recall_at_1_seen_unseen, retrieval_metrics,
                                 topk_acc)

RESULTS_HEADER = ['task', 'numclass', 'cnn_top1', 'nme_top1',
                  'R@1', 'R@5', 'R@10', 'mAP', 'AUROC', 'FPR95',
                  'Recall@1_seen', 'Recall@1_unseen',
                  'Recall@1_unseen_energy',
                  'Plasticity', 'Forgetting', 'Overall']


def _fmt(v):
  if v is None:
    return 'NA'
  if isinstance(v, (bool, np.bool_)):
    return str(int(v))
  if isinstance(v, (int, np.integer)):
    return str(int(v))
  if isinstance(v, (float, np.floating)):
    return '%.6f' % float(v)
  return str(v)


def task_group(model, j):
  """Class indices of task group ``j`` under the 7/6/6/6 IP102 split."""
  if j == 0:
    return list(range(model.fg_nc))
  return list(range(model.fg_nc + (j - 1) * model.task_size,
                     model.fg_nc + j * model.task_size))


def evaluate_task(model, current_task, seen_count=None):
  """Per-task metrics: retrieval + open-world + lifelong (mAP based).

  ``seen_count`` is the number of classes trained up to and including
  ``current_task`` (= fg_nc + task*task_size). It defaults to the model's
  ``numclass`` but MUST be passed explicitly right after ``after_train``
  because that method already advanced ``numclass`` by one task.
  """
  # --- FIX: Khởi tạo/reset mAP_matrix an toàn để tránh IndexError ---
  if not hasattr(model, 'mAP_matrix') or model.mAP_matrix is None:
    model.mAP_matrix = []
  
  # Nếu mAP_matrix có dữ liệu cũ nhưng row lengths không đồng đều -> reset
  if model.mAP_matrix and len(model.mAP_matrix) > 0:
    row_lens = [len(r) for r in model.mAP_matrix]
    expected_len = current_task  # Số row kỳ vọng = số task đã hoàn thành
    if len(model.mAP_matrix) != expected_len or len(set(row_lens)) > 1:
      # Số row không khớp hoặc độ dài row không đồng đều -> reset
      model.mAP_matrix = []
      print(f"⚠️ Reset mAP_matrix: expected {expected_len} rows, got {len(model.mAP_matrix)}; row_lens={row_lens}")
  
  if seen_count is None:
    seen_count = model.numclass
  seen = list(range(seen_count))
  unseen = list(range(seen_count, model.total_nc))

  s = model.collect_scores(seen, split='val', numclass=seen_count)
  if s is None:
    raise RuntimeError('Khong lay duoc scores cho lop da thay (val)')
  ret = retrieval_metrics(s['logits'], s['labels'])
  res = {
      'cnn_top1': topk_acc(s['logits'], s['labels'], (1,))[1],
      'nme_top1': nme_top1(s['features'], s['labels'], model.prototype),
      'R@1': ret['R@1'], 'R@5': ret['R@5'], 'R@10': ret['R@10'],
      'mAP': ret['mAP'],
  }

  if unseen:
    u = model.collect_scores(unseen, split='test', numclass=seen_count)
    if u is None or len(u['logits']) == 0:
      auroc, fpr95, rec_seen, rec_unseen = None, None, None, None
    else:
      auroc, fpr95 = ood_metrics(s['logits'].max(axis=1),
                                 u['logits'].max(axis=1))
      u_full = model.collect_scores(unseen, split='test',
                                    numclass=model.total_nc)
      rec_seen, rec_unseen = recall_at_1_seen_unseen(
          s['logits'], s['labels'], u_full['logits'], u_full['labels'],
          seen_count)
  else:
    auroc, fpr95, rec_seen, rec_unseen = None, None, None, None
  res['AUROC'] = auroc
  res['FPR95'] = fpr95
  res['Recall@1_seen'] = rec_seen
  res['Recall@1_unseen'] = rec_unseen
  res['Recall@1_unseen_energy'] = None  # Default: None (sẽ ghi NA)

  # --- NEW: Thêm energy-based unseen detection metrics ---
  if unseen:
    u_full = model.collect_scores(unseen, split='test',
                                  numclass=model.total_nc)
    if u_full is not None and len(u_full['logits']) > 0:
      unseen_logits = u_full['logits']
      seen_count = len(seen)
      # Energy-based unseen detection
      try:
        _, rec_unseen_energy = recall_at_1_seen_unseen(
            s['logits'], s['labels'],
            unseen_logits, u_full['labels'],
            seen_count,
            temperature=1.0,
            use_energy_for_unseen=True)
        res['Recall@1_unseen_energy'] = rec_unseen_energy
      except Exception as e:
        res['Recall@1_unseen_energy'] = None
        print(f"Warning: Energy-based unseen detection failed: {e}")

  row = []
  for j in range(current_task + 1):
    gs = model.collect_scores(task_group(model, j), split='val')
    row.append(retrieval_metrics(gs['logits'], gs['labels'])['mAP'])
  model.mAP_matrix.append(row)
  # --- FIX: Tính kích thước ma trận dựa trên max row length ---
  max_len = max(len(r) for r in model.mAP_matrix) if model.mAP_matrix else 0
  T = max(max_len, len(model.mAP_matrix))
  if T == 0:
    mat = np.zeros((0, 0))
  else:
    mat = np.zeros((T, T))
  for i, r in enumerate(model.mAP_matrix):
    for j, v in enumerate(r):
      if i < T and j < T:
        mat[i, j] = v
  plastic, forget, overall = lifelong_metrics(mat)
  res['Plasticity'] = plastic
  res['Forgetting'] = forget
  res['Overall'] = overall
  res['mAP_matrix'] = mat.tolist()
  return res


def write_results(out_dir, rows):
  """Append-friendly writer for results.csv (header per specification)."""
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, 'results.csv')
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: _fmt(row[k]) for k in RESULTS_HEADER})
  return path


def write_history(out_dir, cfg, dataset_report, records):
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, 'history.json')
  with open(path, 'w', encoding='utf-8') as f:
    json.dump({'config': cfg, 'dataset': dataset_report,
               'records': records}, f, indent=2)
  return path