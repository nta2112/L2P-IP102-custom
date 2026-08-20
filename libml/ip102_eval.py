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

  row = []
  for j in range(current_task + 1):
    gs = model.collect_scores(task_group(model, j), split='val')
    row.append(retrieval_metrics(gs['logits'], gs['labels'])['mAP'])
  model.mAP_matrix.append(row)
  T = len(model.mAP_matrix)
  mat = np.zeros((T, T))
  for i, r in enumerate(model.mAP_matrix):
    for j, v in enumerate(r):
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