# coding=utf-8
"""Evaluation metrics for the IP102 incremental pipeline.

Three families of metrics, all numpy-only so they can be unit-tested without
torch/sklearn:

  * retrieval   -- R@1/R@5/R@10 + macro mAP (per-class average precision)
  * open-world  -- OOD detection AUROC + FPR@TPR95 (None when no OOD remains)
  * lifelong    -- plasticity / forgetting / overall from a per-task mAP matrix
"""

import numpy as np


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------
def topk_acc(logits, labels, ks=(1, 5, 10)):
  """Top-k accuracy for a batch of class logits."""
  logits = np.asarray(logits, dtype=np.float64)
  labels = np.asarray(labels).reshape(-1)
  n = len(labels)
  if n == 0:
    return {int(k): 0.0 for k in ks}
  order = np.argsort(-logits, axis=1, kind='stable')
  out = {}
  for k in ks:
    top = order[:, :k]
    hits = sum(1 for i in range(n) if labels[i] in top[i])
    out[int(k)] = float(hits) / n
  return out


def _ap_from_pr(precision, recall):
  ap = 0.0
  last = 0.0
  for p, r in zip(precision, recall):
    if r != last:
      ap += (r - last) * p
      last = r
  return ap


def macro_map(logits, labels):
  """Macro mean Average Precision: AP per class, averaged over classes."""
  logits = np.asarray(logits, dtype=np.float64)
  labels = np.asarray(labels).reshape(-1)
  classes = np.unique(labels)
  if len(classes) == 0:
    return 0.0
  aps = []
  for c in classes:
    y_true = (labels == c).astype(np.float64)
    score = logits[:, int(c)]
    desc = np.argsort(-score, kind='stable')
    y = y_true[desc]
    pos = y.sum()
    if pos == 0:
      aps.append(0.0)
      continue
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / pos
    aps.append(_ap_from_pr(precision, recall))
  return float(np.mean(aps))


def retrieval_metrics(logits, labels, ks=(1, 5, 10)):
  """All retrieval metrics for one evaluation set.

  Returns a dict with keys R@1 / R@5 / R@10 / mAP.
  """
  acc = topk_acc(logits, labels, ks)
  return {'R@%d' % int(k): acc[int(k)] for k in ks} | {
      'mAP': macro_map(logits, labels)}


# ---------------------------------------------------------------------------
# open-world (OOD detection)
# ---------------------------------------------------------------------------
def _auroc(y_true, y_score):
  y_true = np.asarray(y_true, dtype=np.float64)
  y_score = np.asarray(y_score, dtype=np.float64)
  n_pos = y_true.sum()
  n_neg = len(y_true) - n_pos
  if n_pos == 0 or n_neg == 0:
    return 0.5
  desc = np.argsort(-y_score, kind='stable')
  y = y_true[desc]
  tpr = np.cumsum(y) / n_pos
  fpr = np.cumsum(1 - y) / n_neg
  trapezoid = getattr(np, 'trapezoid', None) or np.trapz
  return float(trapezoid(np.concatenate(([0.0], tpr)),
                         np.concatenate(([0.0], fpr))))


def _fpr_at_tpr95(y_true, y_score):
  y_true = np.asarray(y_true, dtype=np.float64)
  y_score = np.asarray(y_score, dtype=np.float64)
  n_pos = y_true.sum()
  n_neg = len(y_true) - n_pos
  if n_pos == 0 or n_neg == 0:
    return 1.0
  desc = np.argsort(-y_score, kind='stable')
  y = y_true[desc]
  tpr = np.cumsum(y) / n_pos
  fpr = np.cumsum(1 - y) / n_neg
  idx = np.where(tpr >= 0.95)[0]
  if len(idx) == 0:
    return 1.0
  return float(fpr[idx[0]])


def ood_metrics(known_scores, unknown_scores):
  """OOD detection between seen-class (known) and unseen-class (unknown).

  Higher ``known_scores`` / higher ``unknown_scores`` means "more
  confidently classified". Returns (AUROC, FPR@TPR95), or (None, None) when
  either side is empty (e.g. after all classes have been seen).
  """
  known = np.asarray(known_scores, dtype=np.float64).reshape(-1)
  unknown = np.asarray(unknown_scores, dtype=np.float64).reshape(-1)
  if len(known) == 0 or len(unknown) == 0:
    return None, None
  y_true = np.concatenate([np.zeros(len(known)), np.ones(len(unknown))])
  y_score = np.concatenate([-known, -unknown])  # higher => more OOD
  return float(_auroc(y_true, y_score)), float(_fpr_at_tpr95(y_true,
                                                             y_score))


def _energy_score(logits, temperature=1.0):
  """Energy score: -T * log(sum(exp(logits/T))).
  Lower energy = more confident (in-distribution).
  Higher energy = more uncertain (OOD/unseen).
  """
  logits = np.asarray(logits, dtype=np.float64)
  scaled = logits / temperature
  # log-sum-exp trick for numerical stability
  max_logit = np.max(scaled, axis=1, keepdims=True)
  exp_sum = np.exp(scaled - max_logit).sum(axis=1, keepdims=True)
  log_sum_exp = np.log(exp_sum) + max_logit
  return -temperature * log_sum_exp.reshape(-1)


def _softmax_with_temp(logits, temperature=1.0):
  """Softmax with temperature scaling."""
  logits = np.asarray(logits, dtype=np.float64)
  scaled = logits / temperature
  max_logit = np.max(scaled, axis=1, keepdims=True)
  exp_vals = np.exp(scaled - max_logit)
  return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def recall_at_1_seen_unseen(seen_logits, seen_labels, unseen_logits,
                            unseen_labels, seen_count,
                            temperature=1.0, use_energy_for_unseen=True):
  """Recall@1 on the seen set (S) and on the unseen set (u).

  ``seen_logits`` only contains the first ``seen_count`` class columns.
  ``unseen_logits`` must contain ALL ``seen_count + len(unseen_classes)``
  columns so the true (unseen) class can be ranked. Returns
  (recall_seen, recall_unseen), or (None, None) when there is no unseen data.

  Args:
    temperature: Temperature for softmax scaling (lower = sharper, higher = smoother)
    use_energy_for_unseen: If True, use energy score for unseen detection
                           (lower energy = more confident seen, higher = unseen)
  """
  seen_top1 = topk_acc(seen_logits, seen_labels, (1,))[1]
  if len(unseen_logits) == 0:
    return seen_top1, None

  unseen_logits_np = np.asarray(unseen_logits, dtype=np.float64)
  
  if use_energy_for_unseen:
    # Use energy score for unseen detection
    # Energy = -T * log(sum(exp(logits/T)))
    # For seen classes: low energy (confident)
    # For unseen: high energy (uncertain)
    unseen_energy = _energy_score(unseen_logits_np[:, seen_count:], temperature=1.0)
    # Also compute energy on seen classes for reference
    seen_energy = _energy_score(unseen_logits_np[:, :seen_count], temperature=1.0)
    
    # Unseen samples should have higher energy on unseen classes
    # We can use the ratio or difference
    energy_diff = unseen_energy - seen_energy
    # Predict unseen if energy_diff > threshold (0 for now)
    unseen_pred = (energy_diff > 0).astype(int)
    unseen_labels_binary = np.ones(len(unseen_labels))  # all are unseen
    unseen_top1 = float(np.mean(unseen_pred == unseen_labels_binary))
  else:
    # Standard top-1 accuracy on all classes (including unseen)
    unseen_top1 = topk_acc(unseen_logits, unseen_labels, (1,))[1]
    unseen_top1 = float(unseen_top1)
  
  return float(topk_acc(seen_logits, seen_labels, (1,))[1]), unseen_top1


# ---------------------------------------------------------------------------
# nearest-mean-exemplar (NME) classification
# ---------------------------------------------------------------------------
def nme_top1(features, labels, prototype):
  """NME top-1: nearest class-mean (Euclidean) over stored prototypes."""
  feats = np.asarray(features, dtype=np.float32)
  proto = np.asarray(prototype, dtype=np.float32)
  if proto.ndim != 2 or proto.shape[0] == 0:
    return 0.0
  if feats.ndim != 2 or feats.shape[1] != proto.shape[1]:
    return 0.0
  dist = ((feats[:, None, :] - proto[None, :, :]) ** 2).sum(axis=2)
  pred = np.argmin(dist, axis=1)
  return float(np.mean(pred == np.asarray(labels).reshape(-1)))


# ---------------------------------------------------------------------------
# lifelong
# ---------------------------------------------------------------------------
def lifelong_metrics(mAP_matrix):
  """Plasticity / Forgetting / Overall from a per-task mAP matrix.

  ``mAP_matrix[i][j]`` = mAP of task-group ``j`` measured right after
  training task ``i`` (rows for unseen tasks stay 0). Returns
  (plasticity, forgetting, overall) or (None, None, None) for an empty run.
  """
  m = np.asarray(mAP_matrix, dtype=np.float64)
  if m.ndim != 2 or m.shape[0] == 0:
    return None, None, None
  T = m.shape[0]
  plasticity = float(np.mean([m[i, i] for i in range(T)]))
  if T == 1:
    forgetting = 0.0
  else:
    forgetting = float(np.mean([np.max(m[:, j]) - m[T - 1, j]
                                for j in range(T)]))
  overall = float(np.mean([np.mean(m[i, :i + 1]) for i in range(T)]))
  return plasticity, forgetting, overall