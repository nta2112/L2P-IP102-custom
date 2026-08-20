"""Unit tests for the IP102 metrics module (numpy-only).

Run:  python tests/test_ip102_metrics.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from libml.ip102_metrics import (lifelong_metrics, macro_map, nme_top1,  # noqa: E402
                                 ood_metrics, retrieval_metrics, topk_acc)


def test_perfect_retrieval():
  n = 10
  labels = np.arange(n)
  logits = np.zeros((n, n))
  for i in range(n):
    logits[i, i] = 1.0
  res = retrieval_metrics(logits, labels)
  assert res['R@1'] == 1.0
  assert res['R@5'] == 1.0
  assert res['R@10'] == 1.0
  assert res['mAP'] == 1.0
  assert topk_acc(logits, labels, (1,))[1] == 1.0
  assert macro_map(logits, labels) == 1.0
  print('ok  test_perfect_retrieval (R@1=R@5=R@10=mAP=1.0)')


def test_ood_perfect():
  auroc, fpr95 = ood_metrics([0.9, 1.0], [0.1, 0.2])
  assert auroc == 1.0
  assert fpr95 == 0.0
  auroc, fpr95 = ood_metrics([1.0, 0.95, 0.8], [0.3, 0.2, 0.1])
  assert auroc == 1.0
  assert fpr95 == 0.0
  print('ok  test_ood_perfect (AUROC=1.0, FPR95=0.0)')


def test_ood_none_when_no_unknown():
  auroc, fpr95 = ood_metrics([0.9, 1.0], [])
  assert auroc is None and fpr95 is None
  auroc, fpr95 = ood_metrics([], [0.1, 0.2])
  assert auroc is None and fpr95 is None
  print('ok  test_ood_none_when_no_unknown (all classes seen -> None)')


def test_ood_realistic_separation():
  rng = np.random.RandomState(0)
  known = rng.normal(1.0, 0.1, 200)
  unknown = rng.normal(0.2, 0.1, 200)
  auroc, fpr95 = ood_metrics(known, unknown)
  assert auroc > 0.99
  assert fpr95 < 0.05
  print('ok  test_ood_realistic_separation (auroc=%.4f fpr95=%.4f)'
        % (auroc, fpr95))


def test_lifelong_hand_computed():
  m = np.array([[1.0, 0.0, 0.0],
                [0.8, 0.9, 0.0],
                [0.6, 0.7, 0.8]])
  p, f, o = lifelong_metrics(m)
  # plasticity = (1.0 + 0.9 + 0.8) / 3
  assert abs(p - 0.9) < 1e-9
  # forgetting = mean(max_j - last_j) = (0.4 + 0.2 + 0.0) / 3
  assert abs(f - 0.2) < 1e-9
  # overall = mean of row means (1.0, 0.85, 0.7)
  assert abs(o - 0.85) < 1e-9
  p, f, o = lifelong_metrics(np.ones((3, 3)))
  assert p == 1.0 and f == 0.0 and o == 1.0
  p, f, o = lifelong_metrics(np.array([[0.5]]))
  assert p == 0.5 and f == 0.0 and o == 0.5
  print('ok  test_lifelong_hand_computed (plasticity/forgetting/overall)')


def test_nme_perfect():
  feats = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
  proto = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
  labels = np.array([0, 1, 2])
  assert nme_top1(feats, labels, proto) == 1.0
  assert nme_top1(feats, np.array([0, 0, 0]), proto[:1]) == 1.0
  print('ok  test_nme_perfect (NME top-1 = 1.0)')


def test_recall_seen_unseen():
  from libml.ip102_metrics import recall_at_1_seen_unseen
  seen_logits = np.array([[1.0, 0.0], [0.0, 1.0]])
  seen_labels = np.array([0, 1])
  # unseen full logits over 4 classes: true unseen class = 2 / 3
  unseen_logits = np.array([[0.0, 0.1, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]])
  unseen_labels = np.array([2, 3])
  rec_seen, rec_unseen = recall_at_1_seen_unseen(
      seen_logits, seen_labels, unseen_logits, unseen_labels, seen_count=2)
  assert rec_seen == 1.0
  assert rec_unseen == 1.0
  rec_seen, rec_unseen = recall_at_1_seen_unseen(
      seen_logits, seen_labels, np.zeros((0, 4)), np.zeros((0,), dtype=int),
      seen_count=2)
  assert rec_seen == 1.0 and rec_unseen is None
  print('ok  test_recall_seen_unseen')


if __name__ == '__main__':
  test_perfect_retrieval()
  test_ood_perfect()
  test_ood_none_when_no_unknown()
  test_ood_realistic_separation()
  test_lifelong_hand_computed()
  test_nme_perfect()
  test_recall_seen_unseen()
  print('\nALL METRICS TESTS PASSED')