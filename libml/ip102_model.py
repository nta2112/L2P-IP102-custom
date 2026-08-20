# coding=utf-8
"""JAX adapter exposing the trained L2P model to ``ip102_eval``.

Bridges the PASS-style evaluation interface
(``collect_scores(classes, split, numclass)``, ``prototype``,
``mAP_matrix``) to a flax/pmap L2P model. NumPy/JAX/TF only, no torch.
"""

import functools

import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from libml import input_pipeline


def _collect_step(model, state, batch, original_vit_model,
                  original_vit_params):
  """One pmap step: returns (logits, pre_logits/features, labels)."""
  if original_vit_model is not None:
    original_vit_variables = {"params": original_vit_params}
    original_vit_res = original_vit_model(train=False).apply(
        original_vit_variables, batch["image"], mutable=False)
    cls_features = original_vit_res["pre_logits"]
  else:
    cls_features = None
  variables = {"params": state.optimizer.target}
  variables.update(state.model_state)
  res = model(train=False).apply(
      variables, batch["image"], cls_features=cls_features, mutable=False)
  return res["logits"], res["pre_logits"], batch["label"]


class L2PIP102Model(object):
  """PASS-style model interface backed by a flax/pmap L2P model."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               model,
               state,
               original_vit_model=None,
               original_vit_params=None,
               dm=None,
               num_total_class=25):
    self.config = config
    self.model = model
    self.state = state  # unreplicated TrainState
    self.original_vit_model = original_vit_model
    self.original_vit_params = original_vit_params
    self.dm = dm
    self.total_nc = num_total_class
    self.fg_nc = dm.task_sizes[0]
    self.task_size = dm.task_sizes[1]
    self.numclass = self.fg_nc
    self.prototype = None
    self.mAP_matrix = []
    self._p_collect = jax.pmap(
        functools.partial(
            _collect_step,
            original_vit_model=original_vit_model,
            original_vit_params=original_vit_params),
        axis_name="batch",
        static_broadcasted_argnums=0)

  def set_state(self, state):
    self.state = state

  def _run(self, paths, labels, numclass):
    n = len(paths)
    if n == 0:
      return {
          "logits": np.zeros((0, numclass), dtype=np.float32),
          "features": np.zeros((0, 1), dtype=np.float32),
          "labels": np.zeros((0,), dtype=np.int64),
      }
    ds = input_pipeline.create_ip102_eval_ds(self.config, paths, labels)
    state = flax_utils.replicate(self.state)
    logits_all, feats_all, labels_all = [], [], []
    for batch in ds:
      batch = jax.tree_map(np.asarray, batch)
      logits, feats, labs = self._p_collect(self.model, state, batch)
      logits_all.append(np.asarray(logits).reshape(-1, logits.shape[-1]))
      feats_all.append(np.asarray(feats).reshape(-1, feats.shape[-1]))
      labels_all.append(np.asarray(labs).reshape(-1))
    logits = np.concatenate(logits_all, axis=0)[:n]
    feats = np.concatenate(feats_all, axis=0)[:n]
    labs = np.concatenate(labels_all, axis=0)[:n]
    return {
        "logits": logits[:, :numclass],
        "features": feats,
        "labels": labs.astype(np.int64),
    }

  def collect_scores(self, classes, split="val", numclass=None):
    """Scores of examples from ``classes`` on the first ``numclass`` columns."""
    if numclass is None:
      numclass = self.numclass
    paths, labels = self.dm.get_paths_by_class(classes, split=split)
    return self._run(paths, labels, numclass)

  def update_prototype(self, seen_count=None):
    """Class-mean train features for the first ``seen_count`` classes (NME)."""
    if seen_count is None:
      seen_count = self.numclass
    seen = list(range(seen_count))
    s = self.collect_scores(seen, split="train", numclass=seen_count)
    feat_dim = s["features"].shape[1]
    proto = np.zeros((self.total_nc, feat_dim), dtype=np.float32)
    for c in seen:
      mask = s["labels"] == c
      if mask.sum() > 0:
        proto[c] = s["features"][mask].mean(axis=0)
    self.prototype = proto
    return proto