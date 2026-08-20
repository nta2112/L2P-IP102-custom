# coding=utf-8
"""Entry point for L2P on IP102 (open-world lifelong retrieval).

Provides ``run_train(model, max_tasks, memory_size)`` used by the Kaggle
launcher notebook and a CLI fallback. Mirrors the iCaRL / PASS launcher
convention: max_tasks=1 runs a fast smoke test, max_tasks=0 runs all tasks.
"""

import importlib.util
import os
import urllib.request

# Kaggle dual-T4 is PCIe only (no NVLink); NCCL P2P all-reduce fails with
# "ncclAllReduce ... invalid argument". Must be set before JAX initializes
# any NCCL communicator.
os.environ.setdefault('NCCL_P2P_DISABLE', '1')
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_DEBUG', 'WARN')

# TensorFlow and JAX cannot both load their own CUDA runtimes into one
# process (jax issue #34918 / #17497): whichever imports first wins, and the
# other's NCCL all-reduce then fails with "invalid argument". TF is only used
# here for CPU-side data loading, so run JAX on a single GPU where pmap needs
# no NCCL at all (jax issue #15628). Override by setting CUDA_VISIBLE_DEVICES.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from absl import logging
import jax
import ml_collections
import tensorflow as tf

# Keep TensorFlow away from the GPUs so they are available to JAX. This must
# happen before any TF runtime is initialized (e.g. by importing modules below
# that pull in tensorflow_datasets / clu), otherwise set_visible_devices raises
# "Visible devices cannot be modified after being initialized".
try:
  tf.config.experimental.set_visible_devices([], "GPU")
except RuntimeError as e:
  logging.warning("Could not hide GPUs from TensorFlow: %s", e)

from libml import ip102_data
import train_continual

VIT_B16_URL = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"
CONFIG_MODULE = "configs.ip102_l2p"


def _load_config(config_path=None):
  if config_path:
    spec = importlib.util.spec_from_file_location("ip102_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
  else:
    mod = importlib.import_module(CONFIG_MODULE)
  return mod.get_config()


def _ensure_init_checkpoint(config: ml_collections.ConfigDict,
                            data_root=None, output_dir=None):
  """Download the ViT-B_16 pretrained weights if not present.

  Kaggle's /kaggle/input is read-only, so the checkpoint is stored next to
  the (writable) output directory / current directory.
  """
  path = config.get("init_checkpoint")
  if path and os.path.exists(str(path)):
    return str(path)
  base = None
  if output_dir:
    base = os.path.abspath(output_dir)
  elif data_root and os.access(os.path.dirname(data_root), os.W_OK):
    base = os.path.dirname(data_root)
  base = base or os.getcwd()
  os.makedirs(base, exist_ok=True)
  ckpt = os.path.join(base, "ViT-B_16.npz")
  if not os.path.exists(ckpt):
    logging.info("Downloading ViT-B_16 pretrained weights to %s", ckpt)
    urllib.request.urlretrieve(VIT_B16_URL, ckpt)
  return ckpt


def run_train(model="L2P",
              max_tasks=0,
              memory_size=0,
              config_path=None,
              data_root=None,
              output_dir="output_ip102",
              num_epochs=None,
              seed=None):
  """Trains L2P on IP102 and writes results.csv / history.json.

  Args:
    model: Model name. Only "L2P" is supported.
    max_tasks: Number of tasks to train (0 = all). 1 = smoke test.
    memory_size: Caps the number of training images per class (0 = all).
    config_path: Optional path to a get_config() module.
    data_root: Optional dataset root. Auto-discovered otherwise.
    output_dir: Directory for results.
    num_epochs: Overrides config.num_epochs when given.
    seed: Overrides config.seed when given.

  Returns:
    Path to the written results.csv.
  """
  if model != "L2P":
    raise NotImplementedError("Chua ho tro model %r, chi ho tro L2P." % model)

  logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())

  config = _load_config(config_path)
  if max_tasks and max_tasks > 0:
    config.continual.num_tasks = int(max_tasks)
    config.continual.num_classes_per_task = 7
  if memory_size and memory_size > 0:
    config.continual.memory_size = int(memory_size)
  if num_epochs:
    config.num_epochs = int(num_epochs)
  if seed is not None:
    config.seed = int(seed)

  config.init_checkpoint = _ensure_init_checkpoint(config, data_root=data_root,
                                                   output_dir=output_dir)
  if data_root:
    os.environ["IP102_DATA_ROOT"] = os.path.abspath(data_root)

  # dataset sanity check (0 missing images) before training
  dm = ip102_data.get_data_manager(
      "ip102", seed=config.continual.get("rand_seed", 1993))
  report = dm.verify()
  for s in report.get("available_splits", dm.available_splits):
    if report[s]["missing_images"]:
      raise RuntimeError("IP102 %s con anh thieu: %d" %
                         (s, report[s]["missing_images"]))

  train_continual.train_and_evaluate(config, os.path.abspath(output_dir))
  results_path = os.path.join(os.path.abspath(output_dir), "results.csv")
  logging.info("Done. Results written to %s", results_path)
  return results_path


def main(argv=None):
  del argv
  run_train()


if __name__ == "__main__":
  main()