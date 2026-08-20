"""Re-install the deprecated ``flax.optim`` API for the L2P training code.

Modern flax (>=0.6.0) removed ``flax.optim``, so ``from flax import optim``
fails with ``AttributeError: module 'flax' has no attribute 'optim'``. The
original L2P code relies on that API, so we load the trimmed copy vendored
under ``flax_optim/`` (from flax 0.5.3) and register it as ``flax.optim``.
Because the vendored package is registered with ``__package__ == 'flax.optim'``,
its relative imports (``from .. import struct`` etc.) resolve against the
installed modern ``flax`` package (struct / serialization / traverse_util /
jax_utils / core all still exist there).

Usage::

    import flax
    from libml import flax_optim_compat  # installs flax.optim
    from flax import optim  # now works
"""

import importlib.machinery
import importlib.util
import os
import sys

import flax


def install():
  """Idempotently register the vendored flax.optim under the flax namespace."""
  if hasattr(flax, 'optim'):
    return
  pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'flax_optim')
  loader = importlib.machinery.SourceFileLoader(
      'flax.optim', os.path.join(pkg_dir, '__init__.py'))
  spec = importlib.util.spec_from_loader('flax.optim', loader, is_package=True)
  spec.submodule_search_locations = [pkg_dir]
  mod = importlib.util.module_from_spec(spec)
  sys.modules['flax.optim'] = mod
  loader.exec_module(mod)
  flax.optim = mod


install()