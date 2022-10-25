# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy import utils.
"""

from __future__ import annotations

import builtins
import contextlib
import dataclasses
import functools
import importlib
import time
import types
from typing import Any, Callable, Iterator, Optional, Tuple

from tensorflow_datasets.core.tf_compat import ensure_tf_version

Callback = Callable[..., None]


@dataclasses.dataclass
class LazyModule:
  """Module loaded lazily during first call."""

  module_name: str
  module: Optional[types.ModuleType] = None
  fromlist: Optional[Tuple[str, ...]] = ()
  error_callback: Optional[Callback] = None
  success_callback: Optional[Callback] = None

  @classmethod
  @functools.lru_cache(maxsize=None)
  def from_cache(cls, **kwargs):
    """Factory to cache all instances of module.

    Note: The cache is global to all instances of the
    `lazy_imports` context manager.

    Args:
      **kwargs: Init kwargs

    Returns:
      New object
    """
    return cls(**kwargs)

  def __getattr__(self, name: str) -> Any:
    if self.fromlist and name == self.fromlist[0]:
      module_name = f"{self.module_name}.{name}"
      return self.from_cache(
          module_name=module_name, fromlist=self.fromlist[1:])
    if self.module is None:  # Load on first call
      try:
        start_import_time = time.time()
        self.module = importlib.import_module(self.module_name)
        import_time_ms = int((time.time() - start_import_time) * 1000)
        if self.success_callback is not None:
          self.success_callback(
              import_time_ms=import_time_ms,
              module=self.module,
              module_name=self.module_name)
      except ImportError as exception:
        if self.error_callback is not None:
          self.error_callback(exception=exception, module_name=self.module_name)
        raise exception
    return getattr(self.module, name)


@contextlib.contextmanager
def lazy_imports(error_callback: Optional[Callback] = None,
                 success_callback: Optional[Callback] = None) -> Iterator[None]:
  """Context Manager which lazy loads packages.

  Their import is not executed immediately, but is postponed to the first
  call of one of their attributes.

  Warning:

  - `import x.y.z` and all its variants are possible.
  - The syntax `from ... import ...` is not implemented yet and will fail.

  Usage:

  ```python
  from tensorflow_datasets.core import utils

  with utils.lazy_imports():
    import tensorflow as tf
  ```

  Args:
    error_callback: a callback to trigger each time one of the imports fails. It
      takes as argument a kwargs containing: - exception: the exception that was
      raised after the error - module_name: the name of the imported module
    success_callback: a callback to trigger each time of the imports succeed. It
      takes as argument a kwargs containing: - import_time_ms: the import time
      (in milliseconds) - module: the imported module - module_name: the name of
      the imported module

  Yields:
    None
  """
  # Need to mock `__import__` (instead of `sys.meta_path`, as we do not want
  # to modify the `sys.modules` cache in any way)
  original_import = builtins.__import__
  try:
    builtins.__import__ = functools.partial(
        _lazy_import,
        error_callback=error_callback,
        success_callback=success_callback)
    yield
  finally:
    builtins.__import__ = original_import


def _lazy_import(
    name: str,
    globals_=None,
    locals_=None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
    *,
    error_callback: Optional[Callback],
    success_callback: Optional[Callback],
):
  """Mock of `builtins.__import__`."""
  del globals_, locals_  # Unused

  if level:
    raise ValueError(f"Relative import statements not supported ({name}).")

  if not fromlist:
    # import x.y.z
    # import x.y.z as z
    # In that case, Python would import the entirety of `x`, so we do the same.
    root_name = name.split(".")[0]
    return LazyModule.from_cache(
        module_name=root_name,
        error_callback=error_callback,
        success_callback=success_callback)
  # from x.y.z import a, b
  return LazyModule.from_cache(
      module_name=name,
      fromlist=fromlist,
      error_callback=error_callback,
      success_callback=success_callback)


def tf_error_callback():
  print("\n\n***************************************************************")
  print("Failed to import TensorFlow. Please note that TensorFlow is not "
        "installed by default when you install TFDS. This allow you "
        "to choose to install either `tf-nightly` or `tensorflow`. "
        "Please install the most recent version of TensorFlow, by "
        "following instructions at https://tensorflow.org/install.")
  print("***************************************************************\n\n")


def tf_success_callback(**kwargs):
  ensure_tf_version(kwargs["module"])


with lazy_imports(
    error_callback=tf_error_callback, success_callback=tf_success_callback):
  import tensorflow as tf  # pylint: disable=g-import-not-at-top,unused-import

tensorflow = tf
