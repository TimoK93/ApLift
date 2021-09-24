from itertools import chain

import numpy as np
import torch
from abc import ABC, abstractmethod

from functools import update_wrapper, partial


class Decorator(ABC):
  def __init__(self, f):
    self.func = f
    update_wrapper(self, f, updated=[])  # updated=[] so that 'self' attributes are not overwritten

  @abstractmethod
  def __call__(self, *args, **kwargs):
    pass

  def __get__(self, instance, owner):
    new_f = partial(self.__call__, instance)
    update_wrapper(new_f, self.func)
    return new_f


def dict_to_device(dict_of_tensors, device):
  for key in dict_of_tensors:
    if isinstance(dict_of_tensors[key], torch.Tensor):
      dict_of_tensors[key] = dict_of_tensors[key].to(device)
  return dict_of_tensors


def to_tensor(x, dtype=None, device=None):
  dtype = dtype or torch.float32
  if isinstance(x, np.ndarray) or np.isscalar(x):
    res = torch.from_numpy(np.array(x)).to(dtype)
    if device:
        return res.to(device)
    return res
  else:
    return x


def to_numpy(x):
  if isinstance(x, torch.Tensor):
    return x.cpu().detach().numpy()
  else:
    return x


def iterator_repeating_last(iterable):
  el = None
  for el in iterable:
    yield el
  while True:
    yield el


# noinspection PyPep8Naming
class multiple_to_numpy_and_back(Decorator):
  def __call__(self, *args, **kwargs):
    dtypes = [x.dtype if isinstance(x, torch.Tensor) else None for x in args]
    devices = [x.device if isinstance(x, torch.Tensor) else None for x in args]
    new_args = [to_numpy(arg) for arg in args]
    res = self.func(*new_args)
    out = [to_tensor(x, dtype, device) for x, dtype, device in zip(res,
                                                                    iterator_repeating_last(dtypes),
                                                                    iterator_repeating_last(devices))]
    return tuple(out)


# noinspection PyPep8Naming
class simple_multiple_to_numpy_and_back(Decorator):
  def __call__(self, *args, **kwargs):
    devices = [x.device for x in args if isinstance(x, torch.Tensor)]
    new_args = [to_numpy(arg) for arg in args]
    res = self.func(*new_args)
    out = [to_tensor(x, dtype, device) for x, dtype, device in zip(res,
                                                                    iterator_repeating_last([None]),
                                                                    iterator_repeating_last(devices))]
    return tuple(out)


# noinspection PyPep8Naming
class input_to_tensors(Decorator):
  def __call__(self, *args, **kwargs):
    new_args = [to_tensor(arg) for arg in args]
    new_kwargs = {key: to_tensor(value) for key, value in kwargs.items()}
    return self.func(*new_args, **new_kwargs)


# noinspection PyPep8Naming
class output_to_tensors(Decorator):
  def __call__(self, *args, **kwargs):
    outputs = self.func(*args, **kwargs)
    if isinstance(outputs, np.ndarray):
      return to_tensor(outputs)
    if isinstance(outputs, tuple):
      new_outputs = tuple([to_tensor(item) for item in outputs])
      return new_outputs
    return outputs


# noinspection PyPep8Naming
class input_to_numpy(Decorator):
  def __call__(self, *args, **kwargs):
    new_args = [to_numpy(arg) for arg in args]
    new_kwargs = {key: to_numpy(value) for key, value in kwargs.items()}
    return self.func(*new_args, **new_kwargs)


# noinspection PyPep8Naming
class output_to_numpy(Decorator):
  def __call__(self, *args, **kwargs):
    outputs = self.func(*args, **kwargs)
    if isinstance(outputs, torch.Tensor):
      return to_numpy(outputs)
    if isinstance(outputs, tuple):
      new_outputs = tuple([to_numpy(item) for item in outputs])
      return new_outputs
    return outputs


# noinspection PyPep8Naming
class none_if_missing_arg(Decorator):
  def __call__(self, *args, **kwargs):
    for arg in chain(args, kwargs.values()):
      if arg is None:
        return None

    return self.func(*args, **kwargs)