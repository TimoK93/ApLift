import torch
import numpy as np


def dict_to_torch(dict_of_tensors):
    """ Converts a dict to torch arrays """
    for key in dict_of_tensors:
        if isinstance(dict_of_tensors[key], np.ndarray):
            dict_of_tensors[key] = torch.from_numpy(dict_of_tensors[key])
    return dict_of_tensors


def to_numpy(obj):
    """ Transforms the object to numpy arrays"""
    if type(obj) == dict:
        for key, item in obj.items():
            obj[key] = to_numpy(item)
    elif type(obj) == torch.Tensor:
        if obj.shape[0] == 1:
            obj = obj.detach().cpu().numpy()[0]
        else:
            obj = obj.detach().cpu().numpy()
    else:
        try:
            obj = np.array(obj)
        except Exception as e:
            raise TypeError("Wrong type: ", type(obj))
    return obj
