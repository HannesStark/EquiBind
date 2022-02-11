
import random
from argparse import Namespace
from collections import MutableMapping
from typing import Dict, Any
from joblib import Parallel, delayed, cpu_count
import torch
import numpy as np
import dgl
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from commons.logger import log


def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
  """

  Parallel map using joblib.

  Parameters
  ----------
  pickleable_fn : callable
      Function to map over data.
  data : iterable
      Data over which we want to parallelize the function call.
  n_jobs : int, optional
      The maximum number of concurrently running jobs. By default, it is one less than
      the number of CPUs.
  verbose: int, optional
      The verbosity level. If nonzero, the function prints the progress messages.
      The frequency of the messages increases with the verbosity level. If above 10,
      it reports all iterations. If above 50, it sends the output to stdout.
  kwargs
      Additional arguments for :attr:`pickleable_fn`.

  Returns
  -------
  list
      The i-th element of the list corresponds to the output of applying
      :attr:`pickleable_fn` to :attr:`data[i]`.
  """
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
  )

  return results

def seed_all(seed):
    if not seed:
        seed = 0

    log("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_random_indices(length, seed=123):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def flatten_dict(params: Dict[Any, Any], delimiter: str = '/') -> Dict[str, Any]:
    """
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.
    Examples:
        flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    for d in _dict_generator(value, prefixes + [key]):
                        yield d
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    dictionary = {delimiter.join(keys): val for *keys, val in _dict_generator(params)}
    for k in dictionary.keys():
        # convert relevant np scalars to python types first (instead of str)
        if isinstance(dictionary[k], (np.bool_, np.integer, np.floating)):
            dictionary[k] = dictionary[k].item()
        elif type(dictionary[k]) not in [bool, int, float, str, torch.Tensor]:
            dictionary[k] = str(dictionary[k])
    return dictionary




def tensorboard_gradient_magnitude(optimizer: torch.optim.Optimizer, writer: SummaryWriter, step, param_groups=[0]):
    for i, param_group in enumerate(optimizer.param_groups):
        if i in param_groups:
            all_params = []
            for params in param_group['params']:
                if params.grad != None:
                    all_params.append(params.grad.view(-1))
            writer.add_scalar(f'gradient_magnitude_param_group_{i}', torch.cat(all_params).abs().mean(),
                              global_step=step)

def move_to_device(element, device):
    '''
    takes arbitrarily nested list and moves everything in it to device if it is a dgl graph or a torch tensor
    :param element: arbitrarily nested list
    :param device:
    :return:
    '''
    if isinstance(element, list):
        return [move_to_device(x, device) for x in element]
    else:
        return element.to(device) if isinstance(element,(torch.Tensor, dgl.DGLGraph)) else element

def list_detach(element):
    '''
    takes arbitrarily nested list and detaches everyting from computation graph
    :param element: arbitrarily nested list
    :return:
    '''
    if isinstance(element, list):
        return [list_detach(x) for x in element]
    else:
        return element.detach()

def concat_if_list(tensor_or_tensors):
    return torch.cat(tensor_or_tensors) if isinstance(tensor_or_tensors, list) else tensor_or_tensors

def write_strings_to_txt(strings: list, path):
    # every string of the list will be saved in one line
    textfile = open(path, "w")
    for element in strings:
        textfile.write(element + "\n")
    textfile.close()

def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]