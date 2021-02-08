import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
import math

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
  """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
      Args:
          logits: logits distribution shape (..., vocabulary size)
          top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
          top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
              whose total probability mass is greater than or equal to the threshold top_p.
              In practice, we select the highest probability tokens whose cumulative probability mass exceeds
              the threshold top_p.
          threshold: a minimal threshold to keep logits
  """
  top_k = min(top_k, logits.size(-1))
  if top_k > 0:
    # Remove all tokens with a probability less than the last token in the top-k tokens
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value

  if top_p > 0.0:
    # Compute cumulative probabilities of sorted tokens
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probabilities > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Back to unsorted indices and set them to -infinity
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value

  indices_to_remove = logits < threshold
  logits[indices_to_remove] = filter_value

  return logits


class SequentialDistributedSampler(Sampler):
  """
  Distributed Sampler that subsamples indicies sequentially,
  making it easier to collate all results at the end.

  Even though we only use this sampler for eval and predict (no training),
  which means that the model params won't have to be synced (i.e. will not hang
  for synchronization even if varied number of forward passes), we still add extra
  samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
  to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
  """

  def __init__(self, dataset, num_replicas=None, rank=None):
    if num_replicas is None:
      if not torch.distributed.is_available():
        raise RuntimeError("Requires distributed package to be available")
      num_replicas = torch.distributed.get_world_size()
    if rank is None:
      if not torch.distributed.is_available():
        raise RuntimeError("Requires distributed package to be available")
      rank = torch.distributed.get_rank()
    self.dataset = dataset
    self.num_replicas = num_replicas
    self.rank = rank
    self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
    self.total_size = self.num_samples * self.num_replicas

  def __iter__(self):
    indices = list(range(len(self.dataset)))

    # add extra samples to make it evenly divisible
    indices += indices[: (self.total_size - len(indices))]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
    assert len(indices) == self.num_samples

    return iter(indices)

  def __len__(self):
    return self.num_samples

def checkeq(num, target_list, eps=1e-2):
  if type(target_list) == int:
    target_list = [target_list]
  for n in target_list:
    if abs(num - n) < eps:
      return True
  return False

class RunningAverage(object):
  def __init__(self, alpha: float = 0.98):
    self.alpha = alpha
    self.reset()

  def reset(self):
    self._value = None

  def add(self, now_value):
    if self._value is None:
      self._value = now_value
    else:
      self._value = self._value * self.alpha + (1.0 - self.alpha) * now_value
    return self._value