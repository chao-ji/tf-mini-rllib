import numpy as np
import scipy.signal


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PolicyGradientBuffer(object):
  """Buffer that caches episode experience data for training policy gradient
  actors and critics.
  """
  def __init__(
      self, obs_shape, act_shape, size, gamma=0.99, lam=0.95,
    act_type="discrete"):
    """Constructor.

    Args:
      obs_shape (tuple): the shape of a single observation tensor.
      act_shape (tuple): the shape of a single action tensor.
      size (int): the size of the buffer. 
      gamma (float): discount factor
      lam (float): the lambda coefficient for GAE.
      act_type (str): type of action space ("discrete" or "continuous").
    """
    self._obs_buf = np.zeros((size, *obs_shape), dtype="float32")
    if act_type == "discrete":
      self._act_buf = np.zeros(size, dtype="int32")
    else:
      self._act_buf = np.zeros((size, *act_shape), dtype="float32")

    self._rew_buf = np.zeros(size, dtype="float32")
    self._val_buf = np.zeros(size, dtype="float32")

    self._adv_buf = np.zeros(size, dtype="float32")
    self._ret_buf = np.zeros(size, dtype="float32")
    self._logp_buf = np.zeros(size, dtype="float32")

    self._gamma = gamma
    self._lam = lam
    self._ptr = 0
    self._path_start_idx = 0
    self._max_size = size

  def store(self, obs, act, rew, val, logp):
    """Cache a experience data for a single agent-interaction step.

    Args:
      obs (Tensor): the observation tensor.
      act (Tensor): the action tensor. 
      rew (float): the immediate rewawrd.
      val (float): the estimated state value.
      logp (float): the log-prob of the action computed by the current policy.
    """
    # make sure the buffer still as room
    assert self._ptr < self._max_size
    self._obs_buf[self._ptr] = obs
    self._act_buf[self._ptr] = act
    self._rew_buf[self._ptr] = rew
    self._val_buf[self._ptr] = val
    self._logp_buf[self._ptr] = logp

    self._ptr += 1

  def finish_path(self, last_val=0):
    """Computes derived data the when an episode has just ended.

    Args:
      last_val (float): the estimated state value for the last step in episode.
    """
    path_slice = slice(self._path_start_idx, self._ptr)
    rews = np.append(self._rew_buf[path_slice], last_val)
    vals = np.append(self._val_buf[path_slice], last_val)

    # computes GAE-Lambda advantage
    deltas = rews[:-1] + self._gamma * vals[1:] - vals[:-1]
    self._adv_buf[path_slice] = discount_cumsum(deltas, self._gamma * self._lam)

    # computes rewards-to-go as the target for the critic
    self._ret_buf[path_slice] = discount_cumsum(rews, self._gamma)[:-1]

    # update the index for the next episode
    self._path_start_idx = self._ptr

  def get(self):
    """Retrieve cached data when the buffer is full.

    Returns:
      obs (Tensor): observation tensor of shape [batch, ...].
      act (Tensor): action tensor of shape [batch] for discrete actions, or
        [batch, act_dim] for continuous actions.
      ret (Tensor): the return tensor of shape [batch].
      adv (Tensor): the advantage tensor of shape [batch].
      logp_old (Tensor): tensor of shape [batch], the cached log-prob of action
        computed by the policy before being updated.
    """
    # make sure buffer is full
    assert self._ptr == self._max_size
    self._ptr, self._path_start_idx = 0, 0

    # normalize advantage values to zero-mean and unit variance
    adv_mean, adv_std = self._adv_buf.mean(), self._adv_buf.std()
    self._adv_buf = (self._adv_buf - adv_mean) / adv_std
    return self._obs_buf, self._act_buf, self._ret_buf, self._adv_buf, self._logp_buf


class ReplayBuffer(object):
  def __init__(self, obs_dim, act_dim, size=int(1e6)):
    self._obs_buf = np.zeros((size, obs_dim), dtype="float32")
    self._obs2_buf = np.zeros((size, obs_dim), dtype="float32")
    self._act_buf = np.zeros((size, act_dim), dtype="float32")

    self._rew_buf = np.zeros(size, dtype="float32")
    self._done_buf = np.zeros(size, dtype="float32")
    self._ptr = 0
    self._size = 0
    self._max_size = size

  def store(self, obs, act, rew, next_obs, done):
    self._obs_buf[self._ptr] = obs
    self._obs2_buf[self._ptr] = next_obs
    self._act_buf[self._ptr] = act
    self._rew_buf[self._ptr] = rew
    self._done_buf[self._ptr] = done
    self._ptr = (self._ptr + 1) % self._max_size
    self._size = min(self._size + 1, self._max_size)

  def sample_batch(self, batch_size=32):
    indices = np.random.randint(0, self._size, size=batch_size)
    batch_obs = self._obs_buf[indices]
    batch_obs2 = self._obs2_buf[indices]
    batch_act = self._act_buf[indices]
    batch_rew = self._rew_buf[indices]
    batch_done = self._done_buf[indices]
    return batch_obs, batch_obs2, batch_act, batch_rew, batch_done


