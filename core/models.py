"""Define policy and value function (`v` and `q` functions) as parameterized
neural networks.
""" 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense

import scipy.signal

NEG_INF = -1e9
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class _BaseActor(tf.keras.layers.Layer):
  """Base class for all actors."""
  def call(self, obs):
    """Run policy net on observation tensor to compute the logits."""
    raise NotImplementedError

  @property
  def type(self):
    """Type of action space ("discrete" or "continuous")."""
    raise NotImplementedError


class _ConvActor(_BaseActor):
  """Computes logits given 2D image-like observation."""
  def __init__(self, act_dim, obs_shape, configs):
    """Constructor.

    Args:
      act_dim (int): num of actions (discrete) or action size (continuous). 
      obs_shape (tuple): shape that the `obs` will be reshaped into as input.
      configs (List[dict]): dict mapping Conv2d layer config name to value.
    """
    super().__init__()
    self._act_dim = act_dim
    self._obs_shape = obs_shape
    self._configs = configs

    self._layers = tf.keras.Sequential([Conv2D(**config) for config in configs])

  def call(self, obs):
    """Compute logits.

    Args:
      obs (Tensor): observation tensor of shape [batch, ...].

    Returns:
      logits (Tensor): tensor of shape [batch, act_dim], the logits.
    """
    outputs = self._layers(tf.reshape(obs, self._obs_shape))
    logits = tf.reshape(outputs, [-1, self._act_dim])
    return logits 


class _MLPActor(_BaseActor):
  """Computes mean vector and optionally log_std vector."""
  def __init__(self, obs_dim, configs):
    """Constructor.

    Args:
      obs_dim (int): observation dimensions.
      configs (List[dict]): dict mapping Dense layer config name to value.
    """
    super().__init__()
    self._configs = configs
    self._obs_dim = obs_dim

    self._layers = tf.keras.Sequential([Dense(**config) for config in configs])

  def call(self, obs):
    """Compute mean vector and optionally log_std vector.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim].

    Returns:
      outputs (Tensors): tensor of shape [batch, act_dim (*2)], mean vectors
        and optionally log_std vectors.
    """
    outputs = self._layers(tf.reshape(obs, [-1, self._obs_dim]))
    return outputs


class CategoricalActorMixin(object):
  def get_logp_and_actions(self, logits, act=None, mask=None):
    """Compute log-prob and optionally sample categorical actions. If `act`
    is not None, only return log-probs.

    Args:
      logits (Tensor): tensor of shape [batch, num_act], the logits.
      act (Tensor): (Optional) tensor of shape [batch], integers in
        the range [0, num_act).
      mask (Tensor): (Optional) tensor of shape [batch, num_act], mask
        to be applied to the logits for sampling.

    Returns:
      logp (Tensor): tensor of shape [batch], log prob of actions.
      act (Tensor): (Optional) tensor of shape [batch], sampled actions.
    """
    return_actions = False
    if act is None:
      return_actions = True
      logits_sample = logits
      if mask is not None:
        logits_sample += mask
      act = tf.random.categorical(logits_sample, 1, dtype="int32")[:, 0]

    logp = tf.nn.log_softmax(logits)
    indices = tf.concat([tf.range(tf.size(act), dtype=act.dtype)[
        :, tf.newaxis], tf.reshape(act, (-1, 1))], axis=1)
    logp = tf.gather_nd(logp, indices)

    if return_actions:
      return logp, act
    else:
      return logp

  @property
  def type(self):
    return "discrete"


class DiagonalGaussianActorMixin(object):
  def get_logp_and_actions(self, model_outputs, act=None):
    """Compute log-prob and optionally sample continuous actions. If `act` is
    not None, only return log-probs.

    Args:
      model_outputs (Tensor): tensor of shape [batch, act_dim (*2)], the
        predicted mean vector (and optionally log_std vector).
      act (Tensor): (Optional) tensor of shape [batch, act_dim], actions.

    Returns:
      logp (Tensor): tensor of shape [batch], log prob of actions.
      act (Tensor): (Optional) tensor of shape [batch, act_dim], actions.
    """
    if self._mean_only:
      mean = model_outputs
      log_std = -0.5 * tf.ones_like(mean)
    else:
      mean, log_std = tf.split(model_outputs, 2, axis=-1)
    std = tf.exp(log_std)

    return_actions = False
    if act is None:
      return_actions = True
      act = tf.random.normal(tuple(mean.shape), mean, std)

    pre_sum = -0.5 * (((act - mean) / (tf.exp(log_std) + EPS)) ** 2 +
        2 * log_std + np.log(2 * np.pi))
    logp = tf.reduce_sum(pre_sum, axis=1)
    if return_actions:
      return logp, act
    else:
      return logp

  @property
  def type(self):
    return "continuous"


class CategoricalConvActor(_ConvActor, CategoricalActorMixin):
  def __init__(self, num_act, obs_shape, configs):
    """Constructor.

    Args:
      num_act (int): num of actions.
      obs_shape (tuple): shape that the `obs` will be reshaped into as input.
      configs (List[dict]): dict mapping Conv2d layer config name to value.
    """
    super().__init__(act_dim=num_act, obs_shape=obs_shape, configs=configs)

  def call(self, obs, act=None, mask=None):
    """Compute log-prob and optionally sample categorical actions. If `act` is
    not None, only return log-probs.

    Args:
      obs (Tensor): tensor of shape [batch, ...], observation tensor.
      act (Tensor): (Optional) tensor of shape [batch], integers in the range
        [0, num_act).
      mask (Tensor): (Optional) tensor of shape [batch, num_act], mask to be
        applied to the logits for sampling.

    Returns:
      logp (Tensor): tensor of shape [batch], log-prob of actions.
      act (Tensor): (Optional) tensor of shape [batch], sampled actions.
    """
    logits = super().call(obs)
    if act is None:
      logp, act = self.get_logp_and_actions(logits, act=act, mask=mask)
      return logp, act
    else:
      logp = self.get_logp_and_actions(logits, act=act, mask=mask)
      return logp


class DiagonalGaussianConvActor(_ConvActor, DiagonalGaussianActorMixin):
  def __init__(self, act_dim, obs_shape, configs, mean_only=True):
    """Constructor.

    Args:
      act_dim (int): dimensionality of continuous action space.
      obs_shape (tuple): shape that the `obs` will be reshaped into as input.
      configs (List[dict]): dict mapping Conv2d layer config name to value.
      mean_only (bool): (Optional) whether the policy net computes the mean
        vector only (True), or both mean and logstd vector (False).
    """
    super().__init__(act_dim=act_dim, obs_shape=obs_shape, configs=configs)
    self._mean_only = mean_only

  def call(self, obs, act=None, mask=None):
    """Compute log-prob and optionally sample continuous actions. If `act` is
    not None, only return log-probs.

    Args:
      obs (Tensor): tensor of shape [batch, ...], observation tensor.
      act (Tensor): (Optional) tensor of shape [batch, act_dim], actions.

    Returns:
      logp (Tensor): tensor of shape [batch], log-prob of actions.
      act (Tensor): (Optional) tensor of shape [batch, act_dim], actions.
    """
    model_outputs = super().call(obs)
    if act is None:
      logp, act = self.get_logp_and_actions(model_outputs, act=act)
      return logp, act
    else:
      logp = self.get_logp_and_actions(model_outputs, act=act)
      return logp 


class CategoricalMLPActor(_MLPActor, CategoricalActorMixin):
  def __init__(self, obs_dim, configs):
    """Constructor.

    Args:
      obs_dim (int): observation dimensions.
      configs (List[dict]): dict mapping Dense layer config name to value.
    """
    super().__init__(obs_dim=obs_dim, configs=configs)

  def call(self, obs, act=None, mask=None):
    """Compute log-prob and optionally sample categorical actions. If `act` is
    not None, only return log-probs.

    Args:
      obs (Tensor): tensor of shape [batch, obs_dim], observation tensor.
      act (Tensor): (Optional) tensor of shape [batch], integers in the range
        [0, num_act).

    Returns:
      logp (Tensor): tensor of shape [batch], log-prob of actions.
      act (Tensor): (Optional) tensor of shape [batch], sampled actions.
    """
    logits = super().call(obs)
    if act is None:
      logp, act = self.get_logp_and_actions(logits, act=act, mask=mask)
      return logp, act
    else:
      logp = self.get_logp_and_actions(logits, act=act, mask=mask)
      return logp


class DiagonalGaussianMLPActor(_MLPActor, DiagonalGaussianActorMixin):
  def __init__(self, obs_dim, configs, mean_only=True):
    """Constructor.

    Args:
      obs_dim (int): observation dimensions.
      configs (List[dict]): dict mapping Dense layer config name to value.
      mean_only (bool): (Optional) whether the policy net computes the mean
        vector only (True), or both mean and logstd vector (False).
    """
    super().__init__(obs_dim=obs_dim, configs=configs)
    self._mean_only = mean_only

  def call(self, obs, act=None, mask=None):
    """Compute log-prob and optionally sample continuous actions. If `act` is
    not None, only return log-probs.

    Args:
      obs (Tensor): tensor of shape [batch, obs_dim], observation tensor.
      act (Tensor): (Optional) tensor of shape [batch, act_dim], actions.

    Returns:
      logp (Tensor): tensor of shape [batch], log-prob of actions.
      act (Tensor): (Optional) tensor of shape [batch, act_dim], actions.
    """
    model_outputs = super().call(obs)
    if act is None:
      logp, act = self.get_logp_and_actions(model_outputs, act=act)
      return logp, act
    else:
      logp = self.get_logp_and_actions(model_outputs, act=act)
      return logp


class ConvCriticV(tf.keras.layers.Layer):
  """Critic that computing `V` (state-value) for 2D image-like observations."""
  def __init__(self, obs_shape, configs):
    """Constructor.

    Args:
      obs_shape (tuple): shape that the input observation tensor will be
        reshaped into before passing to the value net.
      configs (List[dict]): dict mapping Conv2d layer config name to value.
    """
    super().__init__()
    self._obs_shape = obs_shape
    self._configs = configs

    self._layers = tf.keras.Sequential([Conv2D(**config) for config in configs])

  def call(self, obs):
    """Compute state values.

    Args:
      obs (Tensor): observation tensor of shape [batch, ...].

    Returns:
      v (Tensor): tensor of shape [batch, 1], the estimated state value.
    """
    v = tf.reshape(self._layers(tf.reshape(obs, self._obs_shape)), (-1, 1))
    return v 


class MLPCriticV(tf.keras.layers.Layer):
  """Critic that computing `v(s)` (state-value) for vector-like observations."""
  def __init__(self, obs_dim, configs):
    """Constructor.

    Args:
      obs_dim (int): observation dimensions.
      configs (List[dict]): dict mapping Dense layer config name to value. 
    """
    super().__init__()
    self._obs_dim = obs_dim
    self._configs = configs
    self._layers = tf.keras.Sequential([Dense(**config) for config in configs])

  def call(self, obs):
    """Compute state values.

    Args:
      obs (Tensor): observation tensor of shape [batch, ...].

    Returns:
      v (Tensor): tensor of shape [batch, 1], the estimated state value.
    """
    v = tf.reshape(self._layers(tf.reshape(obs, [-1, self._obs_dim])), (-1, 1))
    return v


class DeterministicMLPActor(tf.keras.layers.Layer):
  """The deterministic policy `mu(obs)` that output a continuous action."""

  def __init__(self, configs, act_limit=1.):
    """Constructor.

    Args:
      configs (List[dict]): dict mapping Dense layer config name to value. The
        activation of last layer must be "tanh".
      act_limit (float): (Optional) each component of action vector is bounded
        by `-act_limit` and `+act_limit`.
    """
    super().__init__()
    self._act_limit = act_limit
    self._configs = configs

    self._act_dim = configs[-1]["units"]
    assert configs[-1]["activation"] in ("tanh", tf.nn.tanh)
    self._layers = tf.keras.Sequential([Dense(**config) for config in configs])

  def call(self, obs):
    """Computes the action given observation.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim].

    Returns:
      act (Tensor): action tensor of shape [batch, act_dim].
    """
    act = self._act_limit * self._layers(obs)
    return act


class MLPCriticQ(tf.keras.layers.Layer):
  """Critic that computing `q(s, a)` (action-value)."""
  def __init__(self, configs):
    """Constructor.

    Args:
      configs (List[dict]): dict mapping Dense layer config name to value. For
        the last layer the num of filters must be 1, and activation must be None
    """
    super().__init__()
    self._configs = configs
    assert configs[-1]["units"] == 1
    assert "activation" not in configs[-1] or configs[-1]["activation"] is None
    self._layers = tf.keras.Sequential([Dense(**config) for config in configs])

  def call(self, obs, act):
    """Computes q value.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim].
      act (Tensor): action tensor of shape [batch, act_dim].

    Returns:
      q (Tensor): action-value tensor of shape [batch].
    """
    inputs = tf.concat([obs, act], axis=-1)
    q = tf.squeeze(self._layers(inputs), axis=-1)
    return q


class SquashedGaussianActor(tf.keras.layers.Layer):
  """MLP actor that outputs a diagonal Gaussian action with mean vector
  tanh-squashed.
  """
  def __init__(self, configs, act_limit=1.):
    """Constructor.

    Args:
      configs (List[dict]): dict mapping Dense layer config name to value. The
        last two layers compute mean and log_std vector, respectively.
      act_limit (float): (Optional) each component of action vector is bounded
        by `-act_limit` and `+act_limit`.
    """
    super().__init__()
    self._configs = configs
    self._act_limit = act_limit

    self._layers = [Dense(**config) for config in configs] 

  def call(self, obs, sample=True):
    """Computes action and its log-prob.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim].
      sample (bool): whether to sample or return the deterministic action.

    Returns:
      logp (Tensor): tensor of shape [batch], log-prob of actions.
      act (Tensor): (Optional) tensor of shape [batch, act_dim], actions.
    """
    outputs = obs
    for i in range(len(self._layers) - 2):
      outputs = self._layers[i](outputs)
    mean = self._layers[-2](outputs)
    log_std = self._layers[-1](outputs)
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = tf.exp(log_std)

    if sample:
      act = mean + tf.random.normal(tf.shape(mean)) * std
    else:
      act = mean

    pre_sum = -0.5 * (((act - mean) / (tf.exp(log_std) + EPS)) ** 2 +
        2 * log_std + np.log(2 * np.pi))
    logp = tf.reduce_sum(pre_sum, axis=1)
    logp -= tf.reduce_sum(
        2 * (np.log(2) - act - tf.nn.softplus(-2 * act)), axis=-1)
    act = tf.tanh(act) * self._act_limit
    return logp, act

