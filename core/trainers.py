import sys

import tensorflow as tf
import numpy as np


class VPGTrainer(object):
  """Trainer for Vanilla Policy Gradient actor-critic.

  The training loop is divided into `num_iters`, where in each iteration
  we execute the following steps:
  1. Simulate the agent-env interactions to collect `batch` training examples.
  2. Train the actor and critic using the collected data.
  """
  def __init__(
      self, actor, critic, actor_optim, critic_optim, num_steps_critic=80,
      num_iters=1500, batch_size=4000, max_ep_len=1000, log_freq=10,
      ckpt_freq=-1, ckpt_prefix="pg"
    ):
    """Constructor.

    Args:
      actor (Actor): the actor object.
      critic (Critic): the critic object.
      actor_optim: the optimizer for training actor weights.
      critic_optim: the optimizer for training critic weights.
      num_steps_critic (int): num of backward passes to train the critic.
      num_iters (int): total number of iterations in the training loop.
      batch_size (int): num of training examples i.e. (obs, act, ret, adv)
        tuples used in a single update step.
      max_ep_len (int): maximum num of steps the environment will be simulated.
      log_freq (int): print log info every `log_freq` iterations.
      ckpt_freq (int): checkpointt will be saved save `ckpt_freq` iterations. If
        equals -1, no checkpoints will be saved.
      ckpt_prefix (str): the string as the prefix to checkpoint file names. 
    """
    self._actor = actor
    self._critic = critic
    self._actor_optim = actor_optim
    self._critic_optim = critic_optim
    self._num_steps_critic = num_steps_critic
    self._num_iters = num_iters
    self._batch_size = batch_size
    self._max_ep_len = max_ep_len
    self._log_freq = log_freq
    self._ckpt_freq = ckpt_freq
    self._ckpt_prefix = ckpt_prefix

  def _train_step(self, obs, act, ret, adv, _not_used=None):
    """Train the actor and critic using the collected episode data.

    Args:
      obs (Tensor): observation tensor of shape [batch, ...].
      act (Tensor): action tensor of shape [batch] for discrete actions, or
        [batch, act_dim] for continuous actions.
      ret (Tensor): the return tensor of shape [batch].
      adv (Tensor): the advantage tensor of shape [batch].
    """
    with tf.GradientTape() as tape:
      actor_loss = self._compute_actor_loss(obs, act, adv)

    grads = tape.gradient(actor_loss, self._actor.trainable_variables)
    self._actor_optim.apply_gradients(
        zip(grads, self._actor.trainable_variables))

    for _ in range(self._num_steps_critic):
      with tf.GradientTape() as tape:
        critic_loss = self._compute_critic_loss(obs, ret)

      grads = tape.gradient(critic_loss, self._critic.trainable_variables)
      self._critic_optim.apply_gradients(
          zip(grads, self._critic.trainable_variables))

  def _compute_actor_loss(self, obs, act, adv):
    """Computes the actor loss.

    Args:
      obs (Tensor): observation tensor of shape [batch, ...].
      act (Tensor): action tensor of shape [batch] for discrete actions, or
        [batch, act_dim] for continuous actions.
      adv (Tensor): the advantage tensor of shape [batch].

    Returns:
      loss (Tensor): tensor of shape [batch], the loss for actor.
    """
    log_prob = self._actor(obs, act)
    loss = -tf.reduce_mean(log_prob * adv)
    return loss

  def _compute_critic_loss(self, obs, ret):
    """Computes the critic loss.

    Args:
      obs (Tensor): observation tensor of shape [batch, ...].
      ret (Tensor): the advantage tensor of shape [batch].

    Returns:
      loss (Tensor): tensor of shape [batch], the loss for critic.
    """
    loss = tf.reduce_mean((self._critic(obs)[:, 0] - ret) ** 2)
    return loss

  def train(self, env, buf, ckpt, mask_fn=None, eval_agent_fn=None):
    """Train both the actor and critic.

    Args:
      env: the environment object.
      buf: the data buffer.
      ckpt: the Checkpoint object.
      mask_fn: (Optional) a callable that compute the logits mask given the
        input observation (for categorical actor).
      eval_agent_fn: (Optional): a callable used to evaluate agent.
    """
    # initilize the observation
    
    obs, ep_len = env.reset()[0].copy().astype("float32"), 0

    for i in range(self._num_iters):
      # simulate the agent-env interations to collect `batch_size` examples.
      for j in range(self._batch_size):
        mask = None if mask_fn is None else mask_fn(obs)

        # given the current policy, sample an action and compute its log-prob
        logp, act = self._actor(obs, mask=mask)
        logp, act = logp.numpy().item(), act.numpy().squeeze()

        # given the current critic, compute the estimated state value
        val = self._critic(obs).numpy().item()

        # run the environment to get the next observation and reward
        next_obs, rew, done = env.step(act)[:3]
        next_obs = next_obs.copy().astype("float32")
        ep_len += 1

        # save the simulated data to buffer
        buf.store(obs, act, rew, val, logp)

        # update the `obs` tensor
        obs = next_obs

        if done or ep_len == self._max_ep_len or j == self._batch_size - 1:
          # if the last state is *terminal* state, set state value to 0
          # otherwise bootstrap the target value
          val = 0 if done else self._critic(obs).numpy().item()

          buf.finish_path(val)
          obs, ep_len = env.reset()[0].copy().astype("float32"), 0

      self._train_step(*buf.get())

      rew_buf = buf._rew_buf
      if i % self._log_freq == 0:
        if eval_agent_fn is None:
          print("[INFO] iter: %d, avg rew: %.3f, wins: %d, losses: %d" % (
              i, rew_buf.mean(), (rew_buf > 0).sum(), (rew_buf < 0).sum())
          )
        else:
          results = eval_agent_fn()
          eval_str = ""
          for k in results.keys():
            eval_str += f"{k}: {results[k]}, "
          print(f"[INFO] iter: {i}, {eval_str}")

      if self._ckpt_freq != -1 and i % self._ckpt_freq == 0:
        ckpt.save(self._ckpt_prefix)

  def get_eval_fn(self, env_eval, num_eval_episode, obs_shape):
    """Return the agent evaluation function.

    Args:
      env_eval: the environment for agent evaluation.
      num_eval_episode (int): num of evaluation episode.
      obs_shape (tuple): shape of observation tensor (not including batch). 

    Returns:
      eval_agent_fn: evaluation function.
    """
    def eval_agent_fn():
      ep_ret_list = []
      ep_len_list = []

      for i in range(num_eval_episode):
        obs, done, ep_ret, ep_len = env_eval.reset()[0], False, 0, 0
        while not (done or (ep_len == self._max_ep_len)):
          out = self._actor(np.reshape(obs, (-1, *obs_shape)).astype("float32"))
          act = out[1][0].numpy()
          obs, rew, done = env_eval.step(act)[:3]
          ep_ret += rew
          ep_len += 1
        ep_ret_list.append(ep_ret)
        ep_len_list.append(ep_len)
      return {"mean_ep_return": np.mean(ep_ret_list),
          "mean_ep_len": np.mean(ep_len_list)}

    return eval_agent_fn
 

class PPOTrainer(VPGTrainer):
  """Trainer for Proximal Policy Optimization actor-critic."""

  def __init__(
      self, actor, critic, actor_optim, critic_optim, clip_ratio=0.2,
    target_kl=0.01, num_steps_actor=80, num_steps_critic=80,
    num_iters=1500, batch_size=4000, max_ep_len=1000, log_freq=10,
    ckpt_freq=-1, ckpt_prefix="ppo"):
    """Constructor.

    Args:
      actor (Actor): the actor object.
      critic (Critic): the critic object.
      actor_optim: the optim for training actor weights.
      critic_optim: the optim for training critic weights.
      clip_ratio (float): the clip ratio.
      target_kl (float): target value for KL-divergence.
      num_steps_actor (int): num of backward passes to train the actor.
      num_steps_critic (int): num of backward passes to train the critic.
      num_iters (int): total number of iterations in the training loop.
      batch_size (int): num of training examples i.e. (obs, act, ret, adv)
        tuples used in a single update step.
      max_ep_len (int): maximum num of steps the environment will be simulated.
    """
    super().__init__(actor=actor, critic=critic, actor_optim=actor_optim
        ,critic_optim=critic_optim, batch_size=batch_size, max_ep_len=max_ep_len
      ,num_steps_critic=num_steps_critic, num_iters=num_iters, log_freq=log_freq
      ,ckpt_freq=ckpt_freq, ckpt_prefix=ckpt_prefix)
    self._clip_ratio = clip_ratio
    self._target_kl = target_kl
    self._num_steps_actor = num_steps_actor

  def _compute_actor_loss(self, obs, act, adv, logp_old):
    """Computes the actor loss.

    Args:
      obs (Tensor): observation tensor of shape [batch, ...].
      act (Tensor): action tensor of shape [batch] for discrete actions, or
        [batch, act_dim] for continuous actions.
      adv (Tensor): the advantage tensor of shape [batch].
      logp_old (Tensor): tensor of shape [batch], the cached log-prob of action
        computed by the policy before being updated.

    Returns:
      loss (Tensor): tensor of shape [batch], the loss for actor.
      kl (Tensor): tensor of shape [batch], the kl divergence of the
        distribution resulted from the old policy and the one from the most
        up-to-date policy.
    """
    clip_ratio = self._clip_ratio
    logp = self._actor(obs, act)

    ratio = tf.exp(logp - logp_old)
    clip_adv = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss = -tf.reduce_mean(tf.minimum(ratio * adv, clip_adv))
    kl = tf.reduce_mean(logp_old - logp)

    return loss, kl

  def _train_step(self, obs, act, ret, adv, logp_old):
    """Train the actor and critic using the collected episode data.

    Args:
      obs (Tensor): observation tensor of shape [batch, ...].
      act (Tensor): action tensor of shape [batch] for discrete actions, or
        [batch, act_dim] for continuous actions.
      ret (Tensor): the return tensor of shape [batch].
      adv (Tensor): the advantage tensor of shape [batch].
      logp_old (Tensor): tensor of shape [batch], the cached log-prob of action
        computed by the policy before being updated.
    """
    for _ in range(self._num_steps_actor):
      with tf.GradientTape() as tape:
        actor_loss, kl = self._compute_actor_loss(obs, act, adv, logp_old)
      if kl > 1.5 * self._target_kl:
        break
      grads = tape.gradient(actor_loss, self._actor.trainable_variables)
      self._actor_optim.apply_gradients(
          zip(grads, self._actor.trainable_variables))

    for _ in range(self._num_steps_critic):
      with tf.GradientTape() as tape:
        critic_loss = self._compute_critic_loss(obs, ret)
      grads = tape.gradient(critic_loss, self._critic.trainable_variables)
      self._critic_optim.apply_gradients(
          zip(grads, self._critic.trainable_variables))


class DDPGTrainer(object):
  """Trainer for Deep Deterministic Policy Gradient."""
  def __init__(
      self, critic, actor, critic_target, actor_target, critic_optim,
      actor_optim, max_ep_len=1000, act_noise=0.1, polyak=0.995, gamma=0.99,
      num_iters=400000, batch_size=100, start_steps=10000, update_freq=50,
      update_after=1000, log_freq=4000, ckpt_freq=4000, ckpt_prefix="ddpg",
    ):
    """Constructor.

    Args:
      critic: the q-function.
      actor: the policy.
      critic_target: the q-function whose weights are updated periodically.
      actor_target: the policy whose weights are updated periodically.
      critic_optim: the optimizer for critic.
      actor_optim: the optimizer for actor.
      max_ep_len (int): maximum num of steps the environment will be simulated.
      act_noise (float): action will be scaled by `act_noise` for exploration.
      polyak (float): the momentum used to update weights in `actor_target` and
        `critic_target`.
      gamma (float): the discounting factor.
      num_iters (int): total number of iterations in the training loop.
      batch_size (int): num of training examples i.e. (obs, act, ret, adv)
        tuples used in a single update step.
      start_steps (int): actions will be sampled from actor after "this" iters.
      update_freq (int): weight-updates will be done every "this" iters. 
      update_after (int): weight-updates will be done after "this" iters.
      log_freq (int): print log info every `log_freq` iterations.
      ckpt_freq (int): checkpointt will be saved save `ckpt_freq` iterations. If
        equals -1, no checkpoints will be saved.
      ckpt_prefix (str): the string as the prefix to checkpoint file names.
    """
    self._critic = critic
    self._actor = actor
    self._critic_target = critic_target
    self._actor_target = actor_target
    self._critic_optim = critic_optim
    self._actor_optim = actor_optim
    self._max_ep_len = max_ep_len
    self._act_noise = act_noise
    self._polyak = polyak
    self._gamma = gamma
    self._num_iters = num_iters
    self._batch_size = batch_size
    self._start_steps = start_steps
    self._update_freq = update_freq
    self._update_after = update_after
    self._log_freq = log_freq
    self._ckpt_freq = ckpt_freq
    self._ckpt_prefix = ckpt_prefix

  def _compute_critic_loss(self, obs, act, rew, obs2, done):
    """Compute the loss for training the action-value function `q(s, a)`.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim], i.e. `s`.
      act (Tensor): action tensor of shape [batch, act_dim], i.e. `a`.
      rew (Tensor): reward tensor of shape [batch], i.e. `r`.
      obs2 (Tensor): next-step obs. tensor of shape [batch, obs_dim], i.e. `s'`.
      done (Tensor): tensor of shape [batch], variable indicating whether
        episode has ended.

    Returs:
      loss (Tensor): the loss tnesor of shape [batch]. 
    """
    q = self._critic(obs, act)

    q2_tgt = self._critic_target(obs2, self._actor_target(obs2))
    backup = rew + self._gamma * (1 - done) * q2_tgt
    backup = tf.stop_gradient(backup)

    loss = tf.reduce_mean((q - backup) ** 2)
    return loss

  def _compute_actor_loss(self, obs):
    """Compute the loss for training the policy `mu(s)`.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim].

    Returns:
      loss (Tensor): the loss tensor of shape [batch]. 
    """
    q = self._critic(obs, self._actor(obs))
    loss = -tf.reduce_mean(q)
    return loss

  def _get_action(self, obs, sample=False):
    """Get the action from policy.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim].
      sample (bool): ignored for DDPG and TD3.

    Returns:
      act (Tensor): action tensor of shape [batch, act_dim], i.e. `a`.
    """
    obs = obs[tf.newaxis]
    act = self._actor(obs).numpy()
    act += self._act_noise * np.random.randn(self._actor._act_dim)
    act = np.squeeze(act)
    act = np.clip(act, -self._actor._act_limit, self._actor._act_limit)
    return act

  def _train_step(self, obs, obs2, act, rew, done, time_step=None):
    """Train the actor and critic using the collected episode data.

    Args:
      obs (Tensor): tensor of shape [batch, obs_dim], observation tensor.
      obs2 (Tensor): tensor of shape [batch, obs_dim], the next-step
        observation tensor.
      act (Tensor): tensor of shape [batch, act_dim], action tensor.
      rew (Tensor): tensor of shape [batch], the immediate reward.
      done (Tensor): tensor of shape [batch], variable indicating whether
        episode has ended.
      time_step (int): not used in DDPG and SAC, ignored. 
    """
    with tf.GradientTape() as tape:
      loss_critic = self._compute_critic_loss(obs, act, rew, obs2, done)
    grads = tape.gradient(loss_critic, self._critic.trainable_variables)
    self._critic_optim.apply_gradients(
        zip(grads, self._critic.trainable_variables))

    with tf.GradientTape() as tape:
      loss_actor = self._compute_actor_loss(obs)
    grads = tape.gradient(loss_actor, self._actor.trainable_variables)
    self._actor_optim.apply_gradients(
        zip(grads, self._actor.trainable_variables))

    self._update_weights(self._actor, self._actor_target)
    self._update_weights(self._critic, self._critic_target)

  def _update_weights(self, mod_src, mod_tgt):
    """Copy weights from `mod_src` to `mod_tgt`."""
    weights = []
    for v, vt in zip(mod_src.trainable_variables, mod_tgt.trainable_variables):
      weights.append(vt.numpy() * self._polyak + (1 - self._polyak) * v.numpy())
    mod_tgt.set_weights(weights)

  def train(self, env, buf, ckpt, eval_agent_fn=None):
    """Train both the actor (policy `mu(s)`) and critic (action-value function
    `q(s, a)`).

    Args:
      env: the environment object.
      buf: the replay buffer.
      ckpt: the checkpoint object.
      env_eval: teh environment for evaluation.
    """
    obs, ep_len = env.reset()[0], 0

    for i in range(self._num_iters):
      if i > self._start_steps:
        act = self._get_action(obs)
      else:
        act = env.action_space.sample()

      obs2, rew, done = env.step(act)[:3]
      ep_len += 1

      done = False if ep_len == self._max_ep_len else done

      buf.store(obs, act, rew, obs2, done)

      obs = obs2

      if done or (ep_len == self._max_ep_len):
        obs, ep_len = env.reset()[0], 0

      if i >= self._update_after and i % self._update_freq == 0:
        for j in range(self._update_freq):
          batch = buf.sample_batch(self._batch_size)
          self._train_step(*batch, time_step=j)

      if i % self._log_freq == 0:
        results = eval_agent_fn()
        eval_str = ""
        for k in results.keys():
          eval_str += f"{k}: {results[k]}, "
        print(f"[INFO] iter: {i}, {eval_str}")
        sys.stdout.flush()

      if self._ckpt_freq != -1 and i % self._ckpt_freq == 0:
        ckpt.save(self._ckpt_prefix)

  def get_eval_fn(
      self, env_eval, num_eval_episodes=10, sample=False, replay=False):
    """Return the agent evaluation function.

    Args:
      env_eval: the environment for agent evaluation.
      num_eval_episode (int): num of evaluation episode.
      sample (bool): only used for SAC, ignored for DDPG and TD3.

    Returns:
      eval_agent_fn: evaluation function.
    """
    def eval_agent_fn():
      ep_ret_list = []
      ep_len_list = []
      for i in range(num_eval_episodes):
        obs, done, ep_ret, ep_len = env_eval.reset()[0], False, 0, 0
        while not (done or (ep_len == self._max_ep_len)):
          obs, rew, done = env_eval.step(
              self._get_action(obs, sample=sample))[:3]
          ep_ret += rew
          ep_len += 1
        ep_ret_list.append(ep_ret)
        ep_len_list.append(ep_len)
      return {"episode_mean_ret": np.mean(ep_ret_list),
          "episode_mean_length": np.mean(ep_len_list)}

    if replay:
      def replay_fn(gif_filename):
        from PIL import Image
        import os

        os.environ["MUJOCO_GL"] = "egl"

        self._max_ep_len = 1000
        obs, done, _, ep_len = env_eval.reset()[0], False, 0, 0
        frame = env_eval.render().copy()
        frames = [frame]
        while not (done or (ep_len == self._max_ep_len)):
          obs, rew, done = env_eval.step(
                self._get_action(obs, sample=sample))[:3]
          frame = env_eval.render().copy()
          frames.append(frame)
          ep_len += 1

        print("aaaaaaaa")
        imgs = [Image.fromarray(img) for img in frames]
        print("bbbbbbb")
        imgs[0].save(gif_filename, save_all=True, append_images=imgs[1:],
            duration=50, loop=0)
      return replay_fn

    return eval_agent_fn


class TD3Trainer(DDPGTrainer):
  def __init__(
      self, critic, critic2, actor, critic_target, critic2_target, actor_target,
    critic_optim, actor_optim, max_ep_len=1000, act_noise=0.1, polyak=0.995,
    gamma=0.99, num_iters=400000, batch_size=100, start_steps=10000,
    update_freq=50, update_after=1000, log_freq=4000, ckpt_freq=4000,
    ckpt_prefix="td3", noise_clip=0.5, target_noise=0.2, policy_delay=2):
    """Constructor.

    Args:
      critic: the q-function.
      critic2: the second q-function.
      actor: the policy.
      critic_target: the q-function whose weights are updated periodically.
      critic2_target: the target q-function.
      actor_target: the policy whose weights are updated periodically.
      critic_optim: the optimizer for critic.
      actor_optim: the optimizer for actor.
      max_ep_len (int): maximum num of steps the environment will be simulated.
      act_noise (float): action will be scaled by `act_noise` for exploration.
      polyak (float): the momentum used to update weights in `actor_target` and
        `critic_target`.
      gamma (float): the discounting factor.
      num_iters (int): total number of iterations in the training loop.
      batch_size (int): num of training examples i.e. (obs, act, ret, adv)
        tuples used in a single update step.
      start_steps (int): actions will be sampled from actor after "this" iters.
      update_freq (int): weight-updates will be done every "this" iters. 
      update_after (int): weight-updates will be done after "this" iters.
      log_freq (int): print log info every `log_freq` iterations.
      ckpt_freq (int): checkpointt will be saved save `ckpt_freq` iterations. If
        equals -1, no checkpoints will be saved.
      ckpt_prefix (str): the string as the prefix to checkpoint file names.
      noise_clip (float): limit for absolute value of target policy noise.
      target_noise (float): stddev for noise added to target policy.
      policy_delay (int): policy will only be updated once every "this" times
        for each update of the Q-function.
    """
    super().__init__(critic=critic, actor=actor, critic_target=critic_target,
        actor_target=actor_target, critic_optim=critic_optim, log_freq=log_freq,
      actor_optim=actor_optim, max_ep_len=max_ep_len, act_noise=act_noise,
      polyak=polyak, gamma=gamma, num_iters=num_iters, batch_size=batch_size,
      start_steps=start_steps, update_freq=update_freq, ckpt_freq=ckpt_freq,
      update_after=update_after, ckpt_prefix=ckpt_prefix,
    )
    self._critic2 = critic2
    self._critic2_target = critic2_target

    self._noise_clip = noise_clip
    self._target_noise = target_noise
    self._policy_delay = policy_delay

  def _compute_critic_loss(self, obs, act, rew, obs2, done):
    """Compute the loss for training the action-value function `q(s, a)`.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim], i.e. `s`.
      act (Tensor): action tensor of shape [batch, act_dim], i.e. `a`.
      rew (Tensor): reward tensor of shape [batch], i.e. `r`.
      obs2 (Tensor): next-step obs. tensor of shape [batch, obs_dim], i.e. `s'`.
      done (Tensor): tensor of shape [batch], variable indicating whether
        episode has ended.

    Returs:
      loss (Tensor): the loss tnesor of shape [batch]. 
    """
    q = self._critic(obs, act)
    q2 = self._critic2(obs, act)

    act2_tgt = self._actor_target(obs2)
    eps = tf.random.normal(tf.shape(act2_tgt)) * self._target_noise
    eps = tf.clip_by_value(eps, -self._noise_clip, self._noise_clip)
    act2_tgt = act2_tgt + eps
    act_limit = self._actor._act_limit
    act2_tgt = tf.clip_by_value(act2_tgt, -act_limit, act_limit)

    q_tgt = self._critic_target(obs2, act2_tgt)
    q2_tgt = self._critic2_target(obs2, act2_tgt)
    q_tgt = tf.minimum(q_tgt, q2_tgt)
    backup = rew + self._gamma * (1 - done) * q_tgt
    backup = tf.stop_gradient(backup)

    loss_q = tf.reduce_mean((q - backup) ** 2)
    loss_q2 = tf.reduce_mean((q2 - backup) ** 2)
    loss = loss_q + loss_q2
    return loss

  def _train_step(self, obs, obs2, act, rew, done, time_step):
    """Train the actor and critic using the collected episode data.

    Args:
      obs (Tensor): tensor of shape [batch, obs_dim], observation tensor.
      obs2 (Tensor): tensor of shape [batch, obs_dim], the next-step
        observation tensor.
      act (Tensor): tensor of shape [batch, act_dim], action tensor.
      rew (Tensor): tensor of shape [batch], the immediate reward.
      done (Tensor): tensor of shape [batch], variable indicating whether
        episode has ended.
      time_step (int): index of the global training iteration.
    """
    with tf.GradientTape() as tape:
      loss_critic = self._compute_critic_loss(obs, act, rew, obs2, done)
    critic_vars = (
        self._critic.trainable_variables + self._critic2.trainable_variables
    )
    grads = tape.gradient(loss_critic, critic_vars)
    self._critic_optim.apply_gradients(zip(grads, critic_vars))

    if self._policy_delay == -1 or time_step % self._policy_delay == 0:
      with tf.GradientTape() as tape:
        loss_actor = self._compute_actor_loss(obs)
      grads = tape.gradient(loss_actor, self._actor.trainable_variables)
      self._actor_optim.apply_gradients(
          zip(grads, self._actor.trainable_variables))

      self._update_weights(self._actor, self._actor_target)
      self._update_weights(self._critic, self._critic_target)
      self._update_weights(self._critic2, self._critic2_target)


class SACTrainer(TD3Trainer):
  """Trainer for Soft Actor Critic model."""

  def __init__(
      self, critic, critic2, actor, critic_target, critic2_target, actor_target,
    critic_optim, actor_optim, max_ep_len=1000, polyak=0.995, gamma=0.99,
    num_iters=400000, batch_size=100, start_steps=10000, update_freq=50,
    update_after=1000, log_freq=4000, ckpt_freq=4000,ckpt_prefix="sac",
    alpha=0.2):
    """Constructor.

    Args:
      critic: the q-function.
      critic2: the second q-function.
      actor: the policy.
      critic_target: the q-function whose weights are updated periodically.
      critic2_target: the target q-function.
      actor_target: the policy whose weights are updated periodically.
      critic_optim: the optimizer for critic.
      actor_optim: the optimizer for actor.
      max_ep_len (int): maximum num of steps the environment will be simulated.
      polyak (float): the momentum used to update weights in `actor_target` and
        `critic_target`.
      gamma (float): the discounting factor.
      num_iters (int): total number of iterations in the training loop.
      batch_size (int): num of training examples i.e. (obs, act, ret, adv)
        tuples used in a single update step.
      start_steps (int): actions will be sampled from actor after "this" iters.
      update_freq (int): weight-updates will be done every "this" iters. 
      update_after (int): weight-updates will be done after "this" iters.
      log_freq (int): print log info every `log_freq` iterations.
      ckpt_freq (int): checkpointt will be saved save `ckpt_freq` iterations. If
        equals -1, no checkpoints will be saved.
      ckpt_prefix (str): the string as the prefix to checkpoint file names.
      alpha (float): entropy regularization coefficient.
    """
    super().__init__(critic=critic, critic2=critic2, actor=actor,
        critic_target=critic_target, critic2_target=critic2_target,
      actor_target=actor_target, critic_optim=critic_optim,
      actor_optim=actor_optim, max_ep_len=max_ep_len, polyak=polyak,
      gamma=gamma, num_iters=num_iters, batch_size=batch_size,
      start_steps=start_steps, update_freq=update_freq,
      update_after=update_after, log_freq=log_freq, ckpt_freq=ckpt_freq,
      ckpt_prefix=ckpt_prefix, policy_delay=-1)
    self._alpha = alpha

  def _compute_critic_loss(self, obs, act, rew, obs2, done):
    """Compute the loss for training the action-value function `q(s, a)`.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim], i.e. `s`.
      act (Tensor): action tensor of shape [batch, act_dim], i.e. `a`.
      rew (Tensor): reward tensor of shape [batch], i.e. `r`.
      obs2 (Tensor): next-step obs. tensor of shape [batch, obs_dim], i.e. `s'`.
      done (Tensor): tensor of shape [batch], variable indicating whether
        episode has ended.

    Returs:
      loss (Tensor): the loss tnesor of shape [batch]. 
    """
    q = self._critic(obs, act)
    q2 = self._critic2(obs, act)

    logp_act2, act2 = self._actor(obs2)

    q_tgt = self._critic_target(obs2, act2)
    q2_tgt = self._critic2_target(obs2, act2)
    q_tgt = tf.minimum(q_tgt, q2_tgt)
    backup = rew + self._gamma * (1 - done) * (q_tgt - self._alpha * logp_act2)
    backup = tf.stop_gradient(backup)

    loss_q = tf.reduce_mean((q - backup) ** 2)
    loss_q2 = tf.reduce_mean((q2 - backup) ** 2)
    loss = loss_q + loss_q2
    return loss

  def _compute_actor_loss(self, obs):
    """Compute the loss for training the policy `mu(s)`.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim].

    Returns:
      loss (Tensor): the loss tensor of shape [batch]. 
    """
    logp_act, act = self._actor(obs)

    q = self._critic(obs, act)
    q2 = self._critic2(obs, act)
    q = tf.minimum(q, q2)
    loss = tf.reduce_mean(self._alpha * logp_act - q)
    return loss

  def _get_action(self, obs, sample=True):
    """Get the action from policy.

    Args:
      obs (Tensor): observation tensor of shape [batch, obs_dim].
      sample (bool): whether to sample actions.

    Returns:
      act (Tensor): action tensor of shape [batch, act_dim], i.e. `a`.
    """
    obs = obs[tf.newaxis]
    logp, act = self._actor(obs, sample=sample)
    act = np.squeeze(act.numpy())
    return act
