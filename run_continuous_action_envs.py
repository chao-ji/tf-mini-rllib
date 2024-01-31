import argparse
import importlib
import yaml

import numpy as np
import tensorflow as tf

import gymnasium as gym
import os

os.environ["MUJOCO_GL"] = "egl"

from core.models import SquashedGaussianActor, MLPCriticQ, DeterministicMLPActor, DiagonalGaussianMLPActor, MLPCriticV
from core.trainers import TD3Trainer, DDPGTrainer, SACTrainer, PPOTrainer
from core.buffers import ReplayBuffer, PolicyGradientBuffer
 

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--train", default="configs/continuous_action_envs.yaml")
  parser.add_argument("-n", "--nn", default="configs/nn.yaml")
  args = parser.parse_args()
  with open(args.train) as f:
    conf = yaml.safe_load(f)

  with open(args.nn) as f:
    nn = yaml.safe_load(f)

  env = gym.make(conf["env"]["class"])
  if conf["replay"]["render"]:
    env_eval = gym.make(conf["env"]["class"], render_mode="rgb_array")
  else:
    env_eval = gym.make(conf["env"]["class"])
  act_limit = env.action_space.high[0]
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  if conf["trainer"] == "SAC":
    Actor = SquashedGaussianActor
    nn[Actor.__name__]["layers"][-1]["units"] = env.action_space.shape[0]
    nn[Actor.__name__]["layers"][-2]["units"] = env.action_space.shape[0]
  elif conf["trainer"] in ("DDPG", "TD3"):
    Actor = DeterministicMLPActor
    nn[Actor.__name__]["layers"][-1]["units"] = env.action_space.shape[0]
  else:
    Actor = DiagonalGaussianMLPActor
    nn[Actor.__name__]["layers"][-1]["units"] = env.action_space.shape[0]

  if conf["trainer"] in ("SAC", "DDPG", "TD3"):
    actor = Actor(
        configs=nn[Actor.__name__]["layers"], act_limit=act_limit)
    actor_target = Actor(
        configs=nn[Actor.__name__]["layers"], act_limit=act_limit)
    critic = MLPCriticQ(configs=nn["MLPCriticQ"]["layers"])
    critic_target = MLPCriticQ(configs=nn["MLPCriticQ"]["layers"])

    if conf["trainer"] in ("TD3", "SAC"):
      critic2 = MLPCriticQ(configs=nn["MLPCriticQ"]["layers"])
      critic2_target = MLPCriticQ(configs=nn["MLPCriticQ"]["layers"])
  else:
    actor = Actor(configs=nn[Actor.__name__]["layers"], obs_dim=obs_dim)
    critic = MLPCriticV(configs=nn["MLPCriticV"]["layers"], obs_dim=obs_dim)

  actor_optim = tf.keras.optimizers.Adam(learning_rate=conf["actor_lr"])
  critic_optim = tf.keras.optimizers.Adam(learning_rate=conf["critic_lr"])

  if conf["trainer"] == "DDPG":
    trainer = DDPGTrainer(critic=critic, actor=actor,
        critic_target=critic_target, actor_target=actor_target,
      critic_optim=critic_optim, actor_optim=actor_optim,
      max_ep_len=conf["max_ep_len"], act_noise=conf["act_noise"],
      polyak=conf["polyak"], gamma=conf["gamma"],
      num_iters=conf["num_iters"], batch_size=conf["batch_size"],
      start_steps=conf["start_steps"], update_freq=conf["update_freq"],
      update_after=conf["update_after"], log_freq=conf["log_freq"],
      ckpt_freq=conf["ckpt_freq"])
  elif conf["trainer"] == "TD3":
    trainer = TD3Trainer(critic=critic, critic2=critic2, actor=actor,
        critic_target=critic_target, critic2_target=critic2_target,
      actor_target=actor_target, critic_optim=critic_optim,
      actor_optim=actor_optim, max_ep_len=conf["max_ep_len"],
      act_noise=conf["act_noise"], polyak=conf["polyak"], gamma=conf["gamma"],
      num_iters=conf["num_iters"], batch_size=conf["batch_size"],
      start_steps=conf["start_steps"], update_freq=conf["update_freq"],
      update_after=conf["update_after"], log_freq=conf["log_freq"],
      ckpt_freq=conf["ckpt_freq"], noise_clip=conf["noise_clip"],
      target_noise=conf["target_noise"], policy_delay=conf["policy_delay"])
  elif conf["trainer"] == "SAC":
    trainer = SACTrainer(critic=critic, critic2=critic2, actor=actor,
        critic_target=critic_target, critic2_target=critic2_target,
      actor_target=actor_target, critic_optim=critic_optim,
      actor_optim=actor_optim, max_ep_len=conf["max_ep_len"],
      polyak=conf["polyak"], gamma=conf["gamma"],
      num_iters=conf["num_iters"], batch_size=conf["batch_size"],
      start_steps=conf["start_steps"], update_freq=conf["update_freq"],
      update_after=conf["update_after"], log_freq=conf["log_freq"],
      ckpt_freq=conf["ckpt_freq"], alpha=conf["alpha"]
    )
  elif conf["trainer"] == "PPO":
    trainer = PPOTrainer(critic=critic, actor=actor, actor_optim=actor_optim,
        critic_optim=critic_optim, clip_ratio=conf["clip_ratio"],
      target_kl=conf["target_kl"], num_steps_actor=conf["num_steps_actor"],
      num_steps_critic=conf["num_steps_critic"], num_iters=conf["num_iters"],
      batch_size=conf["batch_size"], max_ep_len=conf["max_ep_len"],
      log_freq=conf["log_freq"], ckpt_freq=conf["ckpt_freq"]
    )
  else:
    raise NotImplementedError(f"{conf['trainer']} not supported")

  ckpt = tf.train.Checkpoint(model=actor)
  if conf["replay"]["render"]:
    ckpt.restore(conf["replay"]["ckpt_path"])
    if conf["trainer"] in ("DDPG", "TD3", "SAC"):
      replay_fn = trainer.get_eval_fn(env_eval, sample=False, replay=True)
      replay_fn(conf["replay"]["gif_filename"])

  else:
    if conf["trainer"] in ("DDPG", "TD3", "SAC"):
      eval_agent = trainer.get_eval_fn(env_eval, 10, sample=False)
      buf = ReplayBuffer(
          obs_dim=obs_dim, act_dim=act_dim, size=conf["replay_size"])
    else:
      eval_agent = trainer.get_eval_fn(
          env_eval, conf["num_eval_episode"], (obs_dim,))
      buf = PolicyGradientBuffer((obs_dim,), (act_dim,), size=conf["batch_size"]
          , gamma=conf["gamma"], lam=conf["lam"], act_type="continuous")
    trainer.train(env, buf, eval_agent_fn=eval_agent, ckpt=ckpt)
