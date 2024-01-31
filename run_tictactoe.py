import argparse
import importlib
import sys
import yaml

import numpy as np
import tensorflow as tf

from core.models import CategoricalConvActor, ConvCriticV
from core.trainers import PPOTrainer, VPGTrainer
from core.buffers import PolicyGradientBuffer
import core.models as models


NEG_INF = -1e9
OBS_SHAPE = (-1, 3, 3, 1)
NUM_ACT = 9

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--train", default="configs/tictactoe.yaml")
  parser.add_argument("-n", "--nn", default="configs/nn.yaml")
  args = parser.parse_args()

  with open(args.train) as f:
    conf = yaml.safe_load(f)

  with open(args.nn) as f:
    nn = yaml.safe_load(f)

  module = importlib.import_module(conf["env"]["path"])
  env_class = getattr(module, conf["env"]["class"])
  env = env_class(getattr(module, "respond_func"))

  actor = CategoricalConvActor(
      obs_shape=OBS_SHAPE,
      num_act=NUM_ACT,
      configs=nn["CategoricalConvActor"]["layers"],
  )
  critic = ConvCriticV(
      obs_shape=OBS_SHAPE,
      configs=nn["ConvCriticV"]["layers"],
  )

  actor_optim = tf.keras.optimizers.Adam(learning_rate=conf["actor_lr"])
  critic_optim = tf.keras.optimizers.Adam(learning_rate=conf["critic_lr"])

  buf = PolicyGradientBuffer(obs_shape=(NUM_ACT,), act_shape=(),
      size=conf["batch_size"], gamma=conf["gamma"], lam=conf["lam"],
    act_type="discrete",
  )

  if conf["trainer"] == "PPO":
    trainer = PPOTrainer(actor=actor, critic=critic, actor_optim=actor_optim,
        critic_optim=critic_optim, clip_ratio=conf["clip_ratio"],
      target_kl=conf["target_kl"], num_steps_actor=conf["num_steps_actor"],
      num_steps_critic=conf["num_steps_critic"], num_iters=conf["num_iters"],
      batch_size=conf["batch_size"], max_ep_len=conf["max_ep_len"],
    )
  elif conf["trainer"] == "VPG":
    trainer = VPGTrainer(actor=actor, critic=critic, actor_optim=actor_optim,
        critic_optim=critic_optim, num_iters=conf["num_iters"],
      batch_size=conf["batch_size"], max_ep_len=conf["max_ep_len"])
  else:
    raise NotImplementedError(f"{conf['trainer']} not supported")

  mask_fn = lambda obs: tf.reshape(
      tf.cast(obs != 0, "float32") * NEG_INF, (1, NUM_ACT))
  ckpt = tf.train.Checkpoint(
      model=actor, actor_optim=actor_optim, critic_optim=critic_optim)

  trainer.train(env, buf, mask_fn=mask_fn, ckpt=ckpt)
