# mini libray of RL algorithms in TensorFlow

This is a repository of a few Reinforcement Learning algorithms inplemented in TensorFlow 2.x. 
* Using Environments from [gymnasium](https://gymnasium.farama.org/), which is the successor to the outdated OpenAI [Gym](https://github.com/openai/gym). 
* Mujuco (3.x) library is included with `gymnasium` (No need to install it separately) which is required for [physical simulation environments](https://gymnasium.farama.org/environments/mujoco/).

## Quick Start

1. Clone the repo
`git clone https://github.com/chao-ji/tf-mini-rllib` 

2. Install requirements
`pip install -r requirements.txt`

3. Config files
The configurations are separated into two files: `configs/continuous_action_envs.yaml`(or `configs/tictactoe.yaml`), and `configs/nn.yaml`. The former contains the hyperparameters for each RL algorithm, and the latter contains the specifications of the neural networks for the policy function and value function.

4. Train and evaluation

To run RL algorithms on continuous action space environments, run

`python3 run_continuous_action_envs.py -t configs/continuous_action_envs.yaml -n configs/nn.yaml`

To generate a rendered GIF of the Mujoco simulation according policy from a  trained agent, set the relevant fields in `continuous_action_envs.yaml`, e.g.

```
replay:
  render: True
  ckpt_path: "sac-1"
  gif_filename: "replay.gif"
```

This repo also implemented a toy environment for playing the TicTacToe game that conforms to the environment API of [gymnasium](https://gymnasium.farama.org/). To run RL algorithms on the TicTacToe env, run

`python3 run_tictactoe.py -t configs/tictactoe.yaml -n configs/nn.yaml`
