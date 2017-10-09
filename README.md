# StartCraft II Reinforcement Learning Examples

This example program was built on 
- pysc2 (Deepmind) [https://github.com/deepmind/pysc2]
- baselines (OpenAI) [https://github.com/openai/baselines]
- s2client-proto (Blizzard) [https://github.com/Blizzard/s2client-proto]
- Tensorflow 1.3 (Google) [https://github.com/tensorflow/tensorflow]

# Current examples

## Minimaps
- CollectMineralShards with Deep Q Network

![CollectMineralShards](https://media.giphy.com/media/UrgVK9TFfv2AE/giphy.gif "Collect Mineral")

# Quick Start Guide

## 1. Get PySC2

### PyPI

The easiest way to get PySC2 is to use pip:

```shell
$ pip install pysc2
```

Also, you have to install `baselines` library.

```shell
$ pip install baselines
```

## 2. Install StarCraft II

### Mac / Win

You have to purchase StarCraft II and install it. Or even the Starter Edition will work.

http://us.battle.net/sc2/en/legacy-of-the-void/

### Linux Packages

Follow Blizzard's [documentation](https://github.com/Blizzard/s2client-proto#downloads) to
get the linux version. By default, PySC2 expects the game to live in
`~/StarCraftII/`.

* [3.16.1](http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip)

## 3. Download Maps

Download the [ladder maps](https://github.com/Blizzard/s2client-proto#downloads)
and the [mini games](https://github.com/deepmind/pysc2/releases/download/v1.0/mini_games.zip)
and extract them to your `StarcraftII/Maps/` directory.

## 4. Train it!

```shell
$ python train_mineral_shards.py --algorithm=acktr
```

## 5. Enjoy it!

```shell
$ python enjoy_mineral_shards.py
```

## 4-1. Train it with DQN

```shell
$ python train_mineral_shards.py --algorithm=deepq --prioritized=True --dueling=True --timesteps=2000000 --exploration_fraction=0.2
```


## 4-2. Train it with ACKTR(A3C)

```shell
$ python train_mineral_shards.py --algorithm=acktr --num_cpu=16--timesteps=2000000
```


|                      | Description                                     | Default                         | Parameter Type |
|----------------------|-------------------------------------------------|---------------------------------|----------------|
| map                  | Gym Environment                                 | CollectMineralShards            | string         |
| log                  | logging type  : tensorboard, stdout             | tensorboard                     | string         |
| algorithm            | Currently, support 2 algorithms  : deepq, acktr | acktr                           | string         |
| timesteps            | Total training steps                            | 2000000                         | int            |
| exploration_fraction | exploration fraction                            | 0.5                             | float          |
| prioritized          | Whether using prioritized replay for DQN        | False                           | boolean        |
| dueling              | Whether using dueling network for DQN           | False                           | boolean        |
| num_cpu              | number of agents for A3C(acktr)                 | 4                               | int            |

