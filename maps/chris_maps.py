"""Define the mini game map configs. These are maps made by Deepmind."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib

class ChrisMaps(lib.Map):
  directory = "chris_maps"
  download = "https://github.com/chris-chris/pysc2-examples#get-the-maps"
  players = 1
  score_index = 0
  game_steps_per_episode = 0
  step_mul = 8

chris_maps = [
  "DefeatZealots",  # 120s
]

for name in chris_maps:
  globals()[name] = type(name, (ChrisMaps,), dict(filename=name))
