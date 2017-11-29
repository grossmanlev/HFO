#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

'''
s: 
  - robot position (x,y) converted to 0-99 tile number
  - ball position (x,y) converted to 0-99 tile number
a:
  - INTERCEPT
  - REDUCE_ANGLE_TO_GOAL
  - DEFEND_GOAL
  - GO_TO_BALL

  (add later?)
  - MARK_PLAYER, i
  - REORIENT
'''

import itertools
from hfo import *

import random

STATES = 100*100
ACTIONS = 4

def main():
  # Create the HFO Environment
  hfo = HFOEnvironment()
  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                      'bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_right', False)
  
  # create qval array with random vals: [0,1)
  qvals = [0]*(STATES*ACTIONS)
  for i in range(STATES*ACTIONS):
    qvals[i] = random.random()


  for episode in itertools.count():
    status = IN_GAME
    while status == IN_GAME:
      # Grab the state features from the environment
      state = hfo.getState()

      if state[3] < 0: #the ball is "near" the goal
        hfo.act(INTERCEPT)
      else:
        hfo.act(REDUCE_ANGLE_TO_GOAL)

      # Take an action and get the current game status
      #hfo.act(DASH, 20.0, 0.)
      #hfo.act(INTERCEPT)
      # Advance the environment and get the game status
      status = hfo.step()
    # Check the outcome of the episode
    print(('Episode %d ended with %s'%(episode, hfo.statusToString(status))))
    # Quit if the server goes down
    if status == SERVER_DOWN:
      hfo.act(QUIT)
      exit()

if __name__ == '__main__':
  main()
