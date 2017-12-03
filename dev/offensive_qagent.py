#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

'''
s: 
  - bool (teammate within a ball of radius R) (0-1)
  - bool (closest enemy within a ball of radius R, where closest enemy is NOT goalie) (0-1)
  - goalie proximity (0-10)
  - goal center proximity (0-10)
  - goal opening angle (0-10)
a:
  - MOVE
  - SHOOT
  - PASS
  - DRIBBLE
  - NOOP

'''

import itertools
from hfo import *

import random
import math
import numpy as np

STATES = 2*2*11*11*11
ACTIONS = 5

TEAMMATES = 1
OPPONENTS = 2

EPSILON = 0.05 #0.05 #0.05 #0.05 #0.05 #0.05 #0.05 #0.025
ALPHA = 0.25
GAMMA = 0.95

TRAIN = True
RANDOM = False

RADIUS = 0.2

def dist(x1, y1, x2, y2):
  return ((x2-x1)**2 + (y2-y1)**2)**0.5

# Return the trimmed state space
def getTrimmedState(state):
  team_prox = 0
  opp_prox = 0
  g_prox = 0
  gc_prox = 0
  go_angle = 0

  o_x, o_y = (state[0], state[1])
  for i in range(TEAMMATES):
    x, y = (state[9+3*TEAMMATES+3*i+1], state[9+3*TEAMMATES+3*i+2])
    if dist(o_x, o_y, x, y) < RADIUS:
      team_prox = 1
      break

  for i in range(OPPONENTS):
    x, y = (state[9+6*TEAMMATES+3*i+1], state[9+3*TEAMMATES+3*i+2])
    if dist(o_x, o_y, x, y) < RADIUS and state[9+6*TEAMMATES+3*i+3] != 1: #not goalie
      opp_prox = 1
    if state[9+6*TEAMMATES+3*i+3] == 1: #goalie
      g_prox = round(dist(o_x, o_y, x, y) * 10)
      if g_prox > 10:
        g_prox = 10

  gc_prox = round(5*(state[6]+1))
  go_angle = round(5*(state[8]+1))

  return (team_prox, opp_prox, int(g_prox), int(gc_prox), int(go_angle))

def getQvals(qvals, state):
  return qvals[state[0]][state[1]][state[2]][state[3]][state[4]]

  
def main():
  # Create the HFO Environment
  hfo = HFOEnvironment()
  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                      'bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)
  
  if TRAIN:
    qvals = [[[[[[0 for a in range(ACTIONS)] for m in range(11)] for l in range(11)] for k in range(11)] for j in range(2)] for i in range(2)]
    for i in range(2):
      for j in range(2):
        for k in range(11):
          for l in range(11):
            for m in range(11):
              for a in range(ACTIONS):
                qvals[i][j][k][l][m][a] = random.random()
  else:
    qvals = np.load('q.npy').tolist()

  episode_num = 0
  for episode in itertools.count():
    episode_num += 1
    status = IN_GAME

    # state = hfo.getState()
    # TODO add goalie location too?

    state = hfo.getState()
    t_state = getTrimmedState(state)


    while status == IN_GAME:
      #print(state[3])

      if not RANDOM:
        # Pick new action, a', to take with epsilon-greedy strategy
        #print(qvals[goalie_tile][robot_tile][ball_tile])
        a = getQvals(qvals, t_state).index(max(getQvals(qvals, t_state)))
        
        if random.random() < EPSILON:
          a = random.randint(0, ACTIONS-1)
      else:
        a = random.randint(0, ACTIONS-1)

      if a == 0:
        hfo.act(MOVE)
      elif a == 1:
        hfo.act(SHOOT)
      elif a == 1:
        hfo.act(PASS, state[9+6]) #teammate's number
      elif a == 1:
        hfo.act(DRIBBLE)
      else:
        hfo.act(NOOP)


      # Advance the environment and get the game status
      status = hfo.step()

      # Grab the state features from the environment
      state = hfo.getState()
      #print(len(state)) 23?!
      next_t_state = getTrimmedState(state)

      # Get reward, update Q-val

      #TODO: get the reward!
      r = 0
      if status == GOAL:
        r = -1
      if status == CAPTURED_BY_DEFENSE or status == OUT_OF_BOUNDS:
        r = 10

      if TRAIN:
        getQvals(qvals, t_state)[a] += ALPHA*(r + (GAMMA*max(getQvals(qvals, next_t_state))) - getQvals(qvals, t_state)[a])
        
      t_state = next_t_state


    # Check the outcome of the episode
    print(('Episode %d ended with %s'%(episode, hfo.statusToString(status))))
    # Quit if the server goes down
    if TRAIN:
      if episode_num % 5 == 0:
        q = np.array(qvals)
        np.save('q.npy', q)

    if status == SERVER_DOWN:
      hfo.act(QUIT)
      exit()




if __name__ == '__main__':
  main()
