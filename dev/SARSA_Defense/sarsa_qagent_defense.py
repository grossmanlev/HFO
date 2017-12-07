#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

'''
s: 
  - bool (kickable) (0-1)
  - bool (teammate within a ball of radius R) (0-1)
  - bool (closest enemy within a ball of radius R, where closest enemy is NOT goalie) (0-1)
  - bool (goalie within a ball of radius, R) (0-1) goalie proximity (0-10)
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

from operator import add

GOALIE_STATE = 3
ACTIONS = 4

TEAMMATES = 0
OPPONENTS = 2

EPSILON = 0.05 #0.05 #0.05 #0.05 #0.05 #0.05 #0.05 #0.025
ALPHA = 0.125
GAMMA =0.975
XI = 0.5
INTERCEPT_RADIUS = 0.01

TILE_BASE_NUM = 5
STATE_NUM =TILE_BASE_NUM * TILE_BASE_NUM

TRAIN = True
RANDOM = False
SARSA = True

RADIUS = 0.10

def dist(x1, y1, x2, y2):
  return ((x2-x1)**2 + (y2-y1)**2)**0.5

def heuristic(qvals, t_state, state):
  rtn = [4, 0, 0, 1]

  ball_x, ball_y = state[3], state[4]
  def_x, def_y = state [0], state[1]

  distance = dist(def_x, def_y, ball_x, ball_y)

  if distance <= INTERCEPT_RADIUS:
    rtn = [10, 2, 0, 0]

  if distance >= 0.5 and def_x>=0:
    rtn = [4, 0, 0, 10]

  return rtn

def getTile(x, y):
  #print(x, y)
  x = int(math.floor(2.5*(x+1)))
  y = int(math.floor(2.5*(y+1)))

  if x > 4:
    x = 4
  if y > 4:
    y = 4

  return TILE_BASE_NUM * y + x

def getGoalieTile(y):
  # if y < -0.2:
  #   return 0
  if y < -0.1:
    return 0
  # if y < 0:
  #   return 2
  if y < 0.1:
    return 1
  # if y < 0.2:
  #   return 4
  return 2

def oppHasBall(state):
  ball_tile = getTile(state[3], state[4])
  for o in range(OPPONENTS):
    o_x,o_y = (state[9+6*TEAMMATES+(3*o)], state[9+6*TEAMMATES+(3*o)+1])
    opp_tile = getTile(o_x, o_y)
    if ball_tile == opp_tile:
      return True
  return False

# Return the trimmed state space
def getTrimmedState(state):
  team_prox = 0
  opp_prox = 0
  # g_prox = 0
  # gc_prox = 0
  # go_angle = 0

  robot_tile = getTile(state[0], state[1])
  ball_tile = getTile(state[3], state[4])
  goalie_tile = getGoalieTile(state[9+3*TEAMMATES+1])



  o_x, o_y = (state[0], state[1])
  for i in range(TEAMMATES):
    x, y = (state[9+3*TEAMMATES+3*i+1], state[9+3*TEAMMATES+3*i+2])
    if dist(o_x, o_y, x, y) < RADIUS:
      team_prox = 1
      break

  for i in range(OPPONENTS):
    x, y = (state[9+6*TEAMMATES+3*i+1], state[9+6*TEAMMATES+3*i+2])
    if dist(o_x, o_y, x, y) < RADIUS and state[9+6*TEAMMATES+3*i+3] != 1: #not goalie
      opp_prox = 1
    
    # if state[9+6*TEAMMATES+3*i+3] == 1: #goalie
    #   if dist(o_x, o_y, x, y) < RADIUS:
    #     g_prox = 1

  # gc_prox = round(5*(state[6]+1))
  # go_angle = round(5*(state[8]+1))

  return (robot_tile, ball_tile, goalie_tile, team_prox, opp_prox)

  # return (int(state[5]), team_prox, opp_prox, g_prox, int(gc_prox), int(go_angle))

def getQvals(qvals, state):
  return qvals[state[0]][state[1]][state[2]][state[3]][state[4]]


def getAction(qvals, t_state, state):
  if TRAIN:
    if random.random() < EPSILON:
      return random.randint(0, ACTIONS-1)
    qs = map(add, getQvals(qvals, t_state), heuristic(qvals, t_state, state))
  else:
    qs = getQvals(qvals, t_state)


  #print(qs)
  tmp = qs[:]
  # if t_state[0] == 1: #has ball
  #   del tmp[4]
  #   del tmp[0]
  # else: #doesn't have ball
  #   del tmp[1:4]
  #print(tmp)

  #print(qs.index(max(tmp)))

  return qs.index(max(tmp))

def main():

  # Create the HFO Environment
  hfo = HFOEnvironment()

  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                      'bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_right', False)
  
  if TRAIN:
    # qvals = [[[[[[[0 for a in range(ACTIONS)] for m in range(11)] for l in range(11)] for k in range(2)] for j in range(2)] for i in range(2)] for h in range(2)]
    # for h in range(2):
    #   for i in range(2):
    #     for j in range(2):
    #       for k in range(2):
    #         for l in range(11):
    #           for m in range(11):
    #             for a in range(ACTIONS):
    #               qvals[h][i][j][k][l][m][a] = random.random()

    qvals = [[[[[[0 for k in range(ACTIONS)] for op_pr in range(2)] for tm_pr in range (2)] for gli_tile in range(GOALIE_STATE)] for j in range(STATE_NUM)] for i in range (STATE_NUM)]
    # qvals = [[[[[0 for op_pr in range(2)] for tm_pr in range (2)] for gli_tile in range(GOALIE_STATE)] for j in range(STATE_NUM)] for i in range (6)]

    for i in range(STATE_NUM):
      for j in range(STATE_NUM):
        for gli_tile in range (GOALIE_STATE):
          for tm_pr in range(2):
            for op_pr in range (2):
              for k in range(ACTIONS):
                qvals[i][j][gli_tile][tm_pr][op_pr][k] = 0
              # qvals[i][j][gli_tile][tm_pr][op_pr]= 0

  else:
    qvals = np.load('sarsa_defense_RL_heuristic_2300.npy').tolist()

  episode_num = 0
  for episode in itertools.count():
    episode_num += 1
    status = IN_GAME

    # state = hfo.getState()
    # TODO add goalie location too?

    state = hfo.getState()
    t_state = getTrimmedState(state)

    if not RANDOM:
      # Pick new action, a', to take with epsilon-greedy strategy
      a = getAction(qvals, t_state, state)
    else:
      a = random.randint(0, ACTIONS-1)


    while status == IN_GAME:
      if a == 0:
        hfo.act(INTERCEPT)
      elif a == 1:
        hfo.act(GO_TO_BALL)
      elif a == 2:
        hfo.act (NOOP)
      else:
        hfo.act (DEFEND_GOAL)

      # Advance the environment and get the game status
      status = hfo.step()

      # Grab the state features from the environment
      next_state = hfo.getState()
      #print(len(state)) 23?!
      next_t_state = getTrimmedState(next_state)

      # Get reward, update Q-val

      #TODO: get the reward!
      r = 0
      if status == GOAL:
        r -= 20
      if status == OUT_OF_TIME:
        r += 15
      if status == CAPTURED_BY_DEFENSE:
        r += 20
      if status == OUT_OF_BOUNDS:
        r += 15

      if oppHasBall:
        initial_r = -10
        
        distance = abs(state[1] - state[0])
        delta1 = distance%TILE_BASE_NUM + 1
        delta2 = distance/TILE_BASE_NUM + 1

        r = initial_r * (max (delta1, delta2))/TILE_BASE_NUM


      if TRAIN:
        getQvals(qvals, t_state)[a] += ALPHA*(r + (GAMMA*max(getQvals(qvals, next_t_state))) - getQvals(qvals, t_state)[a])

      if not RANDOM:
        # Pick new action, a', to take with epsilon-greedy strategy
        next_a = getAction(qvals, next_t_state, next_state)
      else:
        next_a = random.randint(0, ACTIONS-1)

      if TRAIN and SARSA:
        getQvals(qvals, t_state)[a] += ALPHA*(r + (GAMMA*getQvals(qvals, next_t_state)[next_a]) - getQvals(qvals, t_state)[a])


      state = next_state
      t_state = next_t_state
      a = next_a


    # Check the outcome of the episode
    print(('Episode %d ended with %s'%(episode, hfo.statusToString(status))))
    # Quit if the server goes down

    if TRAIN and episode_num % 500 == 0:
      q = np.array(qvals)
      np.save(('sarsa_defense_RL_heuristic_TEAMMATES_0_OPPONENTS_2_' + str(episode_num) + '.npy'), q)

    if status == SERVER_DOWN:
      hfo.act(QUIT)
      exit()


if __name__ == '__main__':
  main()
