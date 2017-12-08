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
import math
import numpy as np
from operator import add

STATES = 6*25*25*2
ACTIONS = 4

TEAMMATES = 0
OPPONENTS = 2

TILE_BASE_NUM = 5
STATE_NUM =TILE_BASE_NUM * TILE_BASE_NUM

EPSILON = 0.05 #0.05 #0.05 #0.05 #0.05 #0.05 #0.05 #0.025
ALPHA = 0.25
GAMMA = 1.00
XI = 0.5

TRAIN = True
RANDOM = False

# Gets tile in range 0-99 from (x,y) position
def getTile(x, y):
  #print(x, y)
  x = int(math.floor(2.5*(x+1)))
  y = int(math.floor(2.5*(y+1)))

  if x > 4:
    x = 4
  if y > 4:
    y = 4

  return TILE_BASE_NUM * y + x
  
  #print(10*y+x)

  #print(10*y + x)
  # if 10*y + x == 99:
  #   print("99!")
  # return 10*y + x

# Gets tile in range 0-5 from y position of goalie
def getGoalieTile(y):
  if y < -0.2:
    return 0
  if y < -0.1:
    return 1
  if y < 0:
    return 2
  if y < 0.1:
    return 3
  if y < 0.2:
    return 4
  return 5

# returns if an opponent is in same tile as ball (they probably have the ball)
def oppHasBall(state):
  ball_tile = getTile(state[3], state[4])
  for o in range(OPPONENTS):
    o_x,o_y = (state[9+6*TEAMMATES+(3*o)],  state[9+6*TEAMMATES+(3*o)+1])
    opp_tile = getTile(o_x, o_y)
    if ball_tile == opp_tile:
      return True
  return False

def heuristic(state):
  return [XI*10, XI*3, XI*2, XI*1]
  
#def main(alphaValue, gammaValue):
def main ():
  # Create the HFO Environment
  hfo = HFOEnvironment()
  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                      'bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_right', False)
  
  # create qval array with random vals: [0,1)
  # qvals = [0]*(STATES*ACTIONS)
  # for i in range(STATES*ACTIONS):
  #   qvals[i] = random.random()

  if TRAIN:
    # qvals = [[[[0 for k in range(ACTIONS)] for j in range(100)] for i in range(100)] for g in range(6)]
    qvals = [[[[[0 for k in range(ACTIONS)] for o in range(2)] for j in range(STATE_NUM)] for i in range(STATE_NUM)] for g in range (6)]

    for g in range(6):
      for i in range(STATE_NUM):
        for j in range(STATE_NUM):
          for o in range (2):
            for k in range(ACTIONS):
              qvals[g][i][j][o][k] = random.random()
  else:
    qvals = np.load('player2_heuristic_rl_actions4_states25_weights_6_2_3_1_2Dvs1O.npy').tolist()

  episode_num = 0
  for episode in itertools.count():
    episode_num += 1
    status = IN_GAME

    # state = hfo.getState()
    # TODO add goalie location too?
    state = hfo.getState()
    robot_tile = getTile(-state[0], state[1])
    ball_tile = getTile(-state[3], state[4])
    goalie_tile = getGoalieTile(state[9+3*TEAMMATES+1])

    radius = 1
    # print robot_tile
    # print ball_tile
    delta_ball = abs (robot_tile - ball_tile)
    delta_ball_x = delta_ball / TILE_BASE_NUM
    delta_ball_y = delta_ball % TILE_BASE_NUM

    open_ball_in = 0

    if max (delta_ball_x, delta_ball_y) <= radius:
      open_ball_in = 1

    # print "robot:", robot_tile
    # print "ball: ", ball_tile


    while status == IN_GAME:
      #print(state[3])

      if not RANDOM:
        # Pick new action, a', to take with epsilon-greedy strategy
        #print(qvals[goalie_tile][robot_tile][ball_tile])
        if TRAIN:
          a = map(add, qvals[goalie_tile][robot_tile][ball_tile][open_ball_in], heuristic(state)).index(max(map(add, qvals[goalie_tile][robot_tile][ball_tile][open_ball_in],heuristic(state))))
          if random.random() < EPSILON:
            a = random.randint(0, ACTIONS-1)
        else:
          a = qvals[goalie_tile][robot_tile][ball_tile][open_ball_in].index(max(qvals[goalie_tile][robot_tile][ball_tile][open_ball_in]))
          # a = map(add, qvals[goalie_tile][robot_tile][ball_tile][open_ball_in], heuristic(state)).index(max(map(add, qvals[goalie_tile][robot_tile][ball_tile][open_ball_in],heuristic(state))))
      else:
        a = random.randint(0, ACTIONS-1)


      # print "action: ", a

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
      state = hfo.getState()
      #print(len(state)) 23?!
      next_robot_tile = getTile(-state[0], state[1])
      next_ball_tile = getTile(-state[3], state[4])
      next_goalie_tile = getGoalieTile(state[10+3*TEAMMATES+1])

      new_delta_ball = abs (robot_tile - ball_tile)
      new_delta_ball_x = delta_ball / TILE_BASE_NUM
      new_delta_ball_y = delta_ball % TILE_BASE_NUM

      new_open_ball_in = 0

      if max (new_delta_ball_x, new_delta_ball_y) <= radius:
        new_open_ball_in = 1

      # Get reward, update Q-val

      #TODO: get the reward!
      r = 0
      if status == GOAL:
        r = -15
        # distance = ball_tile - robot_tile
        # delta1 = distance%10 + 1
        # delta2 = distance/10 + 1

        # r *= min (delta1, delta2)

      if status == CAPTURED_BY_DEFENSE or status == OUT_OF_BOUNDS:
        r = 10

      if oppHasBall:
        r = -10
        
        distance = abs(ball_tile - robot_tile)
        delta1 = distance%TILE_BASE_NUM + 1
        delta2 = distance/TILE_BASE_NUM + 1

        r *= (max (delta1, delta2))/TILE_BASE_NUM

        # delta = state[9]
        
        # if state [9] < 0:
        #   delta = -state[9] + 1
        # r *= (delta/2)

      if TRAIN:
        qvals[goalie_tile][robot_tile][ball_tile][open_ball_in][a] += ALPHA*(r + (GAMMA*max(qvals[next_goalie_tile][next_robot_tile][next_ball_tile][new_open_ball_in])) - qvals[goalie_tile][robot_tile][ball_tile][open_ball_in][a])
        

      robot_tile = next_robot_tile
      ball_tile = next_ball_tile
      goalie_tile = next_goalie_tile
      open_ball_in = new_open_ball_in




      # reward: -5 if enemy has ball, -20 if goal, +15 OOB, CAPTURED_BY_DEFENSE




      # if state[3] < 0: #the ball is "near" the goal
      #   hfo.act(INTERCEPT)
      # else:
      #   hfo.act(REDUCE_ANGLE_TO_GOAL)

      # Take an action and get the current game status
      #hfo.act(DASH, 20.0, 0.)
      #hfo.act(INTERCEPT)

      # Advance the environment and get the game status
      #status = hfo.step()


    # Check the outcome of the episode
    print(('Episode %d ended with %s'%(episode, hfo.statusToString(status))))
    # Quit if the server goes down
    if TRAIN:
      if episode_num % 5 == 0:
        q = np.array(qvals)
        np.save('player2_heuristic_rl_actions4_states25_weights_6_2_3_1_2Dvs1O.npy', q)
        # np.save('q_erdos_queristiclearning_act4_10K.npy', q)

    if status == SERVER_DOWN:
      hfo.act(QUIT)
      exit()


if __name__ == '__main__':
  main()
