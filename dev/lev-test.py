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

STATES = 100*100
ACTIONS = 4

TEAMMATES = 2
OPPONENTS = 2

EPSILON = 0.05
APLHA = 0.25
GAMMA = 0.9

# Gets tile in range 0-99 from (x,y) position
def getTile(x, y):
  x = math.floor(10*(x+1))
  y = math.floor(10*(y+1))

  if x == 10:
    x = 9
  if y == 10:
    y = 9
  
  return 10*y + x

# returns if an opponent is in same tile as ball (they probably have the ball)
def oppHasBall(state):
  ball_tile = getTile(state[3], state[4])
  for o in range(OPPONENTS):
    o_x,o_y = (state[10+6*TEAMMATES+(3*o)],  state[10+6*TEAMMATES+(3*o)+1])
    opp_tile = getTile(o_x, o_y)
    if ball_tile == opp_tile:
      return True
  return False

  
def main():
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

  qvals = [[[0 for k in range(ACTIONS)] for j in range(100)] for i in range(100)]
  for i in range(100):
    for j in range(100):
      for k in range(ACTIONS):
        qvals[i][j][k] = random.random()



  for episode in itertools.count():
    status = IN_GAME

   # state = hfo.getState()

    state = hfo.getState()
    robot_tile = getTile(state[0], state[1])
    ball_tile = getTile(state[3], state[4])

    while status == IN_GAME:

      # Pick new action, a', to take with epsilon-greedy strategy
      a = qvals[robot_tile][ball_tile].index(max(qvals[robot_tile][ball_tile]))
      if random.random() < EPSILON:
        a = random.randint(0, ACTIONS-1)

      if a == 0:
        hfo.act(INTERCEPT)
      elif a == 1:
        hfo.act(REDUCE_ANGLE_TO_GOAL)
      elif a == 2:
        hfo.act(DEFEND_GOAL)
      else:
        hfo.act(GO_TO_BALL)

      # Advance the environment and get the game status
      status = hfo.step()

      # Grab the state features from the environment
      state = hfo.getState()
      next_robot_tile = getTile(state[0], state[1])
      next_ball_tile = getTile(state[3], state[4])

      # Get reward, update Q-val

      #TODO: get the reward!
      r = 0
      if status == GOAL:
        r = -20
      elif status == CAPTURED_BY_DEFENSE or status == OUT_OF_BOUNDS:
        r = 15
      elif oppHasBall(state):
        r = -5
      else:
        r = 0

      qvals[robot_tile][ball_tile][a] +=
        ALPHA*(r + GAMMA*max(qvals[next_robot_tile][next_ball_tile]) - qvals[robot_tile][ball_tile])

      robot_tile = next_robot_tile
      ball_tile = next_ball_tile


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
    if status == SERVER_DOWN:
      hfo.act(QUIT)
      exit()



if __name__ == '__main__':
  main()
