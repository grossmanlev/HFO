#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

'''
s: 
  - ball position (x,y) converted to 0-99 tile number
  - Goalie position converter to 0-5 number
  - proximity of closest opponent 0-19
 
a:
  - DEFEND_GOAL
  - INTERCEPT
  - MOVE TO POST
  - CATCH
'''

import itertools
from hfo import *

import random
import math
import numpy as np

STATES = 100*100*2
ACTIONS = 4

MAX_PROXIMITY = 20
GOALIE_POSITIONS = 100
BALL_POSITIONS = 100

TEAMMATES = 1
OPPONENTS = 2

EPSILON = .05 #0.05 #0.05 #0.05 #0.05 #0.025
ALPHA = 0.125
GAMMA = .975

#flags
TRAIN = False
RANDOM = False
SARSA = False

# Gets tile in range 0-99 from (x,y) position
def getTile(x, y):
  x = int(math.floor(5*(x+1)))
  y = int(math.floor(5*(y+1)))

  if x > 9:
    x = 9
  if y > 9:
    y = 9
  #print(10*y+x)

  #print(10*y + x)
  return 10*y + x

# Gets tile in range 0-5 from y position of goalie
def getGoalieTile(x, y):
  x = x + 0.85
  y = y + 0.3

  x = int(math.floor(x*25))
  y = int(math.floor(y*9))

  if x > 4:
    x = 4
  if y > 4:
    y = 4
 
  return 5*x + y

#returns the distance to the closest opponent from 0-19
def getClosestOpp(i):
  return int(math.floor(10*(i+1)))

#returns the distance to the ball from 0-19
def distToBall(state):
  return int(math.floor(2.5*((state[4] - state[1])**2 + (state[3] - state[0])**2)))

#returns 0 if goalie is closest to ball, 1 if a teammate is closer, 2 if an opponent is closer
def isClosestToBall(state, ball_x, ball_y, goalie_x, goalie_y):
  closer = 0
  goalieDist = abs(ball_x - goalie_x)**2 + abs(ball_y - goalie_y)**2
  for i in range(TEAMMATES):
    team_x = state[10 + 3*TEAMMATES + 3*i]
    team_y = state[10 + 3*TEAMMATES + 3*i + 1]
    team_dist =  abs(ball_x - team_x)**2 + abs(ball_y - team_y)**2
    if team_dist < goalieDist:
      return 1

  for i in range(OPPONENTS):
    opp_x = state[10 + 6*TEAMMATES + 3*i]
    opp_y = state[10 + 6*TEAMMATES + 3*i + 1]
    opp_dist =  abs(ball_x - opp_x)**2 + abs(ball_y - opp_y)**2
    if opp_dist < goalieDist:
      return 2  

  return 0



# returns if an opponent is in same tile as ball (they probably have the ball)
def oppHasBall(state):
  ball_tile = getTile(state[3], state[4])
  for o in range(OPPONENTS):
    o_x,o_y = (state[9+6*TEAMMATES+(3*o)],  state[9+6*TEAMMATES+(3*o)+1])
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
                      'localhost', 'base_right', True)
  # create qval array with random vals: [0,1)
  # qvals = [0]*(STATES*ACTIONS)
  # for i in range(STATES*ACTIONS):
  #   qvals[i] = random.random()

  #numiters = [[[[0 for k in range(ACTIONS)] for j in range(100)] for i in range(100)] for g in range(6)]
  
  #initilaize q-values array
  if TRAIN:
    qvals = [[[[[0 for k in range(ACTIONS)] for j in range(BALL_POSITIONS)] for p in range(2)] for c in range(3)] for g in range (25)]
    for c in range(3):
      for p in range(2):
        for j in range(BALL_POSITIONS):
          for k in range(ACTIONS):
            qvals[g][c][p][j][k] = random.random()
  else:
    qvals = np.load('goalie_trained_o.npy').tolist()

  #run episodes 
  episode_num = 0
  for episode in itertools.count():
    episode_num += 1
    status = IN_GAME

    state = hfo.getState()
    opp_pos = oppHasBall(state)
    robot_tile = getTile(-state[0], state[1])
    ball_tile = getTile(-state[3], state[4])
    goalie_tile = getGoalieTile(state[0], state[1])
    prox = getClosestOpp(state[9])
    dist = distToBall(state)

    closestToBall = isClosestToBall(state, state[3], state[4], state[0], state[1])

    #pick best action or random with small probability
    if not RANDOM:
        a = qvals[goalie_tile][closestToBall][opp_pos][ball_tile].index(max(qvals[goalie_tile][closestToBall][opp_pos][ball_tile]))
        
        if random.random() < EPSILON:
          a = random.randint(0, ACTIONS-1)
    #take random action
    else:
      a = random.randint(0, ACTIONS-1)

    while status == IN_GAME:

      #perform action 
      if a == 0:
        hfo.act(DEFEND_GOAL)
      elif a == 1:
        if state[4] < 0:
          hfo.act(MOVE_TO, -.78, -0.20)
        else:
          hfo.act(MOVE_TO, -.78, 0.20)
      elif a == 2:
        hfo.act(INTERCEPT)
      else:
        hfo.act(CATCH)

      # Advance the environment and get the game status
      status = hfo.step()

      # Grab the state features from the environment
      state = hfo.getState()
      #print(len(state)) 23?!
      next_robot_tile = getTile(-state[0], state[1])
      next_ball_tile = getTile(-state[3], state[4])
      next_goalie_tile = getGoalieTile(state[0], state[1])
      next_prox = getClosestOpp(state[9])
      next_opp_pos = oppHasBall(state)
      next_closest_to_ball = isClosestToBall(state, state[3], state[4], state[0], state[1])
      next_dist = distToBall(state)
     

      # Get reward, update Q-val
      
      r = 0
      #penalty for goal scored
      if status == GOAL:
        r = -10
      #big reward for goalie save
      elif status == CAPTURED_BY_DEFENSE and next_robot_tile == next_ball_tile:
        r = 100
      #smaller reward for defensive stop
      elif status == CAPTURED_BY_DEFENSE or status == OUT_OF_BOUNDS:
        r = 20
      #small living reward (encourages long possessions)
      else:
        r = .01

      
      #update q values
      if TRAIN:
        qvals[goalie_tile][closestToBall][opp_pos][ball_tile][a] += ALPHA*(r + (GAMMA*max(qvals[next_goalie_tile][next_closest_to_ball][next_opp_pos][next_ball_tile])) - qvals[goalie_tile][closestToBall][opp_pos][ball_tile][a])
        #numiters[goalie_tile][robot_tile][ball_tile][a] += 1

      #get next action
      if not RANDOM:
        next_a = qvals[goalie_tile][closestToBall][opp_pos][ball_tile].index(max(qvals[goalie_tile][closestToBall][opp_pos][ball_tile]))     
        if random.random() < EPSILON:
          next_a = random.randint(0, ACTIONS-1)
      else:
        next_a = random.randint(0, ACTIONS-1)

      #perform slightly different update for SARSA
      if TRAIN and SARSA:
        qvals[goalie_tile][closestToBall][opp_pos][ball_tile][a] += ALPHA*(r + (GAMMA*qvals[next_goalie_tile][next_closest_to_ball][next_opp_pos][next_ball_tile][next_a]) - qvals[goalie_tile][closestToBall][opp_pos][ball_tile][a])

      #set state features for next iteration
      robot_tile = next_robot_tile
      ball_tile = next_ball_tile
      goalie_tile = next_goalie_tile
      prox = next_prox
      opp_pos = next_opp_pos
      closestToBall = next_closest_to_ball
      dist = next_dist
      a = next_a

    # Check the outcome of the episode
    print(('Episode %d ended with %s'%(episode, hfo.statusToString(status))))
    
    #save q-values periodically
    if TRAIN:
      if episode_num % 5 == 0:
        q = np.array(qvals)
        #iters = np.array(numiters)
        np.save('goalie_trained_o.npy', q)
        #np.save('iters.npy', iters)

    # Quit if the server goes down
    if status == SERVER_DOWN:
      hfo.act(QUIT)
      exit()




if __name__ == '__main__':
  main()
