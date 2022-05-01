import random
import numpy as np
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX, SubGoal, Wall
from queue import Queue
import itertools as itt

# Test specifically importing a specific environment
from gym_minigrid.envs import DoorKeyEnv, EmptyEnv16x16, MiniGridEnv, FourRoomsEnv, MultiRoomEnvN4S5, CrossingEnv

# Test importing wrappers
from gym_minigrid.wrappers import *

class EmptyWithSubGoals(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

class EmptySubGoalEnv16x16v4(EmptyWithSubGoals):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)

    def _gen_grid(self, width, height):
    # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        gridString = str(self)
        self.mission = "get to the green goal square"
        
        gridArray = gridString.split("\n")
        self.gridDict = {}
        for i,row in enumerate(gridArray):
            for j,c in enumerate(row):
                self.gridDict[(i,j)] = row[j*2:j*2+2]
                if row[j*2:j*2+2] == "GG":
                    self.goal_position = (i,j)
        cameFrom= self.bfs((1,1),(14,14))
        subGoals = self.addSubGoal(cameFrom,self.goal_position)
        print(subGoals)
        for subGoal in subGoals:
            self.put_obj(SubGoal(), subGoal[0], subGoal[1])

    def addSubGoal(self,cameFrom,goal):
        subGoals = []
        prev = cameFrom[goal]
        i =0
        while prev != None and prev != self.agent_start_pos:
        
            if i > 4 :
                subGoals.append(prev)
                i = 0
            prev = cameFrom[prev]
            i +=1

        return subGoals


    def bfs(self,startPos,goal):

        frontier = Queue()
        frontier.put(startPos)
        reached = set()
        came_from = dict()
        came_from[startPos] = None
        reached.add(startPos)
        goalReached = False

        while not frontier.empty() and not goalReached:
            current = frontier.get()
            for next in self.cur_neighbors(current):
                 if next not in came_from:
                    frontier.put(next)
                    came_from[next] = current
                    if next[0] == goal[0] and next[1] == goal[1]:
                        goalReached = True
                        return came_from
        

    def cur_neighbors(self,pos):
        neighbours = []

        newPos = (pos[0]+1,pos[1])
        neigh = ((pos[0]-1,pos[1]),(pos[0]+1,pos[1]),(pos[0],pos[1]+1),(pos[0]+1,pos[1]-1))
        for newPos in neigh:
            if newPos[0] <=15 or newPos[0] >=0 and newPos[1] <=15 or newPos[1] >=0:
                next_neigh = self.gridDict[newPos]
                if next_neigh != "WG":
                    neighbours.append(newPos)
        return neighbours


class CustomCrossing(CrossingEnv):
    def __init__(self, size=9, num_crossings=2, obstacle_type=Wall):
      super().__init__(size=size,  num_crossings=num_crossings, obstacle_type=obstacle_type)

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Wall
            else "find the opening and get to the green goal square"
        )
        gridString = str(self)
        gridArray = gridString.split("\n")
        self.gridDict = {}
        for i,row in enumerate(gridArray):
            for j,c in enumerate(row):
                self.gridDict[(i,j)] = row[j*2:j*2+2]
                if row[j*2:j*2+2] == "GG":
                    self.goal_position = (i,j)
        cameFrom = self.bfs(self.agent_pos,self.goal_position)
        print(self.goal_position)
        subGoals = self.addSubGoal(cameFrom,self.goal_position)
        print(subGoals)
        for subGoal in subGoals:
            self.put_obj(SubGoal(), subGoal[0], subGoal[1]) 

    def addSubGoal(self,cameFrom,goal):
        subGoals = []
        prev = cameFrom[goal]
        i =0
        while prev != None and prev != self.agent_pos:
        
            if i > 3 :
                subGoals.append(prev)
                i = 0
            prev = cameFrom[prev]
            i +=1

        return subGoals


    def bfs(self,startPos,goal):

        frontier = Queue()
        frontier.put(startPos)
        reached = set()
        came_from = dict()
        came_from[startPos] = None
        reached.add(startPos)
        goalReached = False

        while not frontier.empty() and not goalReached:
            current = frontier.get()
            for next in self.cur_neighbors(current):
                 if next not in came_from:
                    frontier.put(next)
                    came_from[next] = current
                    if next[0] == goal[0] and next[1] == goal[1]:
                        goalReached = True
                        return came_from
        

    def cur_neighbors(self,pos):
        neighbours = []

        neigh = ((pos[0]-1,pos[1]),(pos[0]+1,pos[1]),(pos[0],pos[1]+1),(pos[0]+1,pos[1]-1))
        for newPos in neigh:
            if newPos[0] <=8 or newPos[0] >=0 and newPos[1] <=8 or newPos[1] >=0:
                next_neigh = self.gridDict[newPos]
                if next_neigh != "WB":
                    neighbours.append(newPos)
        return neighbours
    
            


        
if __name__ == "__main__":

    env = CustomCrossing()
    print(env)



