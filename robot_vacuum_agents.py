'''
Use this script file to define your robot vacuum agents.

The run function will generate a map showing the animation of the robot, and return the output of the loss function at the end of the run. The many_runs function will run the simulation multiple times without the visualization and return the average loss. 

You will need to implement a run_all function, which executes the many_runs function for all 12 of your agents (with the correct parameters) and sums up their returned losses as a single value. Your run_all function should use the following parameters for each agent: map_width=20, max_steps=50000 runs=100.
'''

from vacuum import *
from collections import deque

directions = ['north', 'south', 'east', 'west']
numdirections = [0, 1, 2, 3]
prevdirection = 'null'
map_width = 20
max_steps = 50000
runs = 100
#random.seed(69420)
memory = {}
visited_tiles = []
relative_position = [0, 0]
path = []

def memory_reset():
    global memory
    global visited_tiles
    global relative_position
    global path
    memory = {}
    visited_tiles = []
    relative_position = [0, 0]
    path = []

#agent naming format: [knowledge of the world]_[does it have memory]_[loss function]

def map_no(percept, x, y):
    if percept[x][y] == 'dirt':
        return 'clean'
    
    #find distances to dirty tiles
    dirty_distances=[]
    for i in range(len(percept)):
        for j in range(len(percept[0])):
            if percept[i][j] == 'dirt':
                distance = abs(x-i) + abs(y-j)
                dirty_distances.append((distance, i, j))
    
    #sort by distance
    dirty_distances.sort()
    
    if dirty_distances:
        closest_dirty = dirty_distances[0]
        closest_x, closest_y = closest_dirty[1], closest_dirty[2]
        valid = []
        if closest_y > y and percept[x][y+1] != 'wall':
            valid.append('north')
        if closest_y < y and percept[x][y-1] != 'wall':
            valid.append('south')
        if closest_x > x and percept[x+1][y] != 'wall':
            valid.append('east')
        if closest_x < x and percept[x-1][y] != 'wall':
            valid.append('west')
        if valid:
            return random.choice(valid)
    return random.choice(directions)

def breadth_first(percept, start_x, start_y, target_x, target_y):
    queue = deque([(start_x, start_y, [])])
    visited = set()
    
    while queue:
        x, y, path = queue.popleft()
        if (x, y) == (target_x, target_y):
            return path
        if (x, y) in visited:
            continue
        visited.add((x, y))
        
        for dx, dy, move in [(1, 0, 'east'), (-1, 0, 'west'), (0, 1, 'north'), (0, -1, 'south')]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < map_width and 0 <= new_y < map_width and percept[new_x][new_y] != 'wall':
                new_path = path + [move]
                queue.append((new_x, new_y, new_path))
    
    return [] #should never happen

def map_yes(percept, x, y):
    global path
    if path:
        next_move = path.pop(0)
        return next_move
    
    if percept[x][y] == 'dirt':
        return 'clean'
    
    dirty_distances = []
    for i in range(len(percept)):
        for j in range(len(percept[0])):
            if percept[i][j] == 'dirt':
                distance = abs(x - i) + abs(y - j)
                dirty_distances.append((distance, i, j))
    
    dirty_distances.sort()
    
    closest_dirty = dirty_distances[0]
    closest_x, closest_y = closest_dirty[1], closest_dirty[2]
    path = breadth_first(percept, x, y, closest_x, closest_y)
    if path:
        #print("path found")
        next_move = path.pop(0)
        return next_move
    
    return random.choice(directions)

def neighbors_no(percept):
    if  percept[4]=='dirt':
        return 'clean'
    dirtydirections = []
    for i in range(len(percept)):
        if percept[i] == 'dirt':
            dirtydirections.append(directions[i])
    if (dirtydirections):
        return dirtydirections[0]
    return random.choice(directions)

def neighbors_yes(percept):
    global visited_tiles
    global relative_position
    #for this function, visited_tiles will be a list of seen dirty tiles, but not tiles it's been on, so the name visited_tiles is confusing.
    if relative_position in visited_tiles:
            visited_tiles.remove(relative_position)
    if percept[4]=='dirt':
                return 'clean'
    relative_directions = {"0" : [relative_position[0], relative_position[1]+1],
                  "1" : [relative_position[0], relative_position[1]-1],
                  "2" : [relative_position[0]+1, relative_position[1]],
                  "3" : [relative_position[0]-1, relative_position[1]]
                  }
    for direction, position in relative_directions.items():
        if position not in visited_tiles and percept[int(direction)] == 'dirt':
            visited_tiles.append(position)
    #find distances to dirty tiles
    dirty_distances=[]
    for i in range(len(visited_tiles)):
        distance = abs(relative_position[0]-visited_tiles[i][0]) + abs(relative_position[1]-visited_tiles[i][1])
        dirty_distances.append((distance, visited_tiles[i]))
    #sort by distance
    dirty_distances.sort()
    if dirty_distances:
         closest_dirty = dirty_distances[0][1]
         closest_x, closest_y = closest_dirty[0], closest_dirty[1]
         valid = []
         if closest_y > relative_position[1] and percept[0] != 'wall':
             valid.append(0)
         if closest_y < relative_position[1] and percept[1] != 'wall':
             valid.append(1)
         if closest_x > relative_position[0] and percept[2] != 'wall':
             valid.append(2)
         if closest_x < relative_position[0] and percept[3] != 'wall':
             valid.append(3)
         if valid:
             valid_direction = random.choice(valid)
             next_position = relative_directions[str(valid_direction)]
             relative_position = next_position
             return directions[valid_direction]
    rand_direction = random.choice(numdirections)
    next_position = relative_directions[str(rand_direction)]
    relative_position = next_position
    return directions[rand_direction]

def single_no(percept):
    if (percept):
        return 'clean'
    return random.choice(directions)

#idea: use percept of whether the tile is still clean after making a move to determine whether the move was successful or if a wall was hit.
def single_yes(percept):
    global visited_tiles
    global relative_position
    if relative_position not in visited_tiles:
        if (len(visited_tiles)>397):
            visited_tiles.pop(0)
        visited_tiles.append(relative_position)
    if(percept):
        return 'clean'
    relative_directions = {"east" : [relative_position[0]+1, relative_position[1]],
                  "west" : [relative_position[0]-1, relative_position[1]],
                  "north" : [relative_position[0], relative_position[1]+1],
                  "south" : [relative_position[0], relative_position[1]-1]
                  }
    unvisited_neighbors = []
    for direction, position in relative_directions.items():
        if position not in visited_tiles:
            unvisited_neighbors.append(direction)
    if unvisited_neighbors:
        next_direction = random.choice(unvisited_neighbors)
        next_position = relative_directions[next_direction]
        if next_position not in visited_tiles:
            relative_position = next_position
        return next_direction
    else:
        rand_direction = random.choice(directions)
        next_position = relative_directions[rand_direction]
        relative_position = next_position
        return rand_direction

def run_all(map_width=20, max_steps=50000, runs=100):
    total_loss = 0
    mna = many_runs(map_width, max_steps, runs, map_no, 'actions', knowledge='map')
    #print("Map no actions: ", mna)
    total_loss += mna
    mnd = many_runs(map_width, max_steps, runs, map_no, 'dirt', knowledge='map')
    #print("Map no dirt: ", mnd)
    total_loss += mnd
    mya = many_runs(map_width, max_steps, runs, map_yes, 'actions', knowledge='map', agent_reset_function=memory_reset)
    #print("Map yes actions: ", mya)
    total_loss += mya
    myd = many_runs(map_width, max_steps, runs, map_yes, 'dirt', knowledge='map', agent_reset_function=memory_reset)
    #print("Map yes dirt: ", myd)
    total_loss += myd
    nna = many_runs(map_width, max_steps, runs, neighbors_no, 'actions', knowledge='neighbors')
    #print("Neighbors no actions: ", nna)
    total_loss += nna
    nnd = many_runs(map_width, max_steps, runs, neighbors_no, 'dirt', knowledge='neighbors')
    #print("Neighbors no dirt: ", nnd)
    total_loss += nnd
    nya = many_runs(map_width, max_steps, runs, neighbors_yes, 'actions', knowledge='neighbors', agent_reset_function=memory_reset)
    #print("Neighbors yes actions: ", nya)
    total_loss += nya
    nyd = many_runs(map_width, max_steps, runs, neighbors_yes, 'dirt', knowledge='neighbors', agent_reset_function=memory_reset)
    #print("Neighbors yes dirt: ", nyd)
    total_loss += nyd
    sna = many_runs(map_width, max_steps, runs, single_no, 'actions', knowledge='single')
    #print("Single no actions: ", sna)
    total_loss += sna
    snd = many_runs(map_width, max_steps, runs, single_no, 'dirt', knowledge='single')
    #print("Single no dirt: ", snd)
    total_loss += snd
    sya = many_runs(map_width, max_steps, runs, single_yes, 'actions', knowledge='single', agent_reset_function=memory_reset)
    #print("Single yes actions: ", sya)
    total_loss += sya
    syd = many_runs(map_width, max_steps, runs, single_yes, 'dirt', knowledge='single', agent_reset_function=memory_reset)
    #print("Single yes dirt: ", syd)
    total_loss += syd
    action_loss = mna+mya+nna+nya+sna+sya
    dirt_loss = mnd+myd+nnd+nyd+snd+syd
    #print("Total action loss: ", action_loss)
    #print("Total dirt loss: ", dirt_loss)
    #print("Total loss divided by action loss: ", total_loss/action_loss)
    #print("Total loss divided by dirt loss: ", total_loss/dirt_loss)


    return total_loss

total_loss = run_all()
print("Total loss: ", total_loss)
