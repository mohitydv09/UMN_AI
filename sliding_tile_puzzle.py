import sys
from collections import deque
from time import time
import heapq

class Node:
    """Node is a element in the search tree for the problem.
    Takes as input the state of the node, parents of the node,
    action that got us to this node, and path cost."""
    def __init__(self,state,parent=None,action=None,path_cost=0) -> None:
        self.state=state
        self.parent=parent
        self.action=action
        self.depth=self.parent.depth + 1 if self.parent else 0
        self.path_cost=path_cost

    def __repr__(self):
        out_str='\n-------------\n'
        for i in range(3):
            out_str+= "| "
            for j in range(3):
                out_str+=str(self.state[3*i+j])+" | "
            out_str+='\n-------------\n'
        return f"Node object: {out_str}"

    def child_node(self,problem,action):
        """Return the node that is reached form the given node,
          if the given action is taken."""
        next_state=problem.result(self.state,action)
        next_node=Node(next_state,self,action,problem.path_cost(self.path_cost,self.state,action,next_state))
        return next_node

    def expand(self,problem):
        """List the nodes reachable in one step from this node. 
        This takes into consideration the actions that are possible form current node."""
        return [self.child_node(problem,action) for action in problem.actions(self.state)]

    def path(self):
        """Returns the path from the root node to the current node."""
        node=self
        path=[]
        while node:
            path.append(node)
            node=node.parent
        return list(reversed(path))

    def solution(self):
        """Returns a list of actions that were taken to reach this node."""
        #Fisrt element of path is removed as it is root node.
        return [node.action for node in self.path()[1:]]

    def __eq__(self, other_node) -> bool:
        """Nodes with same state are considered equal."""
        return isinstance(other_node,Node) and self.state==other_node.state
    
    def __hash__(self) -> int:
        """Node are hashed for fast checking."""
        return hash(self.state)
        
    def __lt__(self,node):
        return self.state<node.state

class Problem:
    """Problem class contains funtions pertaing to the behaviours as specified in the problem"""
    def  __init__(self,initial,goal=(1,2,3,8,0,4,7,6,5)) -> None:
        self.initial=initial
        self.goal=goal

    def goal_test(self,state):
        #Equality operator used here was defined by using the magin method __eq__ in Node class.
        return self.goal==state

    def index_of_blank(self,state):
        """Return the index of the blank square in a given state."""
        return state.index(0)
    
    def path_cost(self,c,state1,action,state2):
        """Gived the total path cost to reach state2 via state1 with action."""
        # c is the cost to reach state1.
        # Here all the costs are 1, therefore cost form 1 to 2 is 1. Will change depending on the question.
        cost_1_to_2=1
        return c+cost_1_to_2

    def actions(self,state):
        """Returns the list of actions that are possible from the given state."""
        possible_actions=["U","D","R","L"]
        index_blank=self.index_of_blank(state)

        #When blank is in the top row.
        if index_blank<3:
            possible_actions.remove("D")
        #When blank is in the last row.
        if index_blank>5:
            possible_actions.remove("U")
        #When blank is in first column.
        if index_blank%3==0:
            possible_actions.remove("R")
        #When blank is in last column.
        if index_blank%3==2:
            possible_actions.remove("L")
        
        return possible_actions
    
    def result(self, state, action):
        """Returns the new state for given state and action."""
        index_blank=self.index_of_blank(state)
        next_state=list(state)

        delta={ "U" : 3 ,"D" : -3 , "R" : -1 , "L": 1 }
        neighbor_index=index_blank+delta[action]
        #Swap the elements.
        # As state is a tuple therefore it won't be changed here,
        # if it were a list you should have copied using copy.copy().
        next_state[index_blank],next_state[neighbor_index]=next_state[neighbor_index],next_state[index_blank]
        return tuple(next_state)

    def h_num_wrong_tiles(self,node):
        """Return the heuristic(Num of misplaced tiles) value for a given state."""
        # return sum([x!=y for (x,y) in zip(self.goal,node.state)])
        num_of_wrong_tiles=0
        for i in range(1,len(node.state)):
            target_index=self.goal.index(i)
            current_index=node.state.index(i)
            if target_index!=current_index:
                num_of_wrong_tiles+=1
        return num_of_wrong_tiles

    def h_manhattan_distance(self,node):
        """Return the heuristic(Manhattan distance) value for a given state."""
        sum_of_dist=0
        #Range starts from 1 as distance of blank is not cosidered.
        for i in range(1,len(node.state)):
            target_index=self.goal.index(i)
            current_index=node.state.index(i)

            row_target=target_index//3
            row_current=current_index//3
            column_target=target_index%3
            column_current=current_index%3

            man_dist=abs(row_current-row_target) + abs(column_current-column_target)
            sum_of_dist+=man_dist
        return sum_of_dist

class PriorityQueue:
    """Implementation of Priority Queue."""
    def __init__(self, f=lambda x : x) -> None:
        self.heap=[]
        self.f=f
        
    def append(self,item):
        """Insert item with f(item) priority"""
        heapq.heappush(self.heap,(self.f(item),item))

    def pop(self):
        #As f won't be required we don't return it when pop is called.
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception("Pop was called on a empty Priority Queue.")
        
    def __contains__(self,key):
        """Returns True if the key is in PQ"""
        return any([item==key for _,item in self.heap])

    def __delitem__(self,key):
        """Will only delete the first occurance of key."""
        try:
            del self.heap[[item==key for _,item in self.heap].index(True)]
        except:
            raise KeyError(str(key)+" is not in the priority queue.")
        heapq.heapify(self.heap)
            
    def __getitem__(self,key):
        for value, item in self.heap:
            if item==key:
                return value
        raise KeyError(str(key)+" is not in the PQ")

def breadth_first(problem):
    #Initialize the first element as node to check if it is same as goal state.
    node=Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    
    frontier=deque([node])
    explored=set()
    while frontier:
        node=frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None

def dls(problem,limit=sys.maxsize):
    frontier=[(Node(problem.initial))]  #Stack // LiFo
    #Explored is not used here as it stopped the solution from getting to the optimum solution,
    # a node which was added to the frointier and was removed as the solution was not found from it
    # in the limited depth. but this node could have been formed from some other path with less pathcost
    # from where it could have led to the solution within the limit. 
    cutoff_occurred=False
    while frontier:
        node=frontier.pop()
        if problem.goal_test(node.state):
            return node
        for child in node.expand(problem):
            if child not in frontier:
                if child.depth<=limit:
                    frontier.append(child)
                else:
                    cutoff_occurred=True
    return 'cutoff' if cutoff_occurred else None

def iterative_deepening(problem):
    for depth in range(sys.maxsize):
        result=dls(problem,depth)
        if result!="cutoff":
            return result

def astar(problem,h=None):
    #will be implemented by best first search with given hueristic.
    return best_first_search(problem,lambda n : n.path_cost + h(n))

def best_first_search(problem,f):
    """Search the nodes with lowest f score first."""
    node=Node(problem.initial)
    frontier=PriorityQueue(f)
    frontier.append(node)
    explored=set()
    while frontier:
        node=frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None

### Code doesn't validates the inputs, please input in correct format only,
### Corrent format is command line argument as : "python3 eight_puzzel.py 123456780",
### where the last number represent the initial state.
### This code also doesn't check for solvability, therefore only a solvable state must be provided.

raw_user_input=sys.argv[1]
user_input=tuple(map(int,raw_user_input))

#initialize Puzzel with the initial state.
puzzle_instance=Problem(user_input)

#BFS
tic=time()
solution_bfs=breadth_first(puzzle_instance)
toc=time()

if solution_bfs:
    path_to_solution_bfs=solution_bfs.solution()
    print([x for x in path_to_solution_bfs])
    print(f"BFS took {(toc-tic):.20f} seconds")
else:
    print("BFS was not able to find a solution to the specified initial state.")

# IDDFS
tic_iddfs=time()
solution_iddfs=iterative_deepening(puzzle_instance)
toc_iddfs=time()

if solution_iddfs:
    path_to_solution_iddfs=solution_iddfs.solution()
    print([x for x in path_to_solution_iddfs])
    print(f"IDDFS took {(toc_iddfs-tic_iddfs):.20f} seconds")
else:
    print("DFS was not able to find a solution to the specified initial state.")

# A* search with num of wrong tiles huiristics
tic_astar_num_wrong_tile=time()
solution_astar_num_wrong_tile=astar(puzzle_instance,puzzle_instance.h_num_wrong_tiles)
toc_astar_num_wrong_tile=time()

if solution_astar_num_wrong_tile:
    path_to_solution_astar_num_wrong_tile=solution_astar_num_wrong_tile.solution()
    print([x for x in path_to_solution_astar_num_wrong_tile])
    print(f"A* using num_wrong_tiles took {(toc_astar_num_wrong_tile-tic_astar_num_wrong_tile):.20f} seconds")
else:
    print("DFS was not able to find a solution to the specified initial state.")


# A* search with manhattan distance tiles huiristics
tic_astar_manhattan_dist=time()
solution_astar_manhattan_dist=astar(puzzle_instance,puzzle_instance.h_manhattan_distance)
toc_astar_manhattan_dist=time()

if solution_astar_manhattan_dist:
    path_to_solution_astar_manhattan_dist=solution_astar_manhattan_dist.solution()
    print([x for x in path_to_solution_astar_manhattan_dist])
    print(f"A* using manhattan_dist took {(toc_astar_manhattan_dist-tic_astar_manhattan_dist):.20f} seconds")
else:
    print("DFS was not able to find a solution to the specified initial state.")
