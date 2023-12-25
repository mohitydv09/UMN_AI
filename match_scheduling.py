import random
import copy
import matplotlib.pyplot as plt
import math

random.seed(45)

def initial_state():
    #Generate list of characters for later use.
    players=[chr(x) for x in range(ord("A"),ord('M')+1)]

    initial_state={}
    for player in players:
        initial_state[player]=[[],[],[]]
        for i,player_new in enumerate(players):
            if len(initial_state[player][0]) < 4 and player!=player_new:
                initial_state[player][0].append(player_new)
            elif len(initial_state[player][1]) < 4 and player!=player_new:
                initial_state[player][1].append(player_new)
            elif player!=player_new:
                initial_state[player][2].append(player_new)

    # Add some randomness to the intial state.
    for key,value in initial_state.items():
        for i in range(10):
            game1 , game2 = int(random.uniform(0,3)), int(random.uniform(0,3))
            player1, player2 = int(random.uniform(0,4)), int(random.uniform(0,4))
            value[game1][player1] , value[game2][player2] = value[game2][player2] , value[game1][player1]
    return initial_state

class Node:
    """Node is a element in the search space."""
    def __init__(self,state):
        self.state = state
        self.value = self.objective_funtion()

    def __repr__(self) -> str:
        return_string=''
        for item,value in self.state.items():
            return_string += "Bye for Player " + str(item) + ": "
            for match in value:
                return_string += str(match) + ' '
            return_string += '\n'
        return return_string
    
    def __hash__(self) -> int:
        return hash(self.state)
    
    def match_counter(self):
        #Initialize a new dict to keep the counter with zero values.
        counter_dict={}
        for player in self.state:
            counter_dict[player]={}
            for player_j in self.state:
                if player != player_j:
                    counter_dict[player][player_j]=0

        # Find the number of matches between all the teams.
        for item,values in self.state.items():
            for match in values:
                for i in range(4):
                    for j in range(i+1,4):
                        counter_dict[match[i]][match[j]]+=1
                        counter_dict[match[j]][match[i]]+=1
        return counter_dict

    def zero_counter(self,counter_dict):
        counter=0
        for key, value in counter_dict.items():
            for key2, value2 in value.items():
                if value2==0:
                    counter+=1
        return counter

    def repeat_match_counter(self):
        list_of_all_matches=[]
        for key, value in self.state.items():
            for item in value:
                list_of_all_matches.append("".join(sorted(item)))
        return len(list_of_all_matches)-len(set(list_of_all_matches))

    def objective_funtion(self):
        ## Value due to match againt each other counter.
        obj1=0
        counter_dict = self.match_counter()
        for key, value in counter_dict.items():
            for key2, value2 in value.items():
                obj1+=value2**4

        ##Value due to no matched in two teams.
        penalty_for_no_matches = 1000
        zero_matched=self.zero_counter(counter_dict)
        obj2 = zero_matched * penalty_for_no_matches

        # Value due to repeat match
        penalty_for_repeat_match = 500
        repeat_matches = self.repeat_match_counter()
        obj3 = repeat_matches * penalty_for_repeat_match
        # print("Objective value disected : ", obj1 , obj2 , obj3)
        return obj1 + obj2 + obj3

    def next_steps(self):
        """Return the neighbours possible from this state who have lower value than self, and the values of the items too."""
        ## Empty list to keep the copy of states, this can be transversed easily
        neighbours=[]

        # Go to matches in each week.
        for key in self.state:
            ## Swapping the items in first match with items in second and third match.
            ## i,j to transverse matches.
            for i in range(3):
                for j in range(i+1,3):
                    ## k,l to transverse inside matches with players.
                    for k in range(4):
                        for l in range(4):
                            ## Create a copy each time and then save the resulting matrix in the list.
                            copy_state = copy.deepcopy(self.state)
                            copy_state[key][i][k] , copy_state[key][j][l] = copy_state[key][j][l] , copy_state[key][i][k] 
                            ##Create a node of each of this modified state and store in the list.
                            node_of_state = Node(copy_state)
                            if node_of_state.value <= self.value:
                                neighbours.append(node_of_state)
        # print(f"Length of neighbour is : {len(neighbours)}")
        if len(neighbours)==0:
            return [],[]
        weights=[]
        for item in neighbours:
            weights.append(item.value)
        max_weight=max(weights)
        final_weights=[]
        for item in weights:
            final_weights.append(max_weight-item+1)
        return neighbours , final_weights
    

    def next_steps_SA(self):
        """Return the neighbours possible from this state."""
        ## Empty list to keep the copy of states, this can be transversed easily
        neighbours=[]

        # Go to matches in each week.
        for key in self.state:
            ## Swapping the items in first match with items in second and third match.
            ## i,j to transverse matches.
            for i in range(3):
                for j in range(i+1,3):
                    ## k,l to transverse inside matches with players.
                    for k in range(4):
                        for l in range(4):
                            ## Create a copy each time and then save the resulting matrix in the list.
                            copy_state = copy.deepcopy(self.state)
                            copy_state[key][i][k] , copy_state[key][j][l] = copy_state[key][j][l] , copy_state[key][i][k] 
                            ##Create a node of each of this modified state and store in the list.
                            node_of_state=Node(copy_state)
                            neighbours.append(node_of_state)
        weights=[]
        for item in neighbours:
            weights.append(item.value)
        max_weight=max(weights)
        final_weights=[]
        for item in weights:
            final_weights.append(max_weight-item+1)
        return neighbours , final_weights

def hill_decent(start_node ):
    """Stochastic Hill Climb implementation."""
    current = start_node
    value_current=current.value
    value_tracker=[value_current]
    for i in range(10000):
        neighbours , weights = current.next_steps()
        if len(neighbours) == 0:
            return current, value_tracker
        current = random.choices(neighbours,weights)[0]
        value_tracker.append(current.value)
    print('Ended after 10k iters')
    return current, value_tracker

def simulated_annealing(start_node,num_of_iter):
    current = start_node
    obj_tracker=[current.value]
    print("\n \nSimulated Annealing Started.")
    for t in range(num_of_iter):
        if t%50==0: print(f"SA Running, Current Iter No is: {t} , will run till {num_of_iter}")
        T = schedule(t,num_of_iter)
        possible_successors, weights = current.next_steps_SA()
        next = random.choices(possible_successors,weights)[0]
        if T==0 : return current, obj_tracker
        delta_E = current.value - next.value
        # print(f"Current T is: {T:.5f} and Delta E for this: {delta_E}")
        ## If next has lower obj. funtion then change the node for sure.
        if delta_E > 0:
            current = next
        ## Even if next increases your objective function do it sometimes.
        elif random.random() < (math.e)**(delta_E/T):
            current = next
        obj_tracker.append(current.value)

def schedule(t,num_of_iter):
    # return num_of_iter - t -1
    return 10000*(((1/(t+1)) - (1/num_of_iter))**0.8)

## Initial State Initialization.
initial_state__ = initial_state()
initial_node = Node(initial_state__)
print(f"Initial Node:\n{initial_node}")

## For checking the num of matches between two players.
counter_dict=initial_node.match_counter()
print("Initial Match Counter: ")
for key,value in counter_dict.items():
    print( key ,":", value)


### Solution for Hill Decent Algorithm.
result , value_tracker = hill_decent(initial_node)

print(f"Final Node obtained with Hill Decent:\n{result}")

## For checking the num of matches between two players.
counter_dict_final=result.match_counter()
print("** Replay Summary **")
for key,value in counter_dict_final.items():
    print( key ,":", value)

## Printing some stats for use.
print(f"\nObjective funtion of the final result of Hill Decent         :{result.value}")
print("Number of players who had zero matches with Hill Decent      : ",result.zero_counter(result.match_counter()))
print("Number of repeat matches in the final result with Hill Decent: ", result.repeat_match_counter() )


### Solution for Hill Decent Algorithm.
num_of_iters = 200
result_anne, value_tracker_anne = simulated_annealing(initial_node,num_of_iters)
print(f"Final Node with SA:\n{result_anne}")

## For checking the num of matches between two players.
counter_dict_final=result_anne.match_counter()
print("** Replay Summary **")
for key,value in counter_dict_final.items():
    print( key ,":", value)

## Printing some stats for use.
print(f"\nObjective funtion of the final result with SA     :  {result_anne.value}")
print("Number of players who had zero matches with SA    : ",result_anne.zero_counter(result_anne.match_counter()))
print("Number of repeat matches in the final result of SA: ", result_anne.repeat_match_counter() )


### For Plotting the result
## Making value tracker of 500 nodes.
if len(value_tracker)!=num_of_iters:
    to_add = [value_tracker[-1]] * (num_of_iters - len(value_tracker))
    value_tracker.extend(to_add)

## For plotting the Objective Funtion Over time
#Generate x value for graph.
plot_x=[x for x in range(num_of_iters)]
plt.plot(plot_x,value_tracker_anne, label = "Simulated Annealing")
plt.plot(plot_x,value_tracker, label="Stochastic Hill Decent")
plt.title("Obj. function")
plt.xlabel('Num. of Iterations')
plt.ylabel("Obj. Function")
plt.legend()
plt.show()
