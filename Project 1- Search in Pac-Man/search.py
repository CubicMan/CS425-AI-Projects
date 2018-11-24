# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

'''
psuedo code:

function graph-search(problem, fringe, strategy)
    #initialize an empty closed set
    #initalize a fringe
    
    while fringe is not empty:
        remove node from fringe
        if state[node] is not in closed:
            add state[node] to closedSet
        for child-node in expanded(state[node], problem):
            insert(childnode, fringe)

'''

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #print "Start:", problem.getStartState() #(5,5)
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState()) #False
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())
    #[((5, 4), 'South', 1), ((4, 5), 'West', 1)]

    #successors = problem.getSuccessors(problem.getStartState())
    #print ("successors[0]", successors[0]) #((5, 4), 'South', 1))

    closedSet = [] #initalize an empty set called closedSet
    fringe = util.Stack() #initialize the fringe using the stack data structure for DFS
    startState = problem.getStartState() #insert initial state onto the fringe
    fringe.push((startState, [], 0))

    while not fringe.isEmpty():
        node = fringe.pop() # node = removed node from the fringe
        position = node[0] #represents current position

        # print "node[0]", node[0] # (5,5)
        # print "node[1]", node[1] # [] = direction NORTH, EAST, SOUTH, WEST
        # print "node[2]", node[2] # 0 = step cost usually 1

        if problem.isGoalState(position):  # if the position is a valid GoalState (x,y)
            return node[1]  # return list of directions that gets to the state

        if not position in closedSet: #if state[node] is not in closedSet
            closedSet.append(position) #add to the closedSet
            for childNode in problem.getSuccessors(position):
                    fringe.push((childNode[0], node[1] + [childNode[1]], childNode[2]))
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    closedSet = []  # initalize an empty set called closedSet
    fringe = util.Queue()  # initialize the queue for DFS
    startState = problem.getStartState()  # insert initial state onto the fringe
    fringe.push((startState, [], 0))

    while not fringe.isEmpty():
        node = fringe.pop()  # node = removed node from the fringe
        position = node[0] #represents current position
        if problem.isGoalState(position):  # if the state is a valid GoalState (x,y)
            return node[1]  # return list of directions that gets to the state

        if not position in closedSet:  # if state[node] is not in closedSet
            closedSet.append(position)  # add to the closedSet
            for childNode in problem.getSuccessors(position):
                    fringe.push((childNode[0], node[1] + [childNode[1]], childNode[2]))

    if fringe.isEmpty(): #if fringe is empty:
        return node[1] #return list of directions to get to state

    util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"

    closedSet = []  # initalize an empty set called closedSet
    fringe = util.PriorityQueue()  # initialize the queue for UCS
    startState = problem.getStartState()  # insert initial state onto the fringe
    fringe.push((startState, [], 0), 0) #push two arguments - (x,y, [],0) and priority - 0

    while not fringe.isEmpty():
        node = fringe.pop()  # node = removed node from the fringe
        position = node[0] #represents current position

        if problem.isGoalState(position):  # if the state is a valid GoalState (x,y)
            return node[1]  # return list of directions that gets to the state

        if not position in closedSet:  # if state[node] is not in closedSet
            closedSet.append(position)  # add to the closedSet
            for childNode in problem.getSuccessors(position):
                # gets the position of the next node, adds the directions, gets the cost of current node and next node
                    successor = (childNode[0], node[1] + [childNode[1]], childNode[2] + node[2])
                    fringe.push(successor, childNode[2] + node[2]) #push onto the fringe

    if fringe.isEmpty(): #if fringe is empty:
        return node[1] #return list of directions to get to state

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"

    closedSet = [] # initalize an empty set called closedSet
    fringe = util.PriorityQueue() # initialize the queue for UCS
    startState = problem.getStartState() # insert initial state onto the fringe
    fringe.push((startState, [], 0), 0) #push two arguments - (x,y, [],0) and priority - 0

    while not fringe.isEmpty():
        node = fringe.pop() # node = removed node from the fringe
        position = node[0] #represents current position

        if problem.isGoalState(position): # if the state is a valid GoalState (x,y)
            return node[1] # return list of directions that gets to the state
        if not position in closedSet:  # if state[node] is not in closedSet
            closedSet.append(position) #add to the closeSet
            for childNode in problem.getSuccessors(node[0]):
                    # gets the position of the next node, adds the directions, gets the cost of current node and next node
                    successor = (childNode[0], node[1] + [childNode[1]], node[2] + childNode[2])
                    totalCost = successor[2] + heuristic(childNode[0], problem) #gets cost
                    fringe.push(successor, totalCost) #push onto fringe

    if fringe.isEmpty(): #if fringe is empty:
        return node[1] #return list of directions to get to state

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
