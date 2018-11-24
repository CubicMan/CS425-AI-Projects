# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #run the indicated number of iterations:
        for iteration in range(iterations):
            counter = util.Counter()

            #take an mdp on construction
            for state in mdp.getStates():
                list = []  # list of qValues
                #if its a terminal state
                if mdp.isTerminal(state):
                    list.append(0)
                    counter[state] = 0 #set to zero

                possibleActions = mdp.getPossibleActions(state)
                qMax = float("-inf")

                for actions in possibleActions:
                    value = self.computeQValueFromValues(state, actions)
                    list.append(value)

                counter[state] = max(list) #set the counter to the maxValue in list

            self.values = counter #set self.values counter to counter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        stateProbs = self.mdp.getTransitionStatesAndProbs(state, action)

        for newState, probability in stateProbs:
            reward = self.mdp.getReward(state,action,newState)
            discount = self.discount * self.getValue(newState)
            value = value + probability * (reward + discount)

        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        currentValue = float("-inf")  # intialize to negativity infinity
        actionList = self.mdp.getPossibleActions(state) #list of possible actions
        bestAction = None  # stores the best action

        #if there are not legal action
        if not actionList:
            return None #return None

        list = []  # stores values

        for action in actionList:
            value = self.computeQValueFromValues(state, action)
            list.append(value) #stores values

            if value > currentValue: #if the current value is better than the current value
                currentValue = value #set currentValue to the greater value
                bestAction = action #store the best action

        return bestAction #return best action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
