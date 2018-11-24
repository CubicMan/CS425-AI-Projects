# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # should return 0.0 if the (state, action) is not in the counter
        if (state, action) not in self.qValues:
            self.qValues[(state, action)] = 0.0 #set to zero
            return 0.0 #return 0.0

        else:
            return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #get the possible actions
        actionList = self.getLegalActions(state)
        max = 0
        qValues = util.Counter()
        maxValue = float("-inf")  # intialize to negativity infinity

        #if there are no legal actions
        if not actionList:
            return 0.0 #return 0.0

        for action in actionList:
            #gets the Qvalue
            value = self.getQValue(state,action)
            #sets the value to the corresponding element in qValues Counter
            qValues[action] = value

            if value > maxValue:  # if the current value is better than the max value
                maxValue = value  # set maxValue to the greater value

        return maxValue #return maxValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actionList = self.getLegalActions(state) #gets list of Actions
        qValues = util.Counter() #stores the QValues
        bestAction = None #best action
        currentValue = float("-inf")  # intialize to negativity infinity

        # if there are not legal actions:
        if not actionList:
            return None  # return None

        for actions in actionList:
            #gets the QValue
            value = self.getQValue(state, actions)
            # sets the value to the corresponding element in qValues Counter
            qValues[actions] = value

            if value > currentValue:  # if the current value is better than the current value
                currentValue = value  # set currentValue to the greater value
                bestAction = actions  # store the best action

        return bestAction #return the bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # if there are not legal actions:
        if not legalActions:
            return None #return None

        #use probability self.epsilon in util.flipCoin(prob)
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions) #pick randomly from list
            return action

        #take the best policy action otherwise
        else:
            action = self.getPolicy(state)
            return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #uses the formula
        first = (1 - self.alpha) * self.getQValue(state, action)
        second = reward + self.discount * self.getValue(nextState)
        self.qValues[(state, action)] = first + self.alpha * second

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights #return weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # return weight * featureVector
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # stores the featExtractor
        featureVar = self.featExtractor.getFeatures(state, action)
        #stores difference
        difference = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)

        for feature in featureVar:
            #weight = weight + self.alpha * difference * features(state, action)
            self.weights[feature] = self.weights[feature] + self.alpha * difference * featureVar[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print self.weights # print weights
            #pass
