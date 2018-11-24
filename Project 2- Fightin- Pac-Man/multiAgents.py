# multiAgents.py
# --------------
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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def manhattanDistance(xy1, xy2):
        "Returns the Manhattan distance between points xy1 and xy2"
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules = currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules = successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        foodCounter = currentGameState.getNumFood()
        # Uses inverse of Manhattan distance
        # for food, use the minimum of the distances
        # for ghosts, use the maximum of the distances

        "*** YOUR CODE HERE ***"
        foodlist = currentGameState.getFood().asList()
        infinity = float("inf")
        negativeInfinity = float("-inf")
        ghostDistance = negativeInfinity
        getFoodNumber = 1.0 / (1.0 + foodCounter)

        # Minimize distance from food
        for food in foodlist:
            # gets the minimum distance from the food to the current position
            minDistance = min(infinity, manhattanDistance(food, newPos))
            # if stopped, return infinity
            if Directions.STOP in action:
                return negativeInfinity  # return negative infinity

        #Maximize distance from ghosts
        for ghost in newGhostStates:
            #gets the ghost position
            ghostPosition = ghost.getPosition()
            if ghostPosition == newPos:  # if the pacman, hits a ghost
                return negativeInfinity  # return negative infinity
            else:
                ghostDistance = max(ghostDistance, manhattanDistance(newPos, ghostPosition))

        # inverse of manhattan distance from food to position
        inverseDistance = 1.0 / (1.0 + minDistance)
        # inverse of manhattan distance from ghost to position
        inverseGhostDistance = 1.0 / (1.0 + ghostDistance)
        #inverse of food number
        getFoodNumber = 1.0 / (1.0 + foodCounter)

        # return the inverse of the food and ghost distance
        return inverseDistance + inverseGhostDistance

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    # take the code from the slides
    # for pacman you run the max
    # for ghost you run the min
    # game.terminal_test = legalAction isn't empty
    # when depth is one, you pass the action all the way down the tree
    # when depth is anything else you pass the action that has already been passed
    # if depth one, pass the action I'm currently looking
    # if depth isn't one, I pass the action down the recursion
    # return the tuple with weight and action

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        zero = 0
        return self.value(gameState, zero, zero)

    def value(self, gameState, agentIndex, depth):
        num_agents = gameState.getNumAgents()  # number of agents

        #once agentIndex is greater than number of agents, go back to pacman
        if agentIndex > num_agents or agentIndex == num_agents:
            agentIndex = 0  # agentIndex = 0 means Pacman
            depth = depth + 1

        # if the depth is reached or the gameState is won or lost
        if (depth - self.depth) == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)  # return evaluationFunction

        #when agentIndex is zero, you want to maximize for pacman
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)

        # when agentIndex is not zero, you want to minimize for the agents
        elif agentIndex != 0:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        valueofState = float("-inf") #value if initialized to negative infinity
        valueofAction = None
        nextAgent = agentIndex + 1

        for action in gameState.getLegalActions(agentIndex):
            next = gameState.generateSuccessor(agentIndex, action)
            currentValue = self.value(next, agentIndex + 1, depth)

            # compares the currentValue with the valueoftheState
            if currentValue > valueofState:
                # gets the max value
                valueofState = currentValue
                valueofAction = action

        if depth == 0:
            return valueofAction
        elif depth > 0:
            return valueofState

    def minValue(self, gameState, agentIndex, depth):
        valueofState = float("inf")  # value if initialized to infinity
        valueofAction = None
        nextAgent = agentIndex + 1

        for action in gameState.getLegalActions(agentIndex):
            next = gameState.generateSuccessor(agentIndex, action)
            currentValue = self.value(next, agentIndex + 1, depth)

            # compares the currentValue with the valueoftheState
            if currentValue < valueofState:
                # gets the min value
                valueofState = currentValue
                valueofAction = action

        return valueofState

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBetaValue(self, gameState, agentIndex, depth, alpha, beta):
        number_agents = gameState.getNumAgents() #number of agents

        if agentIndex > number_agents or agentIndex == number_agents:
            agentIndex = 0
            depth = depth + 1

        # call evaluationFunction if the game is won/lost or depth is reached
        if (depth - self.depth) == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if (agentIndex - self.index) == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        elif (agentIndex - self.index) != 0:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        valueofState = float("-inf")
        valueofAction = None

        for actions in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, actions)
            currentValue = self.alphaBetaValue(successor, agentIndex + 1, depth, alpha, beta)

            if currentValue > valueofState:
                valueofState = currentValue
                valueofAction = actions

            if valueofState > beta:
                return valueofState

            # alpha = max(alpha, valueofState)
            if alpha < valueofState:
                alpha = valueofState

        # when depth is zero, return the action
        if depth == 0:
            return valueofAction
        # when depth is > 0, return the value
        elif depth > 0:
            return valueofState

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
            valueofState = float("inf")
            valueofAction = None

            for actions in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, actions)
                currentValue = self.alphaBetaValue(successor, agentIndex + 1, depth, alpha, beta)
                nextAgent = agentIndex + 1

                if currentValue < valueofState:
                    valueofState = currentValue
                    actionValue = actions

                if valueofState < alpha:
                    return valueofState

                if beta > valueofState:
                    beta = valueofState


            return valueofState

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        zero = 0
        negative = float("-inf")
        positive = float("inf")

        return self.alphaBetaValue(gameState, zero, zero, negative, positive)
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimaxValue(self, gameState, agentIndex, depth):
        num_agents = gameState.getNumAgents()  # number of agents
        nextAgent = agentIndex + 1

        # call evaluationFunction if the game is won/lost or depth is reached
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex > num_agents or agentIndex == num_agents:
            agentIndex = 0
            depth = depth + 1

        # call evaluationFunction if the game is won/lost or depth is reached
        if (depth - self.depth) == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if (agentIndex - self.index) == 0:
            return self.maxValue(gameState, agentIndex, depth)
        elif (agentIndex - self.index) != 0:
            return self.expValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        valueofState = float("-inf")  # initialize to negative infinity
        valueofAction = None
        nextAgent = agentIndex + 1

        for actions in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, actions)
            currentValue = self.expectimaxValue(successor, agentIndex + 1, depth)

            if currentValue > valueofState:
                valueofState = currentValue
                valueofAction = actions

        # when depth is zero, return the action
        if depth == 0:
            return valueofAction
        # when depth is > 0, return the value
        elif depth > 0:
            return valueofState

    def expValue(self, gameState, agentIndex, depth):
        possibleActions = gameState.getLegalActions(agentIndex)
        zero = 0
        valueofState = zero
        nextAgent = agentIndex + 1

        probabilityValue = 1.0 / len(possibleActions)

        for actions in possibleActions:
            successor = gameState.generateSuccessor(agentIndex, actions)
            current = self.expectimaxValue(successor, agentIndex + 1, depth)
            nextAgent = agentIndex + 1
            factor = current * probabilityValue
            valueofState = valueofState + factor

        return valueofState

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        zero = 0
        return self.expectimaxValue(gameState, zero, zero)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      # sums the current score + inverse distance from ghosts + inverse distance from food
      # currentScore and inverse food distance has 0.5 weight, inverse distance from ghost is given 1.5 weight
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()  # food available from successor state (excludes food@successor)
    currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    currentScore = currentGameState.getScore()
    foodlist = currentGameState.getFood().asList()

    negativeInfinity = float("-inf")
    infinity = float("inf")
    ghostDistance = infinity
    minDistance = 0

    for ghost in newGhostStates:
        ghostPosition = ghost.getPosition() #gets the ghost position
        if ghostPosition == newPos:  # if the pacman, hits a ghost
            return negativeInfinity  # return negative infinity
        else:
            ghostDistance = min(ghostDistance, manhattanDistance(newPos, ghostPosition))

        # Minimize distance from food
        for food in foodlist:
            # gets the minimum distance from the food to the current position
            minDistance = min(infinity, manhattanDistance(food, newPos))

    # inverse of manhattan distance
    inverseDistance = 1.0 / (1.0 + minDistance)
    inverseGhostDistance = 1.0 / (1.0 + ghostDistance)

    return currentScore * 0.5 + inverseDistance * 0.5 + inverseGhostDistance * 1.5
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
