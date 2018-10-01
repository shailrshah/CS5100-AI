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

import util
from util import manhattanDistance
import random
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

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Decompose positions from the successor state
        pacman = successorGameState.getPacmanPosition()
        newGhosts = map(lambda ghostState: ghostState.getPosition(), successorGameState.getGhostStates())
        newFoods = currentGameState.getFood().asList()

        """
        If the new successor and one of the ghost's are at the same position, it's a bad move
        If the action is 'Stop', it's also a bad move
        The evaluation function is essentially the maximum negative
        manhattan distance between Pac-Man's position and the new Foods' positions
        """
        distance, STOP = -float("inf"), 'Stop'
        return distance if pacman in newGhosts or action == STOP \
            else [max(distance, -manhattanDistance(food, pacman)) for food in newFoods]


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
        self.NO_ACTION = "NoAction"


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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
        # Return the action
        return self.helper(gameState, 0, 0)[0]

    def helper(self, gameState, currentAgent, currentDepth):

        # Update currentDepth and currentAgentIndex
        if currentAgent == gameState.getNumAgents():
            currentDepth += 1
        currentAgent %= gameState.getNumAgents()

        # Base case
        if currentDepth == self.depth or not gameState.getLegalActions(currentAgent):
            return self.NO_ACTION, self.evaluationFunction(gameState)

        # Initialize variables according to Maximizer/Minimizer
        isMaximizer = currentAgent is self.index
        startValue = (-1 if isMaximizer else 1) * float("inf")

        action, value = (self.NO_ACTION, startValue)
        # Calculate the action, value tuple
        for nextAction in gameState.getLegalActions(currentAgent):
            nextValue = self.helper(gameState.generateSuccessor(currentAgent, nextAction),
                                    currentAgent + 1, currentDepth)[1]
            nextValue = max(value, nextValue) if isMaximizer else min(value, nextValue)
            action, value = (nextAction if nextValue != value else action, nextValue)

        return action, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        return self.value(gameState, 0, 0)[0]

    def value(self, gameState, currentAgent, currentDepth):

        # Update currentDepth and currentAgentIndex
        if currentAgent == gameState.getNumAgents():
            currentDepth += 1
        currentAgent %= gameState.getNumAgents()

        # Base case
        if currentDepth == self.depth or not gameState.getLegalActions(currentAgent):
            return self.NO_ACTION, self.evaluationFunction(gameState)

        # Initialize variables according to Maximizer/Minimizer
        isMaximizer = currentAgent is self.index
        startValue = -1*float("inf") if isMaximizer else 0
        uniformProbability = 1.0 / len(gameState.getLegalActions(currentAgent))

        # Calculate the action, value tuple
        action, value = self.NO_ACTION, startValue
        for nextAction in gameState.getLegalActions(currentAgent):
            nextValue = self.value(gameState.generateSuccessor(currentAgent, nextAction),
                                   currentAgent + 1, currentDepth)[1]
            newValue = max(value, nextValue) if isMaximizer else value + (nextValue * uniformProbability)
            action, value = nextAction if value != newValue else action, newValue

        return action, value


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: We can get a score from Pac-Man's position, the number of remaining food to eat, and the ghosts.
      If there is nothing left to eat, we have won and so we return a very high score.
      For each ghost, we check its distance to Pac-Man. If the ghost is scared and Pac-Man can reach it in time,
      we add the distance to the score. Otherwise, we subtract the distance from the score.
    """
    # Win state found (No food left to eat). Return a very high score
    if len(currentGameState.getFood().asList()) == 0:
        return 99999999999999999999999

    # Initialize the score to be the current score
    score = currentGameState.getScore()
    for ghost in currentGameState.getGhostStates():

        # Get the Manhattan Distance between Pac-Man and the ghost
        pacManToGhostDistance = manhattanDistance(currentGameState.getPacmanPosition(), ghost.getPosition())

        # If Pac-Man can reach the ghost in time and the ghost is scared, add distance to the score
        # Otherwise, subtract distance from the score
        multiplier = 1 if ghost.scaredTimer >= pacManToGhostDistance else -1
        score += pacManToGhostDistance * multiplier

    return score


# Abbreviation
better = betterEvaluationFunction

