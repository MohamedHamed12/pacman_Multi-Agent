# multiAgents.py
# --------------


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"



        foodDistances = [manhattanDistance(successorGameState.getPacmanPosition(), food) for food in currentGameState.getFood().asList()]
        minFoodDistance = min(foodDistances)

        ghostPositions = currentGameState.getGhostPositions()
        if any(manhattanDistance(successorGameState.getPacmanPosition(), ghost) < 2 for ghost in ghostPositions):
            return -float('inf')  # Assign a low score to actions that lead to ghosts


        totalScore =  -minFoodDistance

        return totalScore
        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return betterEvaluationFunction(currentGameState)


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

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # util.raiseNotDefined()
        # return self.value(gameState, 0, 0)
        return self.maxValue(gameState, 0, 0)[1]

    def is_terminal_state(self, gameState, depth, agent_idx):
        """
        Function to determine if we have reached a leaf node in the state search tree
        """
        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def value(self, gameState: GameState, agentIndex: int, depth: int):
        if self.is_terminal_state(gameState, depth, agentIndex):
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)[0]
        else:
            return self.minValue(gameState, agentIndex, depth)[0]

    def maxValue(self, gameState: GameState, agentIndex: int, depth: int):
        v = (float('-inf'), None)
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            val = self.value(successor, nextAgent, depth + 1)
            v = max(v, (val, action), key=lambda x: x[0])
        return v

    def minValue(self, gameState: GameState, agentIndex: int, depth: int):
        v =( float('inf'), None)
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            val = self.value(successor, nextAgent, depth + 1)
            v = min(v, (val, action), key=lambda x: x[0])
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 0, 0, float('-inf'), float('inf'))[1]
    
    def is_terminal_state(self, gameState, depth, agent_idx):
        """
        Function to determine if we have reached a leaf node in the state search tree
        """

        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        
        # elif gameState.getLegalActions(agent_idx) is 0:
        #     return gameState.getLegalActions(agent_idx)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth
      
    def value(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        if self.is_terminal_state(gameState, depth, agentIndex):
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)[0]
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)[0]

    
    def maxValue(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        v = (float('-inf'), None)
        for action in gameState.getLegalActions(agentIndex):
            v = max(v, (self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), depth+1, alpha, beta), action), key=lambda x: x[0])
            if v[0] > beta:
                return v
            alpha = max(alpha, v[0])
        return v
    
    def minValue(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        v =( float('inf'), None)
        for action in gameState.getLegalActions(agentIndex):
            v = min(v, (self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), depth + 1, alpha, beta), action), key=lambda x: x[0])
            if v[0] < alpha:
                return v
            beta = min(beta, v[0])
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 0, 0)[1]
        util.raiseNotDefined()
    def is_terminal_state(self, gameState, depth, agent_idx):
        """
        Function to determine if we have reached a leaf node in the state search tree
        """
        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        
        # elif gameState.getLegalActions(agent_idx) is 0:
        #     return gameState.getLegalActions(agent_idx)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def maxValue(self, gameState: GameState, agentIndex: int, depth: int):
        v = (float('-inf'), None)
        for action in gameState.getLegalActions(agentIndex):
            v = max(v, (self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), depth+1), action), key=lambda x: x[0])
        return v

    def expected_value (self, gameState: GameState, agentIndex: int, depth: int):
        v = 0
        for action in gameState.getLegalActions(agentIndex):
            v += self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), depth+1) / len(gameState.getLegalActions(agentIndex))
        return [v]

    def value(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        if self.is_terminal_state(gameState, depth, agentIndex):
           

            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)[0]
        else:
            return self.expected_value(gameState, agentIndex, depth, alpha, beta)[0]

      
    def value(self, gameState: GameState, agentIndex: int, depth: int):
        if self.is_terminal_state(gameState, depth, agentIndex):
           

            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)[0]
        else:
            return self.expected_value(gameState, agentIndex, depth)[0]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: the key features of the evaluation is that we need to
    eat as many food dots as possible, as well as eating capsules, the pacman
    also need to stay away from ghosts. Each of these features have weights, they
    don't contribute the same way, as eating more food has more priority than stay
    very far from ghosts, we can be near and still eating food.
    """

    # Useful information you can extract from a GameState (pacman.py)
    pacmanPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    ghostsPos = currentGameState.getGhostPositions()

    nearestGhost = float('inf')
    for x in ghostsPos:
        nearestGhost = min(nearestGhost, manhattanDistance(pacmanPos, x))

    nearestFood = float('inf')
    for x in foodPos:
        nearestFood = min(nearestFood, manhattanDistance(pacmanPos, x))

    if (nearestGhost < 2):
        # not a good state, we need to avoid near ghosts
        return -1e18

    if currentGameState.isLose():
        return -1e18

    if currentGameState.isWin():
        return 1e18

    foodLeft = currentGameState.getNumFood()
    capsLeft = len(currentGameState.getCapsules())

    foodLeftMultiplier = 100000
    capsLeftMultiplier = 1000
    foodDistMultiplier = 10
    score = 1.0/(foodLeft + 1) * foodLeftMultiplier - nearestGhost * 10 + \
           1.0/(nearestFood + 1) * foodDistMultiplier + \
           1.0/(capsLeft + 1) * capsLeftMultiplier
    # print(f"{foodLeft=}")
    # print(f"{score=}")
    return score


# Abbreviation
better = betterEvaluationFunction
