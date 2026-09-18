# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

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
        legalMoves = gameState.getLegalActions() # Pega a lista de acoes legais do Pacman (agentIndex = 0)

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves] # Avalia cada acao legal usando a funcao de avaliacao e armazena os resultados em uma lista de scores
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore] # Encontra os indices das acoes que possuem o melhor score
        chosenIndex = random.choice(bestIndices) # Escolhe aleatoriamente um dos indices das acoes com o melhor score

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex] # Retorna a acao escolhida (aquela com o melhor score)

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
        successorGameState = currentGameState.generatePacmanSuccessor(action) # Gera o estado sucessor do jogo após o Pacman realizar a ação especificada
        succPos = successorGameState.getPacmanPosition() # Obtém a nova posição do Pacman após a ação
        foodsList = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore() # Inicializa a pontuação com o score do estado sucessor

        if foodsList:
            minFoodDist = min(manhattanDistance(succPos, food) for food in foodsList) # Calcula a menor distância de Manhattan entre a nova posição do Pacman e cada comida restante
            score += 5 / minFoodDist # Aumenta a pontuação com base na menor distância para a comida mais próxima (quanto menor a distância, maior o aumento na pontuação)

        for ghostState in newGhostStates: # Itera sobre os estados dos fantasmas no estado sucessor
            ghostDist = manhattanDistance(succPos, ghostState.getPosition()) # Calcula a distância de Manhattan entre a nova posição do Pacman e a posição do fantasma
            if ghostState.scaredTimer > 0: # Se o fantasma estiver assustado, aumenta a pontuação com base na distância para o fantasma (quanto mais próximo, maior o aumento na pontuação)
                if ghostDist > 0:
                    score += 2.0 / ghostDist
            else:
                if ghostDist < 2: # Se o fantasma não estiver assustado e estiver muito próximo do Pacman, penaliza a pontuação
                    score -= 500
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

        numAgents = gameState.getNumAgents()
        bestAction = None
        bestValue = float("-inf")

        for action in gameState.getLegalActions(0): # Itera por cada ação legal do Pacman (agentIndex = 0).
            successor = gameState.generateSuccessor(0, action) # Simula o estado sucessor se o Pacman fizer essa ação.
            v = self.minimax(successor, 0, 1 % numAgents, numAgents) # Chama a função minimax para calcular o valor do estado sucessor, passando a profundidade atual (0), o índice do próximo agente (1 % numAgents)
            if bestAction is None or v > bestValue: 
                bestValue = v
                bestAction = action

        return bestAction

    def minimax(self, state: GameState, depth: int, agentIndex: int, numAgents: int):
        if depth == self.depth or state.isWin() or state.isLose(): # Se a profundidade máxima for atingida 
            return self.evaluationFunction(state) # Retorna o valor da função de avaliação para o estado atual

        legalMoves = state.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(state)

        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth + 1 if nextAgent == 0 else depth # Incrementa a profundidade apenas quando o próximo agente for o Pacman
    
        if agentIndex == 0:
            v = float("-inf")
            for action in legalMoves: # 
                successor = state.generateSuccessor(agentIndex, action) 
                v = max(v, self.minimax(successor, nextDepth, nextAgent, numAgents)) # Atualiza o valor máximo
            return v

        v = float("inf")
        for action in legalMoves:
            successor = state.generateSuccessor(agentIndex, action)
            v = min(v, self.minimax(successor, nextDepth, nextAgent, numAgents)) # Atualiza o valor mínimo

        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents() 
        bestAction = None
        bestValue = float("-inf")
        alpha = float("-inf") # Valor mínimo de corte
        beta = float("inf") # Valor máximo de corte
        
        for action in gameState.getLegalActions(0): # Itera por cada ação legal do Pacman
            successor = gameState.generateSuccessor(0, action)
            v = self.alpha_beta(successor, alpha, beta, 0, 1 % numAgents, numAgents) # Calcula o valor da ação
            if bestAction is None or v > bestValue:
                bestValue = v
                bestAction = action
            alpha = max(alpha, bestValue) # Atualiza o valor de alpha com o melhor valor encontrado até agora

        return bestAction

    def alpha_beta(self, state: GameState, alpha: float, beta: float, depth: int, agentIndex: int, numAgents: int):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalMoves = state.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(state)

        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth + 1 if nextAgent == 0 else depth # Incrementa a profundidade apenas quando o próximo agente for o Pacman

        if agentIndex == 0:
            v = float("-inf")
            for action in legalMoves:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, self.alpha_beta(successor, alpha, beta, nextDepth, nextAgent, numAgents)) # Atualiza o valor máximo
                if v > beta: # Se o valor atual for maior que beta, significa que o nó pai (minimizador) não escolherá esse caminho, então podemos podar a árvore e retornar o valor atual 
                    return v
                alpha = max(alpha, v) # Atualiza o valor de alpha com o melhor valor encontrado até agora
            return v

        v = float("inf")
        for action in legalMoves:
            successor = state.generateSuccessor(agentIndex, action)
            v = min(v, self.alpha_beta(successor, alpha, beta, nextDepth, nextAgent, numAgents))
            if v < alpha: # Se o valor atual for menor que alpha, significa que o nó pai (maximizador) não escolherá esse caminho, então podemos podar a árvore e retornar o valor atual
                return v
            beta = min(beta, v) # Atualiza o valor de beta com o melhor valor encontrado até agora
        return v
        

def probability(numActions):
    if numActions <= 0:
        return 0.0
    return 1.0 / numActions # Retorna a probabilidade uniforme para cada ação legal em um nó de chance


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
        numAgents = gameState.getNumAgents()
        bestAction = None
        bestValue = float("-inf")

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = self.expectimax(successor, 0, 1 % numAgents, numAgents) # Calcula o valor expectativa para cada ação legal
            if bestAction is None or value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

    def expectimax(self, state: GameState, depth: int, agentIndex: int, numAgents: int):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalMoves = state.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(state)

        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth + 1 if nextAgent == 0 else depth # Incrementa a profundidade apenas quando o próximo agente for o Pacman

        if agentIndex == 0:
            value = float("-inf")
            for action in legalMoves:
                successor = state.generateSuccessor(agentIndex, action)
                value = max(value, self.expectimax(successor, nextDepth, nextAgent, numAgents)) # Atualiza o valor máximo
            return value

        value = 0.0
        chance = probability(len(legalMoves))
        for action in legalMoves:
            successor = state.generateSuccessor(agentIndex, action)
            value += chance * self.expectimax(successor, nextDepth, nextAgent, numAgents) # Calcula o valor esperado para os nós de chance (fantasmas), multiplicando a probabilidade uniforme pela avaliação do estado sucessor
        return value
        

def betterEvaluationFunction(currentGameState: GameState):
    """Calcula a distancia de Manhattan entre a posição do Pacman e cada comida restante, cápsula e fantasma, e ajusta a pontuação com base nessas distâncias. Quanto mais próximo o Pacman estiver da comida ou cápsula, maior será o aumento na pontuação. Se o Pacman estiver muito próximo de um fantasma não assustado, a pontuação será penalizada para incentivar evitar colisões."""

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    if food:
        minFoodDist = min(manhattanDistance(pos, f) for f in food) # Calcula a menor distância de Manhattan entre a posição do Pacman e cada comida restante
        score += 10.0 / minFoodDist # Aumenta a pontuação com base na menor distância para a comida mais próxima (quanto menor a distância, maior o aumento na pontuação)

    score -= 10 * len(food)  # Penaliza ter muita comida sobrando (incentiva progresso) 

    if capsules:
        minCapsuleDist = min(manhattanDistance(pos, c) for c in capsules) # Calcula a menor distância de Manhattan entre a posição do Pacman e cada cápsula restante
        score += 8.0 / minCapsuleDist # Aumenta a pontuação com base na menor distância para a cápsula mais próxima (quanto menor a distância, maior o aumento na pontuação)

    score -= 40 * len(capsules)  # Penaliza ter muitas cápsulas sobrando (incentiva progresso)

    for ghost in ghostStates:
        dist = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:  # Se o fantasma estiver assustado
            if dist > 0:
                score += 50.0 / dist # Aumenta a pontuação com base na distância para o fantasma (quanto mais próximo, maior o aumento na pontuação)
        else:
            if dist < 2:
                score -= 70 # Penaliza estar muito próximo de um fantasma não assustado (incentiva evitar colisões)

    return score

# Abbreviation
better = betterEvaluationFunction
