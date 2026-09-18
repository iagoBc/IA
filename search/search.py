# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Pesquise os nós mais profundos na árvore de busca primeiro.

    Seu algoritmo de busca precisa retornar uma lista de ações que alcançam o
    objetivo. Certifique-se de implementar um algoritmo de busca em grafo.

    Para começar, você pode querer tentar alguns desses comandos simples para
    entender o problema de busca que está sendo passado:

    print("Início:", problem.getStartState())
    print("O início é um objetivo?", problem.isGoalState(problem.getStartState()))
    print("Sucessores do início:", problem.getSuccessors(problem.getStartState()))
    """
    init_state = problem.getStartState()
    if problem.isGoalState(init_state):
        return []

    stack = util.Stack()
    visits = list()

    for succ, action, _ in problem.getSuccessors(
        init_state
    ):  # Adiciona os sucessores na pilha
        stack.push((succ, [action]))

    while not stack.isEmpty():
        state, actions = stack.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visits:
            visits.append(state)

        for succ, action, _ in problem.getSuccessors(state):
            if (
                succ not in visits
            ):  # Adiciona os sucessores que ainda nao foram expandidos na pilha
                stack.push((succ, actions + [action]))

    return []


def breadthFirstSearch(problem: SearchProblem):
    """Procura primeiro os nós mais rasos (de menor profundidade) na árvore de busca."""
    init_state = problem.getStartState()
    if problem.isGoalState(init_state):
        return []

    queue = util.Queue()
    visits = list()
    visits.append(
        init_state
    )  # Marca como visitado o estado inicial para nao expandir ele novamente

    for succ, action, _ in problem.getSuccessors(
        init_state
    ):  # Adiciona os sucessores na fila
        visits.append(succ)  # Marca como visitado para nao expandir ele mais de uma vez
        queue.push((succ, [action]))

    while not queue.isEmpty():
        state, actions = queue.pop()

        if problem.isGoalState(state):
            return actions

        for succ, action, _ in problem.getSuccessors(state):
            if succ not in visits:
                visits.append(succ)
                queue.push((succ, actions + [action]))

    return []


def uniformCostSearch(problem: SearchProblem):
    """Expande primeiro os nós com menor custo acumulado (busca de custo uniforme)."""

    init_state = problem.getStartState()
    if problem.isGoalState(init_state):
        return []

    queue = util.PriorityQueue()
    queue.push((init_state, [], 0), 0)  # Adiciona o estado inicial na fila
    best_cost = {init_state: 0}

    while not queue.isEmpty():
        state, actions, costs = queue.pop()

        if (
            costs > best_cost[state]
        ):  # Custo maior que o melhor custo ja encontrado, pula para o prox
            continue

        if problem.isGoalState(state):
            return actions

        for succ, action, step_cost in problem.getSuccessors(state):
            new_cost = costs + step_cost
            if succ not in best_cost or new_cost < best_cost[succ]:
                best_cost[succ] = new_cost
                queue.push((succ, actions + [action], new_cost), new_cost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Expande primeiro os nós com a menor soma entre custo acumulado e valor heurístico (A*)."""
    init_state = problem.getStartState()  # Estado inicial
    if problem.isGoalState(init_state):
        return []

    def priority(item):
        state, _, cost = item
        return cost + heuristic(state, problem)  # g(n) + h(n)

    queue = util.PriorityQueueWithFunction(priority)
    queue.push((init_state, [], 0))

    best_cost = {init_state: 0}

    while not queue.isEmpty():
        state, actions, costs = queue.pop()

        if (
            costs > best_cost[state]
        ):  # Custo maior que o melhor custo ja encontrado, pula para o prox
            continue

        if problem.isGoalState(state):
            return actions

        for succ, action, step_cost in problem.getSuccessors(state):
            new_cost = costs + step_cost
            if succ not in best_cost or new_cost < best_cost[succ]:
                best_cost[succ] = new_cost
                queue.push((succ, actions + [action], new_cost))

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
