# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        """
        Executa a iteração de valor para o número especificado de iterações.
        Atualiza os valores dos estados com base na equação de Bellman.
        """
        for _ in range(self.iterations):                   
            updated = util.Counter()                                # Criar um novo Counter para armazenar os valores atualizados dos estados

            for state in self.mdp.getStates():                      # Loop para cada estado do MDP
                actions = self.mdp.getPossibleActions(state)        # Obter as ações possíveis para o estado atual
                if not actions:                                     # Se não houver ações possíveis, continue para o próximo estado
                    continue

                # Calcular o valor máximo para o estado atual com base nas ações possíveis
                best = float("-inf")
                for action in actions:
                    value = self.computeQValueFromValues(state, action)
                    if value > best:
                        best = value
                updated[state] = best

            self.values = updated                             # Atualizar os valores dos estados com os valores atualizados


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
        q_value = 0                                                             # Inicializar o Q-valor para a ação no estado atual
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)       # Obter os estados de transição e suas probabilidades para a ação no estado atual
        for next, probability in transitions:                                   # Loop para cada estado de transição e sua probabilidade
            reward = self.mdp.getReward(state, action, next)                    # Obter a recompensa para a transição do estado atual para o próximo estado com a ação atual
            q_value += probability * (
                reward + self.discount * self.values[next]
            )                                                                   # Calcular o Q-valor acumulando a recompensa ponderada pela probabilidade e o valor descontado do próximo estado
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)                            # Obter as ações possíveis para o estado atual
        if not actions:                                                         # Se não houver ações possíveis, retornar None (estado terminal)
            return None

        best_action = actions[0]                                                 # Inicializar a melhor ação como a primeira ação possível
        best_value = self.computeQValueFromValues(state, best_action)            # Calcular o valor da melhor ação inicial usando a função computeQValueFromValues
        for action in actions[1:]:                                               # Loop para cada ação possível (exceto a primeira)
            value = self.computeQValueFromValues(state, action)        
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
