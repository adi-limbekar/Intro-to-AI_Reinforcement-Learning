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


import collections
from hashlib import new
from time import sleep

import mdp
import util
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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            newValues = util.Counter()
            
            for currentState in self.mdp.getStates():
                if self.mdp.isTerminal(currentState): 
                    continue
                else:
                    newAction = self.getAction(currentState)   
                    #print("newAction : ", newAction) 
                    newValues[currentState] = self.computeQValueFromValues(currentState, newAction)

            self.values = newValues.copy()  

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

        Q_value = 0

        for nextState, transitionValue in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            nextStateValue = self.discount * self.values[nextState]
            Q_value += transitionValue * (reward + nextStateValue)

        return Q_value    

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        Q_value = util.Counter()

        if len(self.mdp.getPossibleActions(state)) == 0:
            return None

        for action in self.mdp.getPossibleActions(state):
            Q_value[action] = self.computeQValueFromValues(state,action)
        bestActionValue = Q_value.argMax()

        return bestActionValue    

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for state in states:
            self.values[state] = 0

        for i in range(self.iterations):
            newState = states[i % len(states)]
            #print("newState = ",newState)

            if self.mdp.isTerminal(newState):
                continue
            else:
                nextAction = self.getAction(newState)
                #print("newAction = ", nextAction)
                Q_value = self.getQValue(newState, nextAction)
                #print("Q_values = ", Q_value)
                self.values[newState] = Q_value


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
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

    def getPredecessors(self):
        predecessors = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextstate, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                        if nextstate in predecessors:
                            predecessors[nextstate].add(state)
                        else:
                            predecessors[nextstate] = {state}
        return predecessors    
    

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        predecessors = self.getPredecessors()
        priority_Queue= util.PriorityQueue()
        # diff = 0
        # Qlist = []
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    Qvalue = [(self.getQValue(state,action))]
                    #Qlist.append(Qvalue)
                    #print("Qvalue : ", Qvalue)
                max_Qvalue = max(Qvalue)    
                #print("max Qvalue : ", max_Qvalue)
                #print("value : ", self.values[state])
                diff = abs(self.values[state] - max_Qvalue)
                #print("diff1 : ", diff)
                #print("diff1 neg : ", -diff)
                priority_Queue.push(state, -diff)
                #print("priority queue : ", priority_Queue)

        for i in range(self.iterations):
            if priority_Queue.isEmpty():
                break
            newState = priority_Queue.pop()

            if not self.mdp.isTerminal(newState):
                max_Qvalue = max([self.getQValue(newState, action) for action in self.mdp.getPossibleActions(newState)])
                #max_Qvalue = max(Qvalue) 
                #print("max Qvalue2 : ", max_Qvalue)
                #print("value2 : ", self.values[newState]) 
                self.values[newState] = max_Qvalue
                #print("self value update : ",self.values[newState])

            for p in predecessors[newState]:
                if not self.mdp.isTerminal(p):
                    # for action in self.mdp.getPossibleActions(p):
                    #     Qvalue = [(self.getQValue(p,action))]
                    #     print("Qvalue3 : ", Qvalue)
                    # max_Qvalue = max(Qvalue)
                    max_Qvalue = max([self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)])
                    #print("max Qvalue3 : ", max_Qvalue)
                    #print("value3 : ", self.values[p]) 
                    diff = abs(self.values[p] - max_Qvalue)
                    #print("diff3 : ", diff)
                    #print("diff3 neg : ", -diff)
                    if (diff > self.theta):
                        priority_Queue.update(p, -diff)



       

