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
    return  [s, s, w, s, w, w, s, w]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def simpleSearch(problem, tuples):

    # Initialize a list of visited positions
    visited_positions_list = []

    # Get the start position and initialize the list of actions to take to reach the Goal
    start_tuple = (problem.getStartState(), [])

    # Push the start_tuple in the stack
    tuples.push(start_tuple)

    # DFS/BFS stops when the stack becomes empty
    while not tuples.isEmpty():
        # Pop a tuple. If the position is the Goal state, end. If it's visited, skip. Else visit it.
        popped_tuple = tuples.pop()
        position = popped_tuple[0]
        action_list = popped_tuple[1]
        if problem.isGoalState(position):
            return action_list
        if position in visited_positions_list:
            continue
        visited_positions_list.append(position)

        # Iterate through all the successors of the popped position
        for next_tuple in problem.getSuccessors(position):
            next_position = next_tuple[0]
            # Visited positions get be skipped
            if next_position in visited_positions_list:
                continue
            next_action = next_tuple[1]

            # The next position and an updated action list can be pushed on the stack
            tuples.push((next_position, action_list + [next_action]))
    # Goal not found


def prioritySearch(problem, heuristic=nullHeuristic):

    # Initialize a list of visited positions
    visited_positions_list = []

    # Initialize a priority queue. The priority will be based on the cost + heuristic
    queue = util.PriorityQueue()

    # Get the start position and initialize the list of actions to take to reach the Goal
    start_tuple = (problem.getStartState(), [], 0)

    # Push the start_tuple in the queue. Priority = 0 + heuristic
    queue.push(start_tuple, heuristic(problem.getStartState(), problem))

    # UCS/A* stops when the priority queue becomes empty
    while not queue.isEmpty():
        # Pop a tuple. If the position is the Goal state, end. If it's visited, skip. Else visit it.
        popped_tuple = queue.pop()
        position = popped_tuple[0]
        action_list = popped_tuple[1]
        cost = popped_tuple[2]
        if problem.isGoalState(position):
            return action_list
        if position in visited_positions_list:
            continue
        visited_positions_list.append(position)

        # Iterate through all the successors of the popped position
        for next_tuple in problem.getSuccessors(position):
            next_position = next_tuple[0]
            next_action = next_tuple[1]
            next_cost = next_tuple[2]

            # Visited positions get be skipped
            if next_position in visited_positions_list:
                continue

            # The next position and an updated action list can be pushed on the stack
            # priority = cost + next_cost + heuristic
            queue.push((next_position, action_list + [next_action], cost + next_cost),
                       cost + next_cost + heuristic(next_position, problem))
    # Goal not found


def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first."""
    return simpleSearch(problem, util.Stack())


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return simpleSearch(problem, util.Queue())


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return prioritySearch(problem)


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node of least total cost plus heuristic first."""
    return prioritySearch(problem, heuristic)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
