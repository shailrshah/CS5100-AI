# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.
# return format is (answerDiscount, answerNoise, answerLivingReward)

# BridgeGrid
# Can't afford to make a mistake crossing the bridge
def question2():
    return 0.9, 0


# Prefer the close exit (+1), risking the cliff (-10)
# Assign a very small discount so that subsequent value is very small => short term first
def question3a():
    return 0.0000001, 0, 0


# Prefer the close exit (+1), but avoiding the cliff (-10)
# Negative living reward
def question3b():
    return 0.1, 0.1, -0.9


# Prefer the distant exit (+10), risking the cliff (-10)
# discount should be just less than 1 to prefer long term reward
def question3c():
    return 0.999, 0, 0


# Prefer the distant exit (+10), avoiding the cliff (-10)
# Add some noise
def question3d():
    return 0.999, 0.5, 0


# Avoid both exits and the cliff
# Assign a +ve living reward
def question3e():
    return 0, 0, 0.1


def question6():
    return 'NOT POSSIBLE'
