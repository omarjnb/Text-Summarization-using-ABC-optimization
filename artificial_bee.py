import numpy as np

from copy import deepcopy
from abc import ABCMeta
from six import add_metaclass

@add_metaclass(ABCMeta)
class ArtificialBee(object):
    
    TRIAL_INITIAL_DEFAULT_VALUE = 0
    INITIAL_DEFAULT_PROBABILITY = 0.0

    def __init__(self, obj_function):
        self.pos, self.indices = obj_function.sample()
        self.obj_function = obj_function
        self.minf = obj_function.minf
        self.maxf = obj_function.maxf
        self.fitness = obj_function.evaluate(self.pos)
        self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE
        self.prob = ArtificialBee.INITIAL_DEFAULT_PROBABILITY
    
    def update_bee(self, pos, fitness, indices):
        if fitness >= self.fitness:
            self.pos = pos
            self.fitness = fitness
            self.indices = indices
            self.trial
        else:
            self.trial += 1

    def reset_bee(self, max_trials):
        if self.trial >= max_trials:
            self.__reset_bee()

    def __reset_bee(self):
        self.pos = self.obj_function.sample()
        self.fitness = self.obj_function.evaluate(self.pos)
        self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE
        self.prob = ArtificialBee.INITIAL_DEFAULT_PROBABILITY