from artificial_bee import ArtificialBee
import numpy as np

class OnLookerBee(ArtificialBee):

    def onlook(self, best_food_sources, max_trials):
        candidate = np.random.choice(best_food_sources)
        self.__exploit(candidate.pos, candidate.fitness, max_trials)
    
    def __exploit(self, candidate, fitness, max_trials):
        if(self.trial <= max_trials):
            ind = np.random.choice(self.indices, 1)
            self.pos[ind] = 0
            i = np.random.choice(np.arange(self.obj_function.dim), size=1)
            while(i in self.indices):
                i = np.random.choice(np.arange(self.obj_function.dim), size=1)
            n_pos = self.pos
            n_pos[i] = 1
            index = np.argwhere(self.indices==ind)
            y = np.delete(self.indices, index)
            y = np.append(y, i)
            n_indices = np.sort(y)
            n_fitness = self.obj_function.evaluate(n_pos)

            if n_fitness >= fitness:
                self.pos = n_pos
                self.fitness = n_fitness
                self.indices = n_indices
                self.trial = 0
            else:
                self.trial += 1