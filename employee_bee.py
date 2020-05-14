from artificial_bee import ArtificialBee
import numpy as np

class EmployeeBee(ArtificialBee):

    def explore(self, max_trials):
        '''generate a new solution by removing a sentence and adding a new sentence'''
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
            self.update_bee(n_pos, n_fitness, n_indices)
    
    def get_fitness(self):
        return np.abs(self.fitness)
    
    def compute_prob(self, max_fitness):
        self.prob = self.get_fitness() / max_fitness