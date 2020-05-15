import numpy as np

from scipy import optimize
from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass

def cosine_similarity(x, y):
    '''
    Returns a value that represents the similarity between two sentences.
    The arguments are two vectors that represent tf.isf or tf.idf scores of two sentences.
    edit: the arguments are two maps that contaiin ts.idf scores of two sentences.
    '''
    # numerator = np.sum(np.multiply(x, y))
    # denominator = np.sqrt(np.multiply(np.sum(np.square(x)), np.sum(sp.square(y))))
    # return numerator/denominator

    # return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    nu=0
    de1=0
    de2=0
    # print(x, y)
    for a in x.keys():
        nu += (x[a]*y[a])
        de1 += pow(x[a],2)
        de2 += pow(y[a],2)
    de = de1*de2
    de = np.sqrt(de)
    return nu/de

@add_metaclass(ABCMeta)
class ObjectiveFunction(object):
    def __init__(self, dim, s):
        self.dim = dim  # number of sentences in the document
        self.minf = 0   
        self.maxf = 2   
        self.s = s      # number of sentences supposed to be in the summary

    def sample(self):
        '''returns a sample from the list of sentences. returns a decision vector'''
        # se = np.random.randint(low=self.minf, high=self.maxf, size=self.s)
        on = np.zeros(self.dim)
        indices = np.random.choice(np.arange(self.dim), replace=False, size=self.s)
        on[indices] = 1
        return on,indices

    @abstractmethod
    def evaluate(self, x):
        pass

class content_coverage(ObjectiveFunction):
    def __init__(self, dim, s, mean_vector, sentence_map):
        # self.dim = dim
        # self.minf = 0
        # self.maxf = 2
        super(content_coverage, self).__init__(dim, s)
        self.mean_vector = mean_vector
        self.sentence_map = sentence_map

    # def sample(self):
    #     '''returns a sample from the list of sentences. returns a decision vector'''
    #     return np.random.randint(low=self.minf, high=self.maxf, size=self.dim)

    def evaluate(self, pos):
        '''return the score of this solution = Θ-coverage(pos)'''
        # sentence map is a map of sentences in the sense that sentence indexes are the keys and the sentence weight dictionaries are the values.
        # pos is random sample of sentences
        su=0
        # print(self.sentence_map)
        # for x in sentence_map.keys():
        #     print(x, sentence_map[x])
        for i in range(len(pos)):
            if(pos[i] == 1):
                x = cosine_similarity(self.mean_vector, self.sentence_map[i])
                su += x
        return su
