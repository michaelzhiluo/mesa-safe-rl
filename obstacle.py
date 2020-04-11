import numpy as np

class Obstacle:
    def __init__(self, boundsx, boundsy, penalty=100):
        self.boundsx = boundsx
        self.boundsy = boundsy
        self.penalty = 1


    def __call__(self, x):
        return (self.boundsx[0] <= x[0] <= self.boundsx[1] and self.boundsy[0] <= x[1] <= self.boundsy[1]) * self.penalty

class ComplexObstacle(Obstacle):

    def __init__(self, bounds):
        self.obs = []
        for boundsx, boundsy in bounds:
            self.obs.append(Obstacle(boundsx, boundsy))

    def __call__(self, x):
        return np.max([o(x) for o in self.obs])
