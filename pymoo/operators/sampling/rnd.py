import numpy as np

from pymoo.core.sampling import Sampling


def random(problem, n_samples=1):
    X = np.random.random((n_samples, problem.n_var))

    if problem.has_bounds():
        xl, xu = problem.bounds()
        assert np.all(xu >= xl)
        X = xl + (xu - xl) * X

    return X


class FloatRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.random((n_samples, problem.n_var))

        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert np.all(xu >= xl)
            X = xl + (xu - xl) * X

        return X


class CustomDroneFloatRandomSampling(FloatRandomSampling):
    """
    This class samples equal number of samples for each value of the first variable of the design space.
    It creates equal number of individuals having the same value of the first variable.
    The n_arms is the number of different values for the first variable. 
    The default values are 4, 5, 6, 7, 8. Therefore, there will be 5 groups of equal number of individuals. 
    Each grouup will have only one value of the first variable, which is 4, 5, 6, 7, or 8 arms.
    """
    def __init__(self, n_arms=None, **kwargs):
        super().__init__(**kwargs)
        if n_arms is None:
            n_arms = [4, 5, 6, 7, 8]
        self.n_groups = len(n_arms)
        self.n_arms = n_arms
    def _do(self, problem, n_samples, **kwargs):
        # Get the regular random sampling first
        X = super()._do(problem, n_samples, **kwargs)
        
        # For the first variable, create equal groups
        #n_groups = 4  # For example, if you want 4 different values
        samples_per_group = n_samples // self.n_groups
        
        # Create values for each group (for example, values 1 through 4)
        for i,val in enumerate(self.n_arms):
            start_idx = i * samples_per_group
            end_idx = (i + 1) * samples_per_group if i < self.n_groups - 1 else n_samples
            X[start_idx:end_idx, 0] = val  # Assign value to this group
            
        return X

class BinaryRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < 0.5).astype(bool)


class IntegerRandomSampling(FloatRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        n, (xl, xu) = problem.n_var, problem.bounds()
        return np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)])


class PermutationRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), 0, dtype=int)
        for i in range(n_samples):
            X[i, :] = np.random.permutation(problem.n_var)
        return X
