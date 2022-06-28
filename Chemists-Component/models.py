import gpflow
from kernels import Tanimoto


def Tanimoto_model(X,Y):
  '''
    Gaussian process regression model with Tanimoto kernel
    X: (n_samples, n_features)
    Y: (n_samples,) , binary classification
  '''
  m = gpflow.models.GPR((X, Y), kernel=Tanimoto(), mean_function=None,  noise_variance=1)

  opt = gpflow.optimizers.Scipy()
  opt.minimize(m.training_loss, variables=m.trainable_variables)

  return m