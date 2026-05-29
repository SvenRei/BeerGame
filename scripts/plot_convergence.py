import numpy as np
from eval.plotting import plot_convergence

history = np.loadtxt("training_history.txt")
plot_convergence(history)