import numpy as np

from utils.plot_utils import uncertainty_density_plot

y = np.load('result.npz')['y']
y_pred = np.load('result.npz')['y_pred']
y_var = np.load('result.npz')['y_var']

uncertainty_density_plot(y, y_pred, y_var, 'error_vs_correct_v2')

print()
