from os import path

from matplotlib import pyplot as plt
from torchWork.plot_losses import PlotLosses, LossType
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME

try:
    from workspace import EXP_PATH
except ImportError:
    EXP_PATH = input('EXP_PATH=')

LOSS_TYPES_TO_PLOT = [
    LossType('train', 'loss_root.harmonics'), 
]

plotLosses = PlotLosses(
    path.join(EXP_PATH, EXPERIMENT_PY_FILENAME), 
    LOSS_TYPES_TO_PLOT, using_epoch_not_batch=True, 
    average_over=1, start=0, 
    which_legend=0, linewidth=.5, 
)
fig = next(plotLosses)

plt.savefig(path.join(EXP_PATH, 'auto_plot_loss.pdf'))
plt.show()

# # Plot one group at a time:
# for fig in plotLosses:
#     if LOSS_TYPES_TO_PLOT[-1].loss_name == 'linear_proj_mse':
#         fig.axes[-1].set_ylim(0, 1)
#     plt.show()
