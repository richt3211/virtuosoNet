import matplotlib.pyplot as plt
import numpy as np

def plot_loss(train_loss, valid_loss, fig_path):
  X = [i for i in range(train_loss)]
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.bar(X, train_loss, color='b', width=0.25)
  ax.bar(X, valid_loss, color='b', width=0.25)

  plt.savefig(fig_path)
  plt.show()