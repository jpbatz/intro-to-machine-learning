# polynomial linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# visualizing the regression results
def plot(regressor,
         X, y,
         y_pred=None,
         transformer=None,
         minimum=None, maximum=None, interval=None,
         title=None,
         xlabel=None,
         ylabel=None,
         ):
    plt.scatter(X, y, color='red')
    if y_pred is not None:
        X_grid = X
    else:
        if not interval:
            interval = 1.
        if not minimum:
            minimum = min(
                X.values if isinstance(X, pd.core.frame.DataFrame) else X
                )
        if not maximum:
            maximum = max(
                X.values if isinstance(X, pd.core.frame.DataFrame) else X
                )
        X_grid = np.arange(minimum, maximum+interval, interval)
        X_grid = X_grid.reshape(len(X_grid), 1)
        if transformer:
            y_pred = regressor.predict(transformer(X_grid))
        else:
            y_pred = regressor.predict(X_grid)
    plt.plot(X_grid, y_pred, color='blue')
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()
