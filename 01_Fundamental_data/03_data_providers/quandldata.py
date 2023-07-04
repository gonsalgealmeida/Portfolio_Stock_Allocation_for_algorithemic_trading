import os
import quandl
import seaborn as sns
import matplotlib.pyplot as plt

def set_style(style='whitegrid'):
    """
    Set seaborn style for plots.
    """
    sns.set_style(style)

def get_quandl_data(api_key, dataset):
    """
    Fetch dataset from Quandl.
    """
    data = quandl.get(dataset, api_key=api_key)
    return data.squeeze()

def plot_data(data, title, lw=2, figsize=(12, 4)):
    """
    Plot data with title and line width.
    """
    data.plot(lw=lw, title=title, figsize=figsize)
    sns.despine()
    plt.tight_layout()
    plt.show()

