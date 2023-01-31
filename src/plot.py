import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def select_bar_plot(data, labels, title, ylabel):
    fig, ax = plt.subplots()
    ax.bar(labels, data)
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_ylabel(ylabel)
    for i, v in enumerate(data):
        ax.text(i, v + 35, str(v), ha='center')

def age_bar_plot(x, y, title, xlabel, ylabel, mean, high):
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axvline(mean, color='black', linestyle='--', linewidth=1, label=f'Average Age: {mean}')
    ax.axvline(high, color='grey', linestyle='--', linewidth=1, label=f'Most Selected Age: {high}')
    ax.legend()

def violin_plot(x, y, data, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    sns.violinplot(x=x, y=y, data=data, split=True, showmedians=True, ax=ax)
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
