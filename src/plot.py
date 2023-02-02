import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import functions

def select_bar_plot(data, labels, title, ylabel):
    fig, ax = plt.subplots()
    ax.bar(labels, data)
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_ylabel(ylabel)
    for i, v in enumerate(data):
        ax.text(i, v + 35, str(v), ha='center')

def annual_line_plot(df, title, xlabel, ylabel):
    ser = functions.annual_total(df)
    fig, ax = plt.subplots()
    ax.plot(ser.index, ser.values, marker='o')
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def age_bar(ser1, ser2, x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.bar(ser2[x], ser2[y], alpha=.5, label='SFAS')
    ax.bar(ser1[x], ser1[y], label='POAS')
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axvline(ser2[x].mean(), color='black', label=f'SFAS average Age: {ser2[x].mean()}')
    ax.axvline(ser1[x].mean(), color='black', linestyle='--', label=f'POAS average Age: {ser1[x].mean()}')
    ax.legend()

def violin_plot(x, y, data, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    sns.violinplot(x=x, y=y, data=data, split=True, showmedians=True, ax=ax)
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def select_correlation(df, title, xlabel, ylabel):
    corr_matrix = df.corr().loc[:, ['CODE']]
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, ax=ax)
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def distributions(df1, df2):
    features = ['DEP', 'AGE', 'PT', 'LANG', 'GT', 'CO']
    fig, ax = plt.subplots(2, 3)
    for i, feature in enumerate(features):
        row = i//3
        col = i % 3
        ax[row][col].hist(df1[feature], color='orange', alpha=1, label='Selected')
        ax[row][col].hist(df2[feature], color='b', alpha=.7, label='Not Selected')
        ax[row][col].set_title(feature)
    fig.tight_layout()
    
