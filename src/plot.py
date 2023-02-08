import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ptitprince as pt

import functions

def select_bar_plot(data, labels, title, ylabel):
    '''
    Parameters:
    `data`: A list of numeric values to be plotted as bar heights.
    `labels`: A list of strings to be used as x-axis labels.
    `title`: A string to be used as the plot's title.
    `ylabel`: A string to be used as the y-axis label.
        
    The function creates a bar plot by calling plt.subplots() to 
    create a figure and an axis object, then calling ax.bar(labels, data) 
    to plot the bars. The function sets the title of the plot with 
    ax.set_title(title, loc='left', size=18, fontweight='bold') and 
    sets the y-axis label with ax.set_ylabel(ylabel). Finally, the 
    function adds the value of each bar to the plot using 
    ax.text(i, v + 35, str(v), ha='center').'''
    fig, ax = plt.subplots()
    ax.bar(labels, data)
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_ylabel(ylabel)
    for i, v in enumerate(data):
        ax.text(i, v + 35, str(v), ha='center')

def annual_line_plot(df, title, xlabel, ylabel):
    '''
    Parameters:
    `df`: A DataFrame object containing data to be plotted.
    `title`: A string to be used as the plot's title.
    `xlabel`: A string to be used as the x-axis label.
    `ylabel`: A string to be used as the y-axis label.

    The function first calls functions.annual_total(df) to calculate the 
    annual total from the input DataFrame and stores it in the ser object. 
    The function then creates a line plot by calling plt.subplots() to create 
    a figure and an axis object, and calling ax.plot(ser.index, ser.values, 
    marker='o') to plot the line. The function sets the title of the plot with 
    ax.set_title(title, loc='left', size=18, fontweight='bold') and sets the 
    x-axis and y-axis labels with ax.set_xlabel(xlabel) and ax.set_ylabel(ylabel), 
    respectively.'''
    ser = functions.annual_total(df)
    fig, ax = plt.subplots()
    ax.plot(ser.index, ser.values, marker='o')
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def age_bar(ser1, ser2, x, y, title, xlabel, ylabel):
    '''
    Parameters:
    `ser1`: A pandas Series object containing data for one set of bars.
    `ser2`: A pandas Series object containing data for another set of bars.
    `x`: A string that specifies the column to be used as the x-axis values.
    `y`: A string that specifies the column to be used as the y-axis values.
    `title`: A string to be used as the plot's title.
    `xlabel`: A string to be used as the x-axis label.
    `ylabel`: A string to be used as the y-axis label.

    The function creates a bar plot by calling plt.subplots() to create a figure and 
    an axis object, and then calling ax.bar(ser1[x], ser1[y]) and ax.bar(ser2[x], ser2[y], 
    alpha=.5, label='SFAS') to plot the two sets of bars. The function sets the title of 
    the plot with ax.set_title(title, loc='left', size=18, fontweight='bold') and sets 
    the x-axis and y-axis labels with ax.set_xlabel(xlabel) and ax.set_ylabel(ylabel), 
    respectively. The function also adds vertical lines to indicate the average age of 
    each set of bars with ax.axvline(ser1[x].mean(), color='black', linestyle='--', 
    label=f'POAS average Age: {ser1[x].mean()}') and ax.axvline(ser2[x].mean(), color='black', 
    label=f'SFAS average Age: {ser2[x].mean()}'). The function adds a legend with ax.legend().'''
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
    '''
    Parameters:
    `x`: A string that specifies the column to be used as the x-axis values.
    `y`: A string that specifies the column to be used as the y-axis values.
    `data`: A pandas DataFrame containing the data to be plotted.
    `title`: A string to be used as the plot's title.
    `xlabel`: A string to be used as the x-axis label.
    `ylabel`: A string to be used as the y-axis label.

    The function creates a violin plot by calling plt.subplots() to create a figure and an axis 
    object, and then calling sns.violinplot(x=x, y=y, data=data, split=True, showmedians=True, ax=ax) 
    to plot the violin plot. The function sets the title of the plot with ax.set_title(title, 
    loc='left', size=18, fontweight='bold') and sets the x-axis and y-axis labels with 
    ax.set_xlabel(xlabel) and ax.set_ylabel(ylabel), respectively.'''
    fig, ax = plt.subplots()
    sns.violinplot(x=x, y=y, data=data, split=True, showmedians=True, ax=ax)
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def raincloud_plot(x, y, data, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax = pt.RainCloud(data=data, x=x, y=y, orient='h', bw=.1, ax=ax)
    sns.despine()

    means = data.groupby(x)[y].mean()
    maxs = data.groupby(x)[y].max()

    ax.plot(means, means.index, '^', label='Median', color='b')
    ax.plot(maxs, maxs.index, 'o', label='Max', color='r')
    ax.legend()
    
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def raincloud_triple(x, y1, y2, y3, data, data2):
    fig, ax = plt.subplots(3,1)
    ax[0] = pt.RainCloud(data=data2, x=x, y=y1, orient='h', bw=.1, ax=ax[0])
    ax[1] = pt.RainCloud(data=data, x=x, y=y2, orient='h', bw=.1, ax=ax[1])
    ax[2] = pt.RainCloud(data=data, x=x, y=y3, orient='h', bw=.1, ax=ax[2])
    sns.despine()



    

def select_correlation(df, title, xlabel, ylabel):
    '''
    Parameters:
    `df`: The input DataFrame containing the data to be plotted.
    `title`: The title of the heatmap plot.
    `xlabel`: The label of the x-axis.
    `ylabel`: The label of the y-axis.

    This function creates a heatmap of the correlation matrix between the SELECT column 
    and all other columns in the input DataFrame 'df'.
    '''
    corr_matrix = df.corr().loc[:, ['SELECT']]
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, ax=ax)
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def distributions(df1, df2):
    '''
    Parameters:
    df1: The first dataframe to plot the histograms of the features.
    df2: The second dataframe to plot the histograms of the features.

    Plots the histograms of the features for two input dataframes.
    The features being compared are ['DEP', 'AGE', 'PT', 'LANG', 'GT', 'CO'].
    '''
    features = ['DEP', 'AGE', 'PT', 'LANG', 'GT', 'CO']
    fig, ax = plt.subplots(2, 3)
    for i, feature in enumerate(features):
        row = i//3
        col = i % 3
        ax[row][col].hist(df1[feature], color='orange', alpha=1, label='Selected')
        ax[row][col].hist(df2[feature], color='b', alpha=.7, label='Not Selected')
        ax[row][col].set_title(feature)
    fig.tight_layout()
    
