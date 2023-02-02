def import_all():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import scipy.stats as stats
    import statsmodels.api as sm

    from pandas.plotting import scatter_matrix
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
    from sklearn.datasets import make_classification
    from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, auc
    from sklearn.preprocessing import Normalizer
    from sklearn import metrics

    sns.set()