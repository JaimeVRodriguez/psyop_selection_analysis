import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, auc
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import Normalizer

seed = 16486415

def regression_model(X_train, X_test, y_train, y_test, title, subtitle=None):
    '''The function regression_model is a helper function to train 
    a logistic regression model and visualize its performance through 
    a Receiver Operating Characteristic (ROC) curve and performance 
    metrics. It takes in the following parameters:

    X_train: The training data set for the independent variables.
    X_test: The testing data set for the independent variables.
    y_train: The training data set for the dependent variable.
    y_test: The testing data set for the dependent variable.
    title: The title of the plot.
    subtitle (optional): The subtitle of the plot.
    
    The function trains a logistic regression model using the training 
    data set, and calculates the predicted probabilities for the testing 
    data set. Then, the mean of the predicted probabilities is set as the 
    threshold value. Based on the threshold value, binary predictions are 
    made for the testing data set.

    The ROC curve is then plotted using the roc_curve function from the 
    scikit-learn library and the Area Under the Curve (AUC) score is calculated 
    using the auc function. The ROC curve and the AUC score are annotated on 
    the plot.

    Finally, performance metrics such as accuracy, precision, and recall are 
    calculated and annotated on the plot.'''
    model = LogisticRegression(random_state=seed).fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)
    threshold = y_probs[:,1].mean()
    y_pred = np.where(y_probs[:,1] >= threshold, 1, 0)
    y_hat = model.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_hat)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'Score: {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(title, loc='left', size=18, fontweight='bold')
    plt.legend()

    accuracy = accuracy_score(y_test, y_pred)*100
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    plt.annotate(f'Test accuracy: {accuracy:.2f}%', (0.65, 0.15), xycoords='axes fraction')
    plt.annotate(f'Test precision: {precision:.2f}', (0.65, 0.10), xycoords='axes fraction')
    plt.annotate(f'Test recall: {recall:.2f}', (0.65, 0.05), xycoords='axes fraction')

def multiple_regression_model(X, y, n_splits, title):
    '''    The function multiple_regression_model is similar to 
    the regression_model function but performs k-fold cross-validation 
    on the training data set. The function takes in the following parameters:

    X: The data set for the independent variables.
    y: The data set for the dependent variable.
    n_splits: The number of folds for the cross-validation.
    title: The title of the plot.
    The function first normalizes the input data and splits the data 
    into training and testing sets. Then, k-fold cross-validation is 
    performed on the training data set, where in each iteration the data 
    is further split into training and testing sets. A logistic regression 
    model is trained on the training data set and performance metrics and 
    the ROC curve are calculated for the testing data set. This process is 
    repeated for n_splits number of folds.

    Finally, the average accuracy, precision, and recall over all the folds 
    are calculated and annotated on the plot, along with the fold number with 
    the maximum accuracy.'''
    X = X.values
    y = y.values
    X = Normalizer().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    accuracy, precision, recall = [], [], []
    kf = KFold(n_splits=n_splits)
    counter = 1

    for train_index, test_index in kf.split(X_train):
        X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
        y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]
        model = LogisticRegression(random_state=seed).fit(X_fold_train, y_fold_train)
        
        y_probs = model.predict_proba(X_fold_test)
        threshold = y_probs[:,1].mean()
        y_pred = np.where(y_probs[:,1] >= threshold, 1, 0)
        y_hat = model.decision_function(X_fold_test)

        fpr, tpr, thresholds = roc_curve(y_fold_test, y_hat)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='Fold' + f' {counter} ' + 'Score: %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.title(title, loc='left', size=18, fontweight='bold')
        plt.legend()
        
        accuracy.append(accuracy_score(y_fold_test, y_pred))
        precision.append(precision_score(y_fold_test, y_pred))
        recall.append(recall_score(y_fold_test, y_pred))
        
        counter += 1

    avg_accuracy = np.max(accuracy)*100
    avg_precision = np.max(precision)
    avg_recall = np.max(recall)
    fold = accuracy.index(np.max(accuracy))+1

    plt.title(title, loc='left', size=18, fontweight='bold')

    plt.annotate(f'Fold: {fold}', (0.65, 0.20), xycoords='axes fraction')
    plt.annotate(f'Test accuracy: {avg_accuracy:.2f}%', (0.65, 0.15), xycoords='axes fraction')
    plt.annotate(f'Test precision: {avg_precision:.2f}', (0.65, 0.10), xycoords='axes fraction')
    plt.annotate(f'Test recall: {avg_recall:.2f}', (0.65, 0.05), xycoords='axes fraction')