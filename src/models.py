import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, auc
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import Normalizer
seed = 16486415

def regression_model(X_train, X_test, y_train, y_test, title, subtitle=None):
    model = LogisticRegression(random_state=seed).fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)
    threshold = y_probs[:,1].mean()
    y_pred = np.where(y_probs[:,1] >= threshold, 1, 0)
    y_hat = model.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_hat)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'AUC Score: {roc_auc:.2f}')
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
        plt.plot(fpr, tpr, label='Fold' + f' {counter}: ' + 'Area = %0.2f' % roc_auc)
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