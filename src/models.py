import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, auc
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import Normalizer
seed = 8

def regression_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(random_state=seed).fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)
    threshold = y_probs[:,1].mean()
    y_pred = np.where(y_probs[:,1] >= threshold, 1, 0)
    y_hat = model.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_hat)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'ROC Curve {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()

    accuracy = accuracy_score(y_test, y_pred)*100
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f'Test accuracy: {accuracy:.2f}%')
    print(f'Test precision: {precision:.2f}')
    print(f'Test recall: {recall:.2f}')

def multiple_regression_model(X, y, n_splits):
    X = X.values
    y = y.values
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
        plt.legend()
        
        accuracy.append(accuracy_score(y_fold_test, y_pred))
        precision.append(precision_score(y_fold_test, y_pred))
        recall.append(recall_score(y_fold_test, y_pred))
        
        avg_accuracy = np.mean(accuracy)
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)

        counter += 1

    plt.title('ROC Curve', loc='left', size=18, fontweight='bold')
        
    print(f'Mean Accuracy: {(avg_accuracy * 100):.2f}%')
    print(f'Mean Precision: {avg_precision:.2f}')
    print(f'Mean Recall: {avg_recall:.2f}')