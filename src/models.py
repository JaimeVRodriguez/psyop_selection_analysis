import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, auc
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