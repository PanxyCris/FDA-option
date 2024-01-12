"""
Plot Functions that helps to plot different purposes
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, y_true, y_pred, classes):
    """
    Plot Confusion Matrix for machine learning result
    :param model: machine learning model
    :param y_true: Real Target Value
    :param y_pred: Prediction Target Value
    :param classes: Classes defined
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'{model.name} Confusion Matrix')
    plt.show()


def plot_roc_curve(model1, model2, y_test, scores_model1, scores_model2, threshold):
    """
    Plot ROC Curve for 2 models
    :param model1:
    :param model2:
    :param y_test:
    :param scores_model1:
    :param scores_model2:
    :param threshold:
    """
    fpr1, tpr1, _ = roc_curve(y_test, scores_model1)
    roc_auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, _ = roc_curve(y_test, scores_model2)
    roc_auc2 = auc(fpr2, tpr2)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'ROC curve for {model1.name} (area = {roc_auc1:.2f})')
    plt.plot(fpr2, tpr2, color='red', lw=2, label=f'ROC curve for {model2.name} (area = {roc_auc2:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve with returns threshold = {threshold:.3f}')
    plt.legend(loc="lower right")
    plt.show()


def plot_returns(data):
    """
    Plot Scenarios Return Result for the data
    :param data:
    :return:
    """
    plt.figure(figsize=(12, 8))
    # Plotting each scenario
    plt.plot(data['DateTime'], data['cumulative_returns_scenario1'], label='Either Deep Learning or Random Forest')
    plt.plot(data['DateTime'], data['cumulative_returns_scenario2'], label='Both Deep Learning and Random Forest')
    plt.plot(data['DateTime'], data['cumulative_returns_scenario3'], label='Only Deep Learning')
    plt.plot(data['DateTime'], data['cumulative_returns_scenario4'], label='Only Random Forest')
    plt.plot(data['DateTime'], data['cumulative_returns_scenario5'], label='All Buy')

    # Setting labels and title
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Cumulative Returns for All Scenarios')

    # Adding a legend
    plt.legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def plot_var_subplots(data, confidence_level=0.95):
    """
    Plot Value-At-Risk and Conditional Value-At-Risk
    :param data:
    :param confidence_level:
    :return:
    """
    # Define the scenarios
    scenarios = [
        'cumulative_returns_scenario1',
        'cumulative_returns_scenario2',
        'cumulative_returns_scenario3',
        'cumulative_returns_scenario4',
    ]

    scenarios_names = [
        'Either Deep Learning or Random Forest',
        'Both Deep Learning and Random Forest',
        'Only Deep Learning',
        'Only Random Forest',
    ]

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the array for easy iteration

    # Loop through each scenario and plot in a subplot
    for i, scenario in enumerate(scenarios):
        daily_change = data[scenario].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        # Calculate VaR for the scenario
        var = np.percentile(daily_change, 100 * (1 - confidence_level))
        cvar = daily_change[daily_change <= var].mean()

        # Plot histogram on the subplot
        axes[i].hist(daily_change, bins=30, alpha=0.7)
        axes[i].axvline(var, color='r', linestyle='dashed', linewidth=2, label=f'VaR: {var:.2f}%')
        axes[i].axvline(cvar, color='green', linestyle='dashed', linewidth=2, label=f'CVaR: {cvar:.2f}%')
        axes[i].set_title(scenarios_names[i])
        axes[i].set_xlabel('Daily Returns')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_best_threshold_results(model, thresholds, precision_0, precision_1, recall_0, recall_1):
    """
    Plot Scatter Plots for different threshold using a grid search method
    :param model:
    :param thresholds:
    :param precision_0:
    :param precision_1:
    :param recall_0:
    :param recall_1:
    :return:
    """
    plt.figure(figsize=(10, 6))
    for threshold, p0, p1, r0, r1 in zip(thresholds, precision_0, precision_1, recall_0, recall_1):
        plt.scatter(p0 * p1, r0 * r1)
        plt.text(p0 * p1, r0 * r1, f'{threshold:.3f}', fontsize=8)

    plt.xlabel('Precision0 * Precision1')
    plt.ylabel('Recall0 * Recall1')
    plt.title(f'{model.name} Precision and Recall Product for Various Thresholds')
    plt.show()