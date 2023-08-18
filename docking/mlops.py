import mlflow
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    plot_precision_recall_curve,
    plot_roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV


def set_credentials():
    with open("config.json") as f:
        config = json.load(f)

    credentials = config["credentials"]

    for key, value in credentials.items():
        os.environ[key] = value

    mlflow.set_tracking_uri(
        f"https://dagshub.com/"
        + os.environ["MLFLOW_TRACKING_USERNAME"]
        + "/"
        + os.environ["MLFLOW_TRACKING_PROJECTNAME"]
        + ".mlflow"
    )


def train_model(x_train, x_test, y_train, y_test, ml_model, params):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model = GridSearchCV(ml_model, params, cv=kfold, n_jobs=-1)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    best_params = model.best_params_

    fig, ax = plt.subplots()
    plot_precision_recall_curve(model, x_test, y_test, ax=ax)
    plt.savefig("precision_recall_plot.png")
    precision_recall_image = open("precision_recall_plot.png", "rb").read()
    os.remove("precision_recall_plot.png")

    fig, ax = plt.subplots()
    plot_roc_curve(model, x_test, y_test, ax=ax)
    plt.savefig("roc_auc_plot.png")
    roc_auc_image = open("roc_auc_plot.png", "rb").read()
    os.remove("roc_auc_plot.png")

    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.savefig("confusion_matrix.png")
    confusion_matrix_image = open("confusion_matrix.png", "rb").read()
    os.remove("confusion_matrix.png")

    print("Best parameters found by GridSearchCV:", best_params)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "roc_auc": roc_auc,
        "best_params": best_params
    }
