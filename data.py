import os
import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.utils import resample


def undersample(df):
    df_majority = df[df["Outcome"] == 0]
    df_minority = df[df["Outcome"] == 1]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=int(len(df_majority) * 2 / 3),
        random_state=42,
    )

    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced


def _median_target(df, column):
    medians = df.groupby("Outcome")[column].median()

    mask_0 = (df["Outcome"] == 0) & (df[column].isna())
    mask_1 = (df["Outcome"] == 1) & (df[column].isna())

    multiplier = 1
    if column == "Insulin":
        multiplier = 2.75
    elif column == "BMI":
        multiplier = 0.5
    random_offsets_0 = np.random.randint(
        -10 * multiplier, 5 * multiplier, size=mask_0.sum()
    )
    random_offsets_1 = np.random.randint(
        -5 * multiplier, 15 * multiplier, size=mask_1.sum()
    )

    df.loc[mask_0, column] = medians[0] + random_offsets_0
    df.loc[mask_1, column] = medians[1] + random_offsets_1


def _download_dataset() -> pd.DataFrame:
    name = "diabetes.csv"
    path = kagglehub.dataset_download("akshaydattatraykhare/diabetes-dataset")
    df = pd.read_csv(os.path.join(path, name))
    df.drop("DiabetesPedigreeFunction", axis=1, inplace=True)
    df.drop("Pregnancies", axis=1, inplace=True)
    return df


def _format_dataset(df: pd.DataFrame):
    df = df.copy()
    for col in df.columns:
        if col not in ("Outcome"):
            df[col] = df[col].replace(to_replace=0, value=np.nan)
    for col in df.columns:
        _median_target(df, col)
    return df


def _plot_outcome_distribution(df: pd.DataFrame, title=""):
    outcome_counts = df["Outcome"].value_counts()
    ax = outcome_counts.plot(
        figsize=(6, 5), kind="bar", color=["skyblue", "salmon"], edgecolor="black"
    )
    for i, count in enumerate(outcome_counts):
        ax.text(i, count + 1, str(count), ha="center", va="bottom", fontsize=11)

    plt.title("Diabetes Outcome Distribution")
    plt.xlabel("Outcome (0 = Non-Diabetic, 1 = Diabetic)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        f"./images/{f'{title.lower()}_' if title else ''}outcome_distributions.png"
    )
    plt.close()


def _plot_features(df: pd.DataFrame, title=""):
    df = df.drop("Outcome", axis=1)

    num_features = len(df.columns)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 4, num_rows * 3))

    for idx, column in enumerate(df.columns, 1):
        plt.subplot(num_rows, num_cols, idx)
        sns.histplot(data=df[column], kde=True)
        plt.title(column)
        plt.xlabel("")

    plt.tight_layout()
    plt.suptitle("Feature Distributions", fontsize=16, y=1.02)
    filename = f"./images/{f'{title.lower()}_' if title else ''}features.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def _plot_feature_distributions(df: pd.DataFrame, title=""):
    for feature in df.columns.drop("Outcome"):
        plt.figure(figsize=(6, 5))
        sns.histplot(
            data=df,
            x=feature,
            hue="Outcome",
            kde=True,
            palette=["green", "red"],
            alpha=0.6,
        )
        plt.title(f"Distribution of {feature} by Diabetes Outcome")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.savefig(
            f"./images/{f'{title.lower()}_' if title else ''}feature_distributions_{feature}.png"
        )
        plt.close()


def _plot_feature_importance(df: pd.DataFrame, title=""):
    correlations = df.corr()["Outcome"].drop("Outcome")
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=correlations.index,
        y=correlations.values,
        hue=correlations.index,
        palette="coolwarm",
    )
    plt.xticks(rotation=45)
    plt.title("Correlation of Features with Outcome")
    plt.ylabel("Correlation Coefficient")
    plt.xlabel("Features")
    plt.savefig(
        f"./images/{f'{title.lower()}_' if title else ''}feature_importance.png",
        bbox_inches="tight",
    )
    plt.close()


def _plot_correlation_heatmap(df, title=""):
    df = df.drop(columns=["Outcome"])
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        center=0.3,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        f"./images/{f'{title.lower()}_' if title else ''}correlation_heatmap.png",
        bbox_inches="tight",
    )
    plt.close()


def _plot_confusion_matrix(y_test, y_pred, type, accuracy):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix(
            y_test,
            y_pred,
            normalize="true",
        ),
        annot=True,
        cmap="Blues",
        fmt=".2f",
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({type}) ({accuracy:.2f})")
    plt.savefig(f"./results/{type}_confusion_matrix.png", bbox_inches="tight")
    plt.close()


def _plot_roc(y_test, y_pred, type, accuracy):
    fpr, tpr, _thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({type}) ({accuracy:.2f})")
    plt.legend(loc="lower right")
    plt.savefig(f"./results/{type}_roc.png", bbox_inches="tight")
    plt.close()


def get_dataset():
    df = _download_dataset()
    return _format_dataset(df)


def evaluate_model(y_test, y_pred, type):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy ({type}): {accuracy:.2f}")
    _plot_confusion_matrix(y_test, y_pred, type, accuracy)
    _plot_roc(y_test, y_pred, type, accuracy)
    plt.show()


if __name__ == "__main__":
    df = get_dataset()
    df = undersample(df)
    title = "Undersampled"
    sns.set_theme(style="whitegrid")
    _plot_features(df, title)
    _plot_outcome_distribution(df, title)
    _plot_feature_importance(df, title)
    _plot_feature_distributions(df, title)
    _plot_correlation_heatmap(df, title)
