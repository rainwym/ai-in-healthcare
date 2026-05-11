import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

DATA_PATH = "data/healthcare-dataset-stroke-data.csv"
TARGET = "stroke"

def load_data(path): # Stroke dataset
    return pd.read_csv(path)

def clean_data(df): # Cleaning dataset
    df = df.copy()

    # Missing BMI values, filled with median BMI.
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    # Dropped ID
    df = df.drop(columns=["id"])

    return df

def save_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()

def save_glucose_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="avg_glucose_level", bins=30)
    plt.title("Distribution of Average Glucose Level")
    plt.xlabel("Average Glucose Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/glucose_distribution.png")
    plt.close()

def save_smoking_status_countplot(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="smoking_status")
    plt.title("Smoking Status Counts")
    plt.xlabel("Smoking Status")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("outputs/smoking_status_counts.png")
    plt.close()

def save_age_vs_stroke_boxplot(df): # Age vs Stroke 
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=TARGET, y="age")
    plt.title("Age Compared to Stroke")
    plt.xlabel("Stroke")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig("outputs/age_vs_stroke_boxplot.png")
    plt.close()

def save_smoking_vs_stroke_heatmap(df):
    table = pd.crosstab(df["smoking_status"], df[TARGET])

    plt.figure(figsize=(7, 5))
    sns.heatmap(table, annot=True, fmt="d")
    plt.title("Smoking Status vs Stroke")
    plt.xlabel("Stroke")
    plt.ylabel("Smoking Status")
    plt.tight_layout()
    plt.savefig("outputs/smoking_vs_stroke_heatmap.png")
    plt.close()

def save_pca_plot(df):
    X_data = df.drop(columns=[TARGET])
    y_data = df[TARGET]

    X_data = pd.get_dummies(X_data, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df[TARGET] = y_data

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=TARGET)
    plt.title("PCA Visualization of Stroke Dataset")
    plt.tight_layout()
    plt.savefig("outputs/pca_stroke_plot.png")
    plt.close()


# Model Preparation


def create_features(df):
    df = df.copy()

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 40, 60, 120],
        labels=["child", "young_adult", "middle_age", "older_adult"]
    )

    df["log_avg_glucose_level"] = np.log1p(df["avg_glucose_level"])

    df["age_glucose_interaction"] = df["age"] * df["avg_glucose_level"]

    return df

def prepare_model_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X = pd.get_dummies(X, drop_first=True)

    return X, y

def train_model(X, y):
    numerical_columns = [
        "age",
        "avg_glucose_level",
        "bmi",
        "log_avg_glucose_level",
        "age_glucose_interaction"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions

def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.3f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, predictions))

def save_confusion_matrix(y_test, predictions):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        normalize="true"
    )

    plt.title("Confusion Matrix for Stroke Prediction")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = create_features(df)

    save_correlation_heatmap(df)
    save_glucose_distribution(df)
    save_smoking_status_countplot(df)
    save_age_vs_stroke_boxplot(df)
    save_smoking_vs_stroke_heatmap(df)
    save_pca_plot(df)

    # Prep model
    X, y = prepare_model_data(df)
    model, X_test, y_test, predictions = train_model(X, y)
    evaluate_model(y_test, predictions)
    save_confusion_matrix(y_test, predictions)
    
    print("Success")

if __name__ == "__main__":
    main()