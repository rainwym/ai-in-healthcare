import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

TARGET = "stroke"

def split_features_and_target(df): # Separate dataset into X(no target) and y(target/prediction)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y

def build_preprocessor(X): # Preprocessing dataset that can handle both num/cat inputs
    numeric_columns = X.select_dtypes(include="number").columns
    categorical_columns = X.select_dtypes(exclude="number").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ]
    )
    return preprocessor

def evaluate_with_cross_validation(model, X, y): # Model tests on different dataset splits
    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="recall"
    )

    print("Cross-validation recall scores:")
    print(scores)
    print(f"Mean recall: {np.mean(scores):.3f}")
    print(f"Standard deviation: {np.std(scores):.3f}")
    print()

def compare_regularization(preprocessor, X, y): # Regularization
    print("L2 Logistic Regression:")
    l2_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ]
    )
    evaluate_with_cross_validation(l2_model, X, y)

def compare_classification_models(preprocessor, X, y): # Compares LR, Random Forest, & SVC models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "Support Vector Classifier": SVC(class_weight="balanced")
    }

    for model_name, model in models.items():
        print(model_name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )
        evaluate_with_cross_validation(pipeline, X, y)

def train_final_model(preprocessor, X, y): # Final model
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Final Model Classification Report:")
    print(classification_report(y_test, predictions))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        normalize="true"
    )

    plt.title("Final Stroke Prediction Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/final_confusion_matrix.png")
    plt.close()
    return model, X_train, X_test, y_train, y_test

def save_calibration_plot(preprocessor, X_train, X_test, y_train, y_test): # Save model output
    base_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ]
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5
    )

    base_model.fit(X_train, y_train)
    calibrated_model.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(6, 6))

    CalibrationDisplay.from_estimator(
        base_model,
        X_test,
        y_test,
        n_bins=10,
        name="Uncalibrated",
        ax=ax
    )

    CalibrationDisplay.from_estimator(
        calibrated_model,
        X_test,
        y_test,
        n_bins=10,
        name="Calibrated",
        ax=ax
    )

    ax.set_title("Stroke Prediction Calibration Plot")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual stroke rate")
    plt.tight_layout()
    plt.savefig("outputs/calibration_plot.png")
    plt.close()