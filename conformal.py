import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def calculate_coverage_score(y_true, prediction_sets): # Calculates frequency of real answer inside prediction set
    covered = []

    for true_class, prediction_set in zip(y_true, prediction_sets):
        covered.append(true_class in prediction_set)

    return np.mean(covered)


def create_prediction_sets(probabilities, threshold): # Prediction sets using class probabilities
    prediction_sets = []

    for row in probabilities:
        current_set = []

        for class_label, probability in enumerate(row):
            if probability >= threshold:
                current_set.append(class_label)

        prediction_sets.append(current_set)

    return prediction_sets


def save_data_partitions(X_train, X_cal, X_test, y_train, y_cal, y_test): # Saves train/cal/test datasets
    X_train.to_csv("outputs/X_train.csv", index=False)
    X_cal.to_csv("outputs/X_cal.csv", index=False)
    X_test.to_csv("outputs/X_test.csv", index=False)

    y_train.to_csv("outputs/y_train.csv", index=False)
    y_cal.to_csv("outputs/y_cal.csv", index=False)
    y_test.to_csv("outputs/y_test.csv", index=False)


def save_conformal_confidence_plots(probabilities, set_sizes): # Saves plots that shows relationship btwn model confidence/conformal prediction set size
    confidence = probabilities.max(axis=1)

    confidence_df = pd.DataFrame(
        {
            "confidence": confidence,
            "set_size": set_sizes
        }
    )

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=confidence_df, x="set_size", y="confidence")
    plt.title("Model Confidence by Prediction Set Size")
    plt.xlabel("Prediction Set Size")
    plt.ylabel("Model Confidence")
    plt.tight_layout()
    plt.savefig("outputs/conformal_confidence_boxplot.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.violinplot(data=confidence_df, x="set_size", y="confidence")
    plt.title("Confidence Distribution by Prediction Set Size")
    plt.xlabel("Prediction Set Size")
    plt.ylabel("Model Confidence")
    plt.tight_layout()
    plt.savefig("outputs/conformal_confidence_violinplot.png")
    plt.close()


def run_conformal_prediction(preprocessor, X, y): # Manual conformal prediction for stroke classification model
    confidence_level = 0.90
    alpha = 1 - confidence_level

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
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

    calibration_probabilities = model.predict_proba(X_cal)

    calibration_scores = []

    for i, true_class in enumerate(y_cal):
        probability_of_true_class = calibration_probabilities[i, true_class]
        nonconformity_score = 1 - probability_of_true_class
        calibration_scores.append(nonconformity_score)

    conformal_score = np.quantile(calibration_scores, 1 - alpha)
    probability_threshold = 1 - conformal_score

    test_probabilities = model.predict_proba(X_test)
    prediction_sets = create_prediction_sets(test_probabilities, probability_threshold)

    coverage_score = calculate_coverage_score(y_test, prediction_sets)
    set_sizes = [len(prediction_set) for prediction_set in prediction_sets]

    print("Conformal Prediction Results:")
    print(f"Target confidence level: {confidence_level}")
    print(f"Probability threshold: {probability_threshold:.3f}")
    print(f"Coverage score: {coverage_score:.3f}")
    print(f"Average prediction set size: {np.mean(set_sizes):.3f}")
    print("Prediction set size counts:")
    print(pd.Series(set_sizes).value_counts().sort_index())
    print()

    save_conformal_confidence_plots(test_probabilities, set_sizes)
    save_data_partitions(X_train, X_cal, X_test, y_train, y_cal, y_test)

    joblib.dump(model, "outputs/stroke_conformal_model.pkl")

    return prediction_sets