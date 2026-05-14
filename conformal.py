import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def calculate_coverage_score(y_true, prediction_sets):
    """
    Calculates how often the true class is inside the conformal prediction set.

    Example:
    true class = 1
    prediction set = [0, 1]
    This counts as covered because 1 is inside the set.

    true class = 1
    prediction set = [0]
    This does not count as covered.
    """

    covered = []

    for true_class, prediction_set in zip(y_true, prediction_sets):
        covered.append(true_class in prediction_set)

    return np.mean(covered)


def create_prediction_sets(probabilities, threshold):
    """
    Creates conformal prediction sets from predicted probabilities.

    The model gives probabilities like:
    class 0 probability = 0.80
    class 1 probability = 0.20

    We include a class in the prediction set if its probability
    is high enough based on the conformal threshold.
    """

    prediction_sets = []

    for row in probabilities:
        current_set = []

        for class_label, probability in enumerate(row):
            if probability >= threshold:
                current_set.append(class_label)

        prediction_sets.append(current_set)

    return prediction_sets


def run_conformal_prediction(preprocessor, X, y):
    """
    Runs manual conformal prediction for the stroke classification model.

    Normal classification gives one answer:
    0 = no stroke
    1 = stroke

    Conformal prediction gives a set of possible answers:
    [0]
    [1]
    [0, 1]

    [0] means the model is confident in no stroke.
    [1] means the model is confident in stroke.
    [0, 1] means the model is unsure, so it includes both.
    """

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

    conformal_score = np.quantile(
        calibration_scores,
        1 - alpha
    )

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

    save_conformal_confidence_plot(test_probabilities, set_sizes)

    return prediction_sets


def save_conformal_confidence_plot(probabilities, set_sizes):
    """
    Saves a plot comparing model confidence and conformal prediction set size.

    Usually:
    set size 1 = more confident
    set size 2 = less confident
    """

    confidence = probabilities.max(axis=1)

    confidence_df = pd.DataFrame(
        {
            "confidence": confidence,
            "set_size": set_sizes
        }
    )

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=confidence_df, x="set_size", y="confidence")
    plt.title("Model Confidence by Conformal Prediction Set Size")
    plt.xlabel("Prediction Set Size")
    plt.ylabel("Model Confidence")
    plt.tight_layout()
    plt.savefig("outputs/conformal_confidence_boxplot.png")
    plt.close()