import warnings
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import shap

warnings.filterwarnings("ignore", message="X has feature names, but LogisticRegression was fitted without feature names")
warnings.filterwarnings("ignore", category=FutureWarning, message="The NumPy global RNG was seeded")


def run_shap_analysis(model, X_test): # SHAP feature importance analysis for stroke model
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    # Transform X_test to all-numeric using the already-fitted preprocessor
    # SHAP cannot handle string/categorical columns directly
    X_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    sample = X_transformed_df.sample(100, random_state=42)

    masker = shap.maskers.Independent(sample)
    explainer = shap.Explainer(classifier.predict_proba, masker)
    shap_values = explainer(X_transformed_df)

    joblib.dump(shap_values, "outputs/shap_values.joblib")

    stroke_shap = shap_values[:, :, 1] # Index 1 = stroke class (0 = no stroke, 1 = stroke)

    save_beeswarm_plot(stroke_shap, X_transformed_df)
    save_violin_plot(stroke_shap, X_transformed_df)
    save_dependence_plots(stroke_shap, X_transformed_df)
    save_waterfall_plot(stroke_shap)

    print("SHAP analysis complete. Plots saved to outputs/")


def save_beeswarm_plot(stroke_shap, X_transformed_df): # Global feature importance across all predictions
    shap.summary_plot(stroke_shap.values, X_transformed_df, show=False)
    plt.title("SHAP Feature Importance - Stroke Prediction")
    plt.tight_layout()
    plt.savefig("outputs/shap_beeswarm.png", bbox_inches="tight")
    plt.close()


def save_violin_plot(stroke_shap, X_transformed_df): # Same as beeswarm but with smoothed density curves
    shap.plots.violin(stroke_shap, features=X_transformed_df, plot_type="layered_violin", show=False)
    plt.title("SHAP Violin Plot - Stroke Prediction")
    plt.tight_layout()
    plt.savefig("outputs/shap_violin.png", bbox_inches="tight")
    plt.close()


def save_dependence_plots(stroke_shap, X_transformed_df): # How changing each feature shifts the stroke prediction
    key_features = ["numeric__age", "numeric__avg_glucose_level", "numeric__age_glucose_interaction"]

    for feature in key_features:
        shap.dependence_plot(feature, stroke_shap.values, X_transformed_df, show=False)
        plt.tight_layout()
        plt.savefig(f"outputs/shap_dependence_{feature}.png", bbox_inches="tight")
        plt.close()


def save_waterfall_plot(stroke_shap): # Breaks down how features pushed one specific prediction up or down
    shap.plots.waterfall(stroke_shap[0], show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_waterfall_sample0.png", bbox_inches="tight")
    plt.close()
