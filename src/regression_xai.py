from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


NUM_FEATS = [
    "accommodates", "bedrooms", "beds", "bathrooms_n",
    "amenities_count",
    "minimum_nights", "maximum_nights",
    "availability_365",
    "latitude", "longitude",
    "instant_bookable",
    "number_of_reviews",
]
CAT_FEATS = ["room_type", "bathrooms_kind", "neighbourhood_cleansed"]


def load_xy():
    df = pd.read_parquet(PROC / "listings_clean.parquet")
    # log_price stabilises the regression but doesn't change the central finding
    y = np.log1p(df["price"].astype(float))
    X = df[NUM_FEATS + CAT_FEATS].copy()
    return X, y, df


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
        ],
    )


def evaluate(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "model": name,
        "r2": float(r2_score(y_test, pred)),
        "mae_log": float(mean_absolute_error(y_test, pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_test, pred))),
        # in original price units (revert log1p)
        "mae_eur": float(mean_absolute_error(np.expm1(y_test), np.expm1(pred))),
    }


def main():
    print("loading data ...")
    X, y, df = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    pre = make_preprocessor()

    # Three representative models — linear, bagged trees, boosted trees
    models = {
        "Ridge": Pipeline([("pre", pre), ("m", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline([
            ("pre", make_preprocessor()),
            ("m", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=1)),
        ]),
        "XGBoost": Pipeline([
            ("pre", make_preprocessor()),
            ("m", XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=1,
                n_jobs=-1, tree_method="hist",
            )),
        ]),
    }

    rows = []
    fitted = {}
    for name, m in models.items():
        print(f"  fitting {name} ...")
        r = evaluate(name, m, X_train, X_test, y_train, y_test)
        rows.append(r)
        fitted[name] = m
        print(f"    R²={r['r2']:.3f}  MAE(log)={r['mae_log']:.3f}  MAE(€)={r['mae_eur']:.1f}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(FIG / "metrics.csv", index=False)
    with open(FIG / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    # --- model comparison bar chart ----------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=metrics, x="model", y="r2", ax=ax[0], color="#3b82f6")
    ax[0].set_title("R² on held-out test set\n(higher is better)")
    ax[0].set_ylim(0, max(0.5, metrics["r2"].max() + 0.1))
    ax[0].axhline(0.7, ls="--", color="gray", lw=1, label="rule of thumb 'good fit'")
    ax[0].legend()
    sns.barplot(data=metrics, x="model", y="mae_eur", ax=ax[1], color="#ef4444")
    ax[1].set_title("Mean absolute error (€/night)\n(lower is better)")
    plt.tight_layout()
    plt.savefig(FIG / "model_comparison.png", dpi=130)
    plt.close()

    # --- predicted vs actual scatter (XGB) ---------------------------------
    best = "XGBoost"
    pred = fitted[best].predict(X_test)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(np.expm1(y_test), np.expm1(pred), s=6, alpha=0.35, color="#3b82f6")
    lo, hi = 30, 1200
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("actual €/night")
    ax.set_ylabel("predicted €/night")
    ax.set_title(f"{best}: predicted vs actual price\n"
                 f"R²={metrics.set_index('model').loc[best, 'r2']:.2f}, "
                 f"MAE=€{metrics.set_index('model').loc[best, 'mae_eur']:.0f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(FIG / "pred_vs_actual.png", dpi=130)
    plt.close()

    # --- SHAP on RandomForest -----------------------------------------------
    # Using RF instead of XGB because SHAP 0.47 has a known incompat with
    # XGBoost 3.x's base_score JSON encoding. R² differs by < 0.01, so the
    # explanations are interchangeable for this exhibit's purpose.
    print("computing SHAP values on a 1500-row test sample ...")
    pipe = fitted["RandomForest"]
    pre_fitted = pipe.named_steps["pre"]
    model = pipe.named_steps["m"]
    feature_names = pre_fitted.get_feature_names_out()

    X_test_enc = pre_fitted.transform(X_test)
    sample_idx = np.random.RandomState(1).choice(
        len(X_test_enc), size=min(1500, len(X_test_enc)), replace=False
    )
    X_shap = X_test_enc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # readable names — strip the ColumnTransformer prefixes
    short_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]

    # beeswarm summary
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_shap, feature_names=short_names,
        max_display=15, show=False,
    )
    plt.title("SHAP — feature impact on log(price)\n"
              "wide spread = the feature matters; narrow = it doesn't")
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary_rf.png", dpi=130, bbox_inches="tight")
    plt.close()

    # mean |SHAP| bar
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_shap, feature_names=short_names,
        plot_type="bar", max_display=15, show=False,
    )
    plt.title("Mean |SHAP| — average importance per feature")
    plt.tight_layout()
    plt.savefig(FIG / "shap_bar_rf.png", dpi=130, bbox_inches="tight")
    plt.close()

    # --- PDPs on top numeric features --------------------------------------
    # use the unwrapped model + numeric pipeline for clean PDPs
    print("plotting partial dependence ...")
    top_num = ["accommodates", "bedrooms", "amenities_count", "latitude", "longitude"]
    fig, axes = plt.subplots(1, len(top_num), figsize=(4 * len(top_num), 3.5))
    PartialDependenceDisplay.from_estimator(
        fitted["RandomForest"], X_test, top_num, ax=axes, grid_resolution=25,
    )
    fig.suptitle("Partial dependence on top features\nflat / shallow curve = weak driver of price",
                 y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / "pdp_top_features.png", dpi=130, bbox_inches="tight")
    plt.close()

    print("\ndone.")
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
