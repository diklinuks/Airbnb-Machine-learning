
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
WEB = ROOT / "web"
FIG.mkdir(parents=True, exist_ok=True)
WEB.mkdir(parents=True, exist_ok=True)

# Features used inside each stratum's k-means
WITHIN_FEATS = [
    "accommodates", "bedrooms", "beds", "bathrooms_n",
    "amenities_count",
    "latitude", "longitude",
]

# Strata definition — stratum_name -> set of room_type values
STRATA = {
    "entire": {"Entire home/apt"},
    "private": {"Private room"},
    "specialty": {"Hotel room", "Shared room"},
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sweep_k_within(X: np.ndarray, k_range, random_state=1, label=""):
    rows = []
    for k in k_range:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(4000, len(X)),
                               random_state=random_state)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        rows.append({"stratum": label, "k": k, "silhouette": sil,
                     "davies_bouldin": db, "calinski_harabasz": ch,
                     "inertia": km.inertia_})
        print(f"    k={k:2d}  sil={sil:.3f}  DB={db:.3f}  CH={ch:.0f}")
    return pd.DataFrame(rows)


MIN_CLUSTER_SIZE = 50


def pick_k(scores: pd.DataFrame, X: np.ndarray, random_state=1) -> int:
    """Pick k by silhouette + minimum-cluster-size constraint.

    A cluster smaller than MIN_CLUSTER_SIZE almost certainly captures
    outliers, not a market segment. Reject any k whose smallest cluster falls
    below that threshold.
    """
    if scores.empty:
        return 1
    candidates = scores.sort_values("silhouette", ascending=False)
    for _, row in candidates.iterrows():
        k = int(row["k"])
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(X)
        sizes = np.bincount(km.labels_)
        if sizes.min() >= MIN_CLUSTER_SIZE:
            return k
    # nothing satisfies the constraint -> fall back to k=2
    return 2


# per-stratum cap on k — small strata cannot support many segments
K_CAPS = {"entire": 3, "private": 2, "specialty": 1}


def fit_stratum(df_stratum: pd.DataFrame, name: str, random_state=1):
    """Return (labels, kmeans, scaler, best_k, sweep_scores) for a stratum."""
    print(f"\n--- stratum: {name}  (n = {len(df_stratum):,}) ---")
    cap = K_CAPS.get(name, 3)
    if len(df_stratum) < 100 or cap <= 1:
        print(f"  stratum too small or capped at k=1, no sub-clustering")
        return np.zeros(len(df_stratum), dtype=int), None, None, 1, pd.DataFrame()
    X = df_stratum[WITHIN_FEATS].astype(float).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    # sweep wider (2..6) for an informative plot; restrict the *choice* to the cap
    sweep_range = range(2, 7)
    scores = sweep_k_within(Xs, sweep_range, random_state=random_state, label=name)
    eligible = scores[scores["k"] <= cap]
    best_k = pick_k(eligible, Xs, random_state=random_state) if not eligible.empty else 1
    print(f"  chosen k = {best_k}  (cap = {cap})")
    km = KMeans(n_clusters=best_k, n_init=50, random_state=random_state)
    labels = km.fit_predict(Xs)
    return labels, km, scaler, best_k, scores


def characterise(df: pd.DataFrame, cluster_col="cluster"):
    profiles = []
    for c in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == c]
        prof = {
            "cluster": int(c),
            "stratum": sub["stratum"].iloc[0],
            "n": int(len(sub)),
            "share": float(len(sub) / len(df)),
            "median_price": float(sub["price"].median()),
            "mean_price": float(sub["price"].mean()),
            "p25_price": float(sub["price"].quantile(0.25)),
            "p75_price": float(sub["price"].quantile(0.75)),
            "median_accommodates": float(sub["accommodates"].median()),
            "median_bedrooms": float(sub["bedrooms"].median()),
            "median_beds": float(sub["beds"].astype(float).median()),
            "median_bathrooms": float(sub["bathrooms_n"].median()),
            "median_amenities": float(sub["amenities_count"].median()),
            "centroid_lat": float(sub["latitude"].mean()),
            "centroid_lon": float(sub["longitude"].mean()),
            "dominant_room_type": sub["room_type"].mode().iat[0],
            "room_type_breakdown": sub["room_type"].value_counts(normalize=True).round(3).to_dict(),
            "top_property_types": sub["property_type"].value_counts().head(3).to_dict(),
            "top_neighbourhoods": sub["neighbourhood_cleansed"].value_counts().head(5).to_dict(),
        }
        profiles.append(prof)
    return profiles


def label_clusters(profiles):
    by_stratum = {}
    for p in profiles:
        by_stratum.setdefault(p["stratum"], []).append(p)

    for stratum, group in by_stratum.items():
        group.sort(key=lambda p: p["median_accommodates"])
        n_in_stratum = len(group)
        stratum_name = {
            "entire": "entire-place",
            "private": "private-room",
            "specialty": "hotel/shared room",
        }[stratum]

        for i, p in enumerate(group):
            if n_in_stratum == 1:
                p["label"] = stratum_name.capitalize() + "s"
            else:
                if i == 0:
                    band = "Compact"
                elif i == n_in_stratum - 1:
                    band = "Spacious"
                elif n_in_stratum >= 3 and i == 1:
                    band = "Standard"
                else:
                    band = "Mid"
                p["label"] = f"{band} {stratum_name}s"
    return profiles

def plot_kselect_panels(score_frames, chosen_ks, caps):
    n = len(score_frames)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, scores) in zip(axes, score_frames.items()):
        if scores.empty:
            ax.set_title(f"{name}: n too small to cluster")
            ax.axis("off")
            continue
        ax.plot(scores["k"], scores["silhouette"], "o-", color="#3b82f6",
                label="silhouette", lw=1.6, markersize=7)
        cap = caps.get(name, 6)
        if cap < scores["k"].max():
            ax.axvspan(cap + 0.5, scores["k"].max() + 0.5,
                       color="#e5e7eb", alpha=0.5, zorder=0)
            ax.text(
                (cap + 1 + scores["k"].max()) / 2,
                scores["silhouette"].min() + 0.01,
                "excluded by\nstratum size cap",
                ha="center", va="bottom", fontsize=8, color="#6b7280",
            )
        ax.axvline(chosen_ks[name], ls="--", color="#ef4444", lw=1.4,
                   label=f"chosen k = {chosen_ks[name]}")
        ax.set_xlabel("k"); ax.set_ylabel("silhouette")
        ax.set_title(f"stratum: {name}\nsilhouette across k=2..6")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle("Choosing k per stratum — silhouette sweep + minimum-cluster-size guard",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / "cluster_kselect.png", dpi=130, bbox_inches="tight")
    plt.close()


def plot_pca_scatter(df, profiles):
    X = df[WITHIN_FEATS].astype(float).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=1)
    pts = pca.fit_transform(Xs)

    palette = sns.color_palette("Set2", n_colors=len(profiles))
    label_map = {p["cluster"]: p["label"] for p in profiles}

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for p, color in zip(profiles, palette):
        c = p["cluster"]
        mask = df["cluster"].values == c
        ax.scatter(pts[mask, 0], pts[mask, 1], s=5, alpha=0.4,
                   color=color, label=label_map[c])
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Clusters in shared structural-feature PCA space\n(viz only — PCA not used in clustering)")
    ax.legend(fontsize=8, markerscale=3, loc="best")
    plt.tight_layout()
    plt.savefig(FIG / "cluster_pca_scatter.png", dpi=130, bbox_inches="tight")
    plt.close()
    return pca


def plot_profiles(df, profiles):
    palette = sns.color_palette("Set2", n_colors=len(profiles))
    label_map = {p["cluster"]: p["label"] for p in profiles}
    df = df.copy()
    df["cluster_label"] = df["cluster"].map(label_map)
    order = [p["label"] for p in sorted(profiles, key=lambda p: p["median_price"])]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    sns.boxplot(
        data=df, x="cluster_label", y="price", order=order,
        ax=axes[0], palette=palette, showfliers=False,
    )
    axes[0].set_title("Price per cluster\n(OUTPUT — never an input)")
    axes[0].set_xlabel(""); axes[0].set_ylabel("€/night")
    axes[0].tick_params(axis="x", rotation=20, labelsize=8)

    sns.boxplot(
        data=df, x="cluster_label", y="accommodates", order=order,
        ax=axes[1], palette=palette, showfliers=False,
    )
    axes[1].set_title("Accommodates per cluster\n(structural input)")
    axes[1].set_xlabel(""); axes[1].set_ylabel("guests")
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)

    sns.boxplot(
        data=df, x="cluster_label", y="amenities_count", order=order,
        ax=axes[2], palette=palette, showfliers=False,
    )
    axes[2].set_title("Amenities count per cluster\n(structural input)")
    axes[2].set_xlabel(""); axes[2].set_ylabel("# amenities")
    axes[2].tick_params(axis="x", rotation=20, labelsize=8)

    plt.tight_layout()
    plt.savefig(FIG / "cluster_profiles.png", dpi=130, bbox_inches="tight")
    plt.close()


def plot_map(df, profiles):
    palette = sns.color_palette("Set2", n_colors=len(profiles))
    color_map = {p["cluster"]: palette[p["cluster"]] for p in profiles}
    label_map = {p["cluster"]: p["label"] for p in profiles}

    fig, ax = plt.subplots(figsize=(8.5, 9))
    for p in profiles:
        c = p["cluster"]
        sub = df[df["cluster"] == c]
        ax.scatter(sub["longitude"], sub["latitude"], s=4, alpha=0.5,
                   color=color_map[c], label=label_map[c])
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Where each cluster lives in Amsterdam")
    ax.legend(markerscale=3, fontsize=8, loc="upper left")
    ax.set_aspect(1.55)
    plt.tight_layout()
    plt.savefig(FIG / "cluster_map.png", dpi=130, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_parquet(PROC / "listings_clean.parquet").reset_index(drop=True)
    df["stratum"] = df["room_type"].map(
        lambda r: next((s for s, group in STRATA.items() if r in group), "specialty")
    )
    print(f"loaded {len(df):,} listings; stratum counts:")
    print(df["stratum"].value_counts())

    stratum_results = {}
    all_scores = {}
    next_id = 0
    df["cluster"] = -1

    for stratum in ["entire", "private", "specialty"]:
        mask = df["stratum"] == stratum
        sub = df[mask].reset_index()
        if len(sub) == 0:
            continue
        labels, km, scaler, best_k, scores = fit_stratum(sub, stratum)
        all_scores[stratum] = scores
        # remap local cluster ids to global ids
        global_labels = labels + next_id
        df.loc[sub["index"], "cluster"] = global_labels
        stratum_results[stratum] = {
            "kmeans": km, "scaler": scaler, "best_k": best_k,
            "global_id_offset": next_id,
        }
        next_id += best_k

    assert (df["cluster"] >= 0).all(), "some rows did not get a cluster assignment"
    k_total = next_id
    print(f"\ntotal clusters: {k_total}")

    profiles = characterise(df)
    profiles = label_clusters(profiles)

    print("\ncluster profiles (sorted by median price):")
    for p in sorted(profiles, key=lambda p: p["median_price"]):
        print(f"  [{p['cluster']}] {p['label']:<45s} "
              f"n={p['n']:>5,}  median €{p['median_price']:>4.0f}  "
              f"size~{p['median_accommodates']:.0f}  "
              f"dominant: {p['dominant_room_type']}")

    # plots
    plot_kselect_panels(
        all_scores,
        {s: r["best_k"] for s, r in stratum_results.items()},
        K_CAPS,
    )
    pca = plot_pca_scatter(df, profiles)
    plot_profiles(df, profiles)
    plot_map(df, profiles)

    nb = df.groupby(["neighbourhood_cleansed", "cluster"]).size().unstack(fill_value=0)
    cluster_cols = list(range(k_total))
    # ensure every cluster column exists
    for c in cluster_cols:
        if c not in nb.columns:
            nb[c] = 0
    nb = nb[cluster_cols]
    nb["total"] = nb[cluster_cols].sum(axis=1)
    nb["dominant_cluster"] = nb[cluster_cols].idxmax(axis=1).astype(int)

    neigh_breakdown = {}
    for name, row in nb.iterrows():
        counts = {int(c): int(row[c]) for c in cluster_cols}
        neigh_breakdown[name] = {
            "dominant_cluster": int(row["dominant_cluster"]),
            "counts": counts,
            "total": int(row["total"]),
            "median_price": float(df[df["neighbourhood_cleansed"] == name]["price"].median()),
            "median_accommodates": float(df[df["neighbourhood_cleansed"] == name]["accommodates"].median()),
        }

    df.to_parquet(PROC / "listings_with_clusters.parquet", index=False)
    with open(PROC / "cluster_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)
    with open(PROC / "cluster_neighbourhood_breakdown.json", "w") as f:
        json.dump(neigh_breakdown, f, indent=2)
    joblib.dump(
        {
            "stratum_results": stratum_results,
            "within_feats": WITHIN_FEATS,
            "strata": STRATA,
            "k_total": k_total,
        },
        PROC / "cluster_model.joblib",
    )

    centroid_payload = {}
    for stratum, info in stratum_results.items():
        if info["kmeans"] is None:
            continue
        centers_scaled = info["kmeans"].cluster_centers_
        centers_raw = info["scaler"].inverse_transform(centers_scaled)
        centroid_payload[stratum] = {
            "feature_order": WITHIN_FEATS,
            "centroids_raw": centers_raw.tolist(),
            "scaler_mean": info["scaler"].mean_.tolist(),
            "scaler_scale": info["scaler"].scale_.tolist(),
            "global_id_offset": info["global_id_offset"],
            "k": info["best_k"],
        }

    web_payload = {
        "version": 2,
        "n_listings": int(len(df)),
        "k_total": k_total,
        "strata": list(STRATA.keys()),
        "stratum_room_types": {k: list(v) for k, v in STRATA.items()},
        "feature_order": WITHIN_FEATS,
        "centroids_by_stratum": centroid_payload,
        "clusters": [
            {
                "id": p["cluster"],
                "stratum": p["stratum"],
                "label": p["label"],
                "color_index": p["cluster"],
                "n": p["n"],
                "share": p["share"],
                "median_price": p["median_price"],
                "p25_price": p["p25_price"],
                "p75_price": p["p75_price"],
                "median_accommodates": p["median_accommodates"],
                "median_bedrooms": p["median_bedrooms"],
                "median_beds": p["median_beds"],
                "median_bathrooms": p["median_bathrooms"],
                "median_amenities": p["median_amenities"],
                "centroid_lat": p["centroid_lat"],
                "centroid_lon": p["centroid_lon"],
                "dominant_room_type": p["dominant_room_type"],
                "room_type_breakdown": p["room_type_breakdown"],
                "top_property_types": p["top_property_types"],
                "top_neighbourhoods": p["top_neighbourhoods"],
            }
            for p in profiles
        ],
        "neighbourhoods": neigh_breakdown,
        "k_selection": {s: scores.to_dict(orient="records")
                        for s, scores in all_scores.items() if not scores.empty},
    }
    with open(WEB / "data.json", "w") as f:
        json.dump(web_payload, f, indent=2)

    print(f"\nartifacts:\n  {PROC}\n  {FIG}\n  {WEB / 'data.json'}\n")


if __name__ == "__main__":
    main()
