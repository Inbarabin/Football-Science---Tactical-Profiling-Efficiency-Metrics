import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def load_features(league, season=2024, scaled=True):
    suffix = "features_scaled.pkl" if scaled else "features.pkl"
    path = Path("data/features") / league / f"{season}_{suffix}"

    with open(path, "rb") as f:
        features = pickle.load(f)
    print("Data loaded successfully!")
    return features


def select_k(X, k_min=4, k_max=15, random_state=42, sil_tol=0.005, plot=True):

    silhouette = {}
    inertia = {}

    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=random_state
        )
        labels = model.fit_predict(X)

        silhouette[k] = silhouette_score(X, labels)
        inertia[k] = model.inertia_

    # ---- Silhouette threshold ----
    max_sil = max(silhouette.values())

    # candidates close to best silhouette
    candidates = [
        k for k, s in silhouette.items()
        if max_sil - s <= sil_tol
    ]

    # ---- Elbow logic: pick smallest reasonable K ----
    best_k = min(candidates)


    # ---- Plot Elbow (Inertia) ----
    if plot:
        ks = list(inertia.keys())
        inertias = list(inertia.values())
        silhouettes = list(silhouette.values())

        plt.figure()
        plt.plot(ks, inertias, marker='o')
        plt.axvline(best_k, linestyle='--')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method (Inertia vs k)")
        plt.show()

        plt.figure()
        plt.plot(ks, silhouettes, marker='o')
        plt.axvline(best_k, linestyle='--')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette Score vs k")
        plt.show()


    print("Selected K:", best_k)
    print("Silhouette:  \n", silhouette)
    print("Inertia: \n", inertia)

    return best_k
def run_kmeans(features, X, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    features["cluster"] = kmeans.fit_predict(X)
    print("\n📊 Cluster distribution:")
    print(features["cluster"].value_counts())
    for i in range(kmeans.n_clusters):
        print(f"\nCluster {i}")
        print(features[features["cluster"] == i]["team"].values)

    return kmeans


def apply_pca(features, X, name_tag="General", n_components=2,):
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"\n PCA of {name_tag} explained variance ratio: {explained:.2%}")
    # --- Add PCA components to 'features' dataframe ---
    for i in range(n_components):
        col_name = f"{name_tag}_PCA{i+1}"
        features[col_name] = result[:, i]

    # --- Build PCA loadings dynamically ---
    component_names = [f"{name_tag}_PCA{i+1}" for i in range(n_components)]

    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=component_names
    )

    print("\nTop contributing features for each PCA component:")
    for i in range(n_components):
        comp = pca_loadings.iloc[:, i]
        print(f"{name_tag}_PCA{i+1} (explains {pca.explained_variance_ratio_[i]:.2%} of variance):\n")
        print(comp.sort_values(key=abs, ascending=False).head(8))

    return pca, result, pca_loadings


def apply_tsne(X, perplexity=6, learning_rate=200, max_iter=1000):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42
    )
    result = tsne.fit_transform(X)
    print(f"✅ t-SNE finished. KL divergence: {tsne.kl_divergence_:.3f}")
    return tsne, result


def visualize_2d(features, x_col, y_col, cluster_col, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        features[x_col],
        features[y_col],
        c=features[cluster_col],
        cmap="Dark2",
        s=70,
        edgecolor="black"
    )
    for i, team in enumerate(features["team"]):
        plt.text(features[x_col][i], features[y_col][i], team, fontsize=8, alpha=0.7)
    plt.title(title)
    plt.xlabel(f"{x_col}")
    plt.ylabel(f"{y_col}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.show()


def compute_cluster_summary(features):
    only_num = features.drop(columns=["team"])
    cluster_means = only_num.groupby("cluster").mean()
    profiles = (cluster_means - cluster_means.mean()) / cluster_means.std()

    summary = []
    for i in range(profiles.shape[0]):
        data = profiles.iloc[i]
        summary.append({
            "cluster": i,
            "strong_features": (data > 0).sum(),
            "weak_features": (data < 0).sum(),
            "avg_score": data.mean()
        })

    df = pd.DataFrame(summary)
    print("\n📊 Cluster Summary:")
    print(df.sort_values("avg_score", ascending=False))
    return df

def top_nteams(features, n=3):
    for col in features.columns:
        if col not in ["team", "cluster"]:
            topn = features.nlargest(n, col)[["team", col]]
            print(f"\n🏆 Top {n} teams in {col}:")
            print(topn.to_string(index=False))


def main():
    features = load_features()
    X = features.drop(columns=["team"])
    offensive_cols = features.columns[1:13]   # Offensive metrics
    defensive_cols = list(features.columns[1:1]) + list(features.columns[14:24])    # Defensive metrics


    # Cluster profile
    k = select_k(X)
    kmeans = run_kmeans(features, X, k)
    compute_cluster_summary(features)
    # PCA
    pca, pca_result ,pca_loading= apply_pca(features, X)
    visualize_2d(features, "General_PCA1", "General_PCA2", "cluster", "PCA Projection")
    # t-SNE
    tsne, tsne_result = apply_tsne(X,4)
    features["tSNE1"] = tsne_result[:, 0]
    features["tSNE2"] = tsne_result[:, 1]
    visualize_2d(features, "tSNE1", "tSNE2", "cluster", "t-SNE Projection")

    #Offence Defence axis
    X_off = features[offensive_cols]
    X_def = features[defensive_cols]
    pca_off, pca_off_result,pca_off_loading = apply_pca(features, X_off, "offense", 1)
    pca_def, pca_def_result,pca_def_loading = apply_pca(features, X_def, "defense", 1)
    visualize_2d(features, "offense_PCA1", "defense_PCA1", "cluster", "Off/Def Projection")

    #top_nteams(features)



if __name__ == "__main__":
    main()
