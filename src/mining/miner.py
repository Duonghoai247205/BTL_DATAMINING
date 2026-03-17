import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển DataFrame thành ma trận one-hot cho Apriori.

    Mỗi giao dịch gồm: crop_X, rain_X, temp_X, pest_X, yield_X.

    Parameters
    ----------
    df : DataFrame với các cột yield_cat, rain_cat, temp_cat, pest_cat, Item

    Returns
    -------
    df_te : DataFrame boolean (one-hot encoded transactions)
    """
    transactions = []
    for _, row in df.iterrows():
        t = [
            f"crop_{str(row['Item']).replace(', ', '_').replace(' ', '_')}",
            f"rain_{row['rain_cat']}",
            f"temp_{row['temp_cat']}",
            f"pest_{row['pest_cat']}",
            f"yield_{row['yield_cat']}",
        ]
        transactions.append(t)

    te = TransactionEncoder()
    te_arr = te.fit_transform(transactions)
    df_te = pd.DataFrame(te_arr, columns=te.columns_)
    print(f"[miner] Transaction matrix: {df_te.shape[0]:,} dòng × {df_te.shape[1]} items")
    return df_te


def run_apriori(
    df_te: pd.DataFrame,
    min_support: float = 0.04,
    min_confidence: float = 0.45,
    max_len: int = 5,
) -> dict:
    """
    Chạy Apriori và sinh luật kết hợp.

    Parameters
    ----------
    df_te          : kết quả từ build_transactions()
    min_support    : ngưỡng support tối thiểu (default 4%)
    min_confidence : ngưỡng confidence tối thiểu (default 45%)
    max_len        : độ dài tối đa của itemset

    Returns
    -------
    dict với keys: 'freq_sets', 'rules', 'rules_high', 'rules_low'
    """
    print(f"[miner] Apriori: min_support={min_support}, min_confidence={min_confidence}")

    freq_sets = apriori(
        df_te, min_support=min_support, use_colnames=True, max_len=max_len
    )
    rules = association_rules(
        freq_sets, metric="confidence", min_threshold=min_confidence,
        num_itemsets=len(freq_sets)
    )
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    rules_high = rules[rules["consequents"].astype(str).str.contains("yield_High")]
    rules_low  = rules[rules["consequents"].astype(str).str.contains("yield_Low")]

    print(f"  Frequent itemsets : {len(freq_sets)}")
    print(f"  Tổng luật         : {len(rules)}")
    print(f"  Luật → High Yield : {len(rules_high)}")
    print(f"  Luật → Low Yield  : {len(rules_low)}")

    return {
        "freq_sets":   freq_sets,
        "rules":       rules,
        "rules_high":  rules_high,
        "rules_low":   rules_low,
    }


def find_best_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> dict:
    """
    Tìm K tối ưu bằng Elbow method + Silhouette score.

    Returns
    -------
    dict: {'best_k', 'inertias', 'silhouettes', 'k_range'}
    """
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"[miner] Best K = {best_k}  (Silhouette = {max(silhouettes):.4f})")
    return {
        "best_k":      best_k,
        "inertias":    inertias,
        "silhouettes": silhouettes,
        "k_range":     list(k_range),
    }


def run_kmeans(
    df: pd.DataFrame,
    cluster_features: list = None,
    k: int = None,
    random_state: int = 42,
) -> dict:
    """
    Phân cụm K-Means trên dữ liệu cây trồng.

    Parameters
    ----------
    df               : DataFrame đã qua preprocessing
    cluster_features : danh sách cột dùng để cluster (mặc định 4 cột khí hậu)
    k                : số cụm (None = tự tìm bằng Silhouette)

    Returns
    -------
    dict: {'labels', 'k', 'silhouette', 'profiles', 'scaler', 'model'}
    """
    if cluster_features is None:
        cluster_features = [
            "average_rain_fall_mm_per_year", "avg_temp",
            "pesticides_tonnes", "hg/ha_yield"
        ]

    X_c = df[cluster_features].copy()
    X_c["hg/ha_yield"] = X_c["hg/ha_yield"] / 1000  # scale down yield
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_c)

    if k is None:
        elbow = find_best_k(X_scaled)
        k = elbow["best_k"]

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    df = df.copy()
    df["cluster"] = labels

    # Tạo profile từng cluster
    profiles = []
    for c in range(k):
        sub = df[df.cluster == c]
        profiles.append({
            "cluster":    c,
            "n":          int(len(sub)),
            "yield_mean": float(sub["hg/ha_yield"].mean()),
            "rain_mean":  float(sub["average_rain_fall_mm_per_year"].mean()),
            "temp_mean":  float(sub["avg_temp"].mean()),
            "pest_mean":  float(sub["pesticides_tonnes"].mean()),
            "top_crops":  sub["Item"].value_counts().head(3).index.tolist(),
            "top_areas":  sub["Area"].value_counts().head(3).index.tolist(),
        })

    print(f"[miner] K-Means: K={k}, Silhouette={sil:.4f}")
    return {
        "labels":     labels,
        "k":          k,
        "silhouette": float(sil),
        "profiles":   profiles,
        "scaler":     scaler,
        "model":      km,
    }
