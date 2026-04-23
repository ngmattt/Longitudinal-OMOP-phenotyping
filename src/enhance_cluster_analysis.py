from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
FEATURES_CSV = PROJECT_FOLDER / "outputs" / "patient_features_wide.csv"
OUTPUT_DIR = PROJECT_FOLDER / "outputs"
STABILITY_RUNS = 25


def standardize_feature_table(feature_table):
    """Standardize feature columns before distance-based analysis."""
    feature_names = []
    for column_name in feature_table.columns:
        if column_name not in {"person_id", "cluster_id", "cluster_label"}:
            feature_names.append(column_name)

    feature_matrix = feature_table[feature_names].to_numpy(dtype=float)
    means = np.mean(feature_matrix, axis=0)
    stds = np.std(feature_matrix, axis=0)
    stds[stds == 0] = 1.0
    scaled_matrix = (feature_matrix - means) / stds
    return scaled_matrix, feature_names


def run_simple_kmeans(feature_matrix, n_clusters, random_seed=42):
    """A readable k-means implementation that matches the main script."""
    rng = np.random.default_rng(random_seed)
    starting_indices = rng.choice(len(feature_matrix), size=n_clusters, replace=False)
    centroids = feature_matrix[starting_indices].copy()

    for _iteration in range(50):
        all_distances = np.sqrt(
            ((feature_matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        )
        labels = all_distances.argmin(axis=1)

        new_centroids = []
        for cluster_id in range(n_clusters):
            cluster_members = feature_matrix[labels == cluster_id]
            if len(cluster_members) == 0:
                random_index = rng.integers(0, len(feature_matrix))
                new_centroids.append(feature_matrix[random_index])
            else:
                new_centroids.append(cluster_members.mean(axis=0))

        new_centroids = np.vstack(new_centroids)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


def adjusted_rand_index(reference_labels, new_labels):
    """Compare two cluster labelings."""
    contingency_table = pd.crosstab(reference_labels, new_labels).to_numpy(dtype=float)
    total_n = contingency_table.sum()

    if total_n <= 1:
        return 1.0

    def choose_two(values):
        return values * (values - 1) / 2.0

    index_value = choose_two(contingency_table).sum()
    row_totals = contingency_table.sum(axis=1)
    column_totals = contingency_table.sum(axis=0)

    row_index = choose_two(row_totals).sum()
    column_index = choose_two(column_totals).sum()
    total_pairs = choose_two(np.array([total_n]))[0]

    expected_index = 0.0
    if total_pairs != 0:
        expected_index = (row_index * column_index) / total_pairs

    max_index = 0.5 * (row_index + column_index)
    denominator = max_index - expected_index

    if denominator == 0:
        return 1.0

    return float((index_value - expected_index) / denominator)


def silhouette_score(feature_matrix, labels):
    """Compute a basic silhouette score."""
    unique_cluster_ids = np.unique(labels)
    if len(unique_cluster_ids) < 2:
        return 0.0

    distance_matrix = np.sqrt(
        ((feature_matrix[:, None, :] - feature_matrix[None, :, :]) ** 2).sum(axis=2)
    )

    all_scores = []
    for row_index in range(len(feature_matrix)):
        same_cluster_mask = labels == labels[row_index]
        same_cluster_mask[row_index] = False

        if same_cluster_mask.sum() == 0:
            within_cluster_distance = 0.0
        else:
            within_cluster_distance = distance_matrix[row_index, same_cluster_mask].mean()

        nearest_other_cluster_distance = np.inf
        for other_cluster_id in unique_cluster_ids:
            if other_cluster_id == labels[row_index]:
                continue

            other_cluster_mask = labels == other_cluster_id
            if other_cluster_mask.sum() == 0:
                continue

            candidate_distance = distance_matrix[row_index, other_cluster_mask].mean()
            nearest_other_cluster_distance = min(
                nearest_other_cluster_distance, candidate_distance
            )

        if not np.isfinite(nearest_other_cluster_distance):
            one_score = 0.0
        else:
            denominator = max(within_cluster_distance, nearest_other_cluster_distance)
            if denominator == 0:
                one_score = 0.0
            else:
                one_score = (
                    nearest_other_cluster_distance - within_cluster_distance
                ) / denominator

        all_scores.append(one_score)

    return float(np.mean(all_scores))


def compute_centroids(feature_matrix, labels):
    """Return one centroid per cluster."""
    centroid_rows = []
    for cluster_id in sorted(np.unique(labels)):
        cluster_members = feature_matrix[labels == cluster_id]
        centroid_rows.append(cluster_members.mean(axis=0))
    return np.vstack(centroid_rows)


def measure_cluster_compactness(feature_matrix, labels):
    """Describe how close members are to their cluster centroid."""
    compactness_rows = []

    for cluster_id in sorted(np.unique(labels)):
        cluster_members = feature_matrix[labels == cluster_id]
        cluster_centroid = cluster_members.mean(axis=0)
        distances = np.sqrt(((cluster_members - cluster_centroid) ** 2).sum(axis=1))

        compactness_rows.append(
            {
                "cluster_id": int(cluster_id),
                "patients": int(len(cluster_members)),
                "mean_distance_to_centroid": float(np.mean(distances)),
                "max_distance_to_centroid": float(np.max(distances)),
            }
        )

    return pd.DataFrame(compactness_rows)


def measure_feature_separation(feature_table, feature_names):
    """
    Measure how much each feature separates the clusters using eta-squared.
    """
    cluster_ids = feature_table["cluster_id"].to_numpy()
    unique_cluster_ids = sorted(feature_table["cluster_id"].unique())

    separation_rows = []
    for feature_name in feature_names:
        values = feature_table[feature_name].to_numpy(dtype=float)
        grand_mean = np.mean(values)
        total_sum_of_squares = np.sum((values - grand_mean) ** 2)

        between_cluster_sum_of_squares = 0.0
        for cluster_id in unique_cluster_ids:
            cluster_values = values[cluster_ids == cluster_id]
            cluster_mean = np.mean(cluster_values)
            between_cluster_sum_of_squares += len(cluster_values) * (
                cluster_mean - grand_mean
            ) ** 2

        if total_sum_of_squares == 0:
            eta_squared = 0.0
        else:
            eta_squared = between_cluster_sum_of_squares / total_sum_of_squares

        separation_rows.append(
            {"feature": feature_name, "eta_squared": float(eta_squared)}
        )

    separation_table = pd.DataFrame(separation_rows)
    separation_table = separation_table.sort_values(
        "eta_squared", ascending=False
    ).reset_index(drop=True)
    return separation_table


def rename_clusters_for_interpretation(feature_table):
    """Create human-readable names for each cluster."""
    cluster_summary = (
        feature_table.groupby("cluster_id")
        .agg(
            creatinine_mean=("creatinine_mean_value", "mean"),
            creatinine_delta=("creatinine_delta_value", "mean"),
            bun_mean=("urea_nitrogen_mean_value", "mean"),
        )
        .reset_index()
    )

    cluster_name_map = {}
    for row in cluster_summary.itertuples(index=False):
        if row.creatinine_mean >= 3 or row.bun_mean >= 45:
            cluster_name_map[row.cluster_id] = "severe-kidney-dysfunction"
        elif row.creatinine_delta > 0.2:
            cluster_name_map[row.cluster_id] = "worsening-kidney-trajectory"
        else:
            cluster_name_map[row.cluster_id] = "recovering-or-stable-kidney-trajectory"

    return cluster_name_map


def pick_exemplar_patients(feature_table, feature_matrix, labels):
    """
    Pick a few example patients near the center of each cluster.
    """
    unique_cluster_ids = sorted(np.unique(labels))
    centroid_matrix = compute_centroids(feature_matrix, labels)
    exemplar_rows = []

    for centroid_index, cluster_id in enumerate(unique_cluster_ids):
        cluster_mask = labels == cluster_id
        cluster_members = feature_matrix[cluster_mask]
        cluster_people = feature_table.loc[cluster_mask, "person_id"].to_numpy()
        centroid = centroid_matrix[centroid_index]

        distances = np.sqrt(((cluster_members - centroid) ** 2).sum(axis=1))
        sorted_indices = np.argsort(distances)

        for rank, member_index in enumerate(sorted_indices[:3], start=1):
            exemplar_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "person_id": int(cluster_people[member_index]),
                    "rank_within_cluster": rank,
                    "distance_to_centroid": float(distances[member_index]),
                }
            )

    return pd.DataFrame(exemplar_rows)


def calculate_pca_projection(feature_matrix):
    """Project the high-dimensional features into 2D using SVD-based PCA."""
    centered_matrix = feature_matrix - np.mean(feature_matrix, axis=0)
    _u, _s, vt = np.linalg.svd(centered_matrix, full_matrices=False)
    first_two_components = vt[:2]
    projected_matrix = centered_matrix @ first_two_components.T
    return projected_matrix


def create_embedding_svg(projection_table, output_path):
    """Make a lightweight SVG scatter plot."""
    width = 900
    height = 620
    margin = 70

    color_map = {
        "recovering-or-stable-kidney-trajectory": "#1f77b4",
        "severe-kidney-dysfunction": "#d62728",
        "worsening-kidney-trajectory": "#ff7f0e",
    }

    x_min = projection_table["pc1"].min()
    x_max = projection_table["pc1"].max()
    y_min = projection_table["pc2"].min()
    y_max = projection_table["pc2"].max()

    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    def scale_x(value: float) -> float:
        return margin + ((value - x_min) / x_range) * (width - 2 * margin)

    def scale_y(value: float) -> float:
        return height - margin - ((value - y_min) / y_range) * (height - 2 * margin)

    svg_lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Arial,sans-serif;font-size:12px;} .title{font-size:20px;font-weight:bold;} .axis{stroke:#555;stroke-width:1;} .label{font-size:13px;font-weight:bold;}</style>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<text x='30' y='30' class='title'>Patient Clusters In 2D Feature Space</text>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' class='axis' />",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' class='axis' />",
        f"<text x='{width / 2}' y='{height - 20}' class='label'>Principal Component 1</text>",
        f"<text x='15' y='{height / 2}' class='label' transform='rotate(-90 15 {height / 2})'>Principal Component 2</text>",
    ]

    for _, row in projection_table.iterrows():
        x_position = scale_x(row["pc1"])
        y_position = scale_y(row["pc2"])
        color = color_map[row["cluster_profile"]]
        svg_lines.append(
            f"<circle cx='{x_position:.1f}' cy='{y_position:.1f}' r='5' fill='{color}' fill-opacity='0.7' />"
        )

    legend_x = 530
    legend_y = 70
    for i, (label, color) in enumerate(color_map.items()):
        y_position = legend_y + i * 24
        svg_lines.append(
            f"<rect x='{legend_x}' y='{y_position - 10}' width='14' height='14' fill='{color}' />"
        )
        svg_lines.append(
            f"<text x='{legend_x + 22}' y='{y_position + 1}'>{label}</text>"
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_enhanced_report(
    validation_table,
    compactness_table,
    top_feature_table,
    profile_counts,
    output_path,
):
    """Write a plain-language summary of the validation step."""
    lines = [
        "# Enhanced Project 1 Report",
        "",
        "## Why This Second Report Exists",
        "",
        "The first script built the clusters.",
        "This second script asks whether the clusters are reasonably stable and which features explain them best.",
        "",
        "## Validation Summary",
        "",
    ]

    for row in validation_table.itertuples(index=False):
        lines.append(f"- `{row.metric}`: {row.value}")

    lines.append("")
    lines.append("## Cluster Sizes By Interpretable Profile")
    lines.append("")

    for profile_name, patient_count in profile_counts.items():
        lines.append(f"- `{profile_name}`: {int(patient_count)} patients")

    lines.append("")
    lines.append("## Cluster Compactness")
    lines.append("")
    lines.append(compactness_table.to_csv(index=False))
    lines.append("")
    lines.append("## Top Separating Features")
    lines.append("")

    for row in top_feature_table.itertuples(index=False):
        lines.append(f"- `{row.feature}`: eta-squared={row.eta_squared:.3f}")

    lines.extend(
        [
            "",
            "## Beginner Interpretation",
            "",
            "A good clustering result should usually show two things:",
            "1. patients inside one cluster are relatively similar to each other",
            "2. the solution is fairly stable when you re-run the algorithm with different random starts",
            "",
            "This report gives you both of those checks in a simple form.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading patient feature table...")
    feature_table = pd.read_csv(FEATURES_CSV)

    print("Step 2: Standardizing numeric features...")
    scaled_matrix, feature_names = standardize_feature_table(feature_table)

    original_cluster_labels = feature_table["cluster_id"].to_numpy(dtype=int)
    number_of_clusters = len(np.unique(original_cluster_labels))

    print("Step 3: Calculating silhouette score...")
    silhouette = silhouette_score(scaled_matrix, original_cluster_labels)

    print("Step 4: Re-running clustering several times for a stability check...")
    ari_scores = []
    for seed_value in range(STABILITY_RUNS):
        rerun_labels = run_simple_kmeans(
            scaled_matrix, n_clusters=number_of_clusters, random_seed=seed_value + 1
        )
        ari = adjusted_rand_index(original_cluster_labels, rerun_labels)
        ari_scores.append(ari)

    validation_table = pd.DataFrame(
        [
            {"metric": "n_patients", "value": int(len(feature_table))},
            {"metric": "n_clusters", "value": int(number_of_clusters)},
            {"metric": "silhouette_score", "value": round(float(silhouette), 4)},
            {
                "metric": "mean_adjusted_rand_index",
                "value": round(float(np.mean(ari_scores)), 4),
            },
            {
                "metric": "min_adjusted_rand_index",
                "value": round(float(np.min(ari_scores)), 4),
            },
            {
                "metric": "max_adjusted_rand_index",
                "value": round(float(np.max(ari_scores)), 4),
            },
        ]
    )

    print("Step 5: Measuring cluster compactness and feature separation...")
    compactness_table = measure_cluster_compactness(
        scaled_matrix, original_cluster_labels
    )
    top_feature_table = measure_feature_separation(feature_table, feature_names).head(12)

    print("Step 6: Creating human-readable cluster names...")
    cluster_name_map = rename_clusters_for_interpretation(feature_table)
    feature_table["cluster_profile"] = feature_table["cluster_id"].map(cluster_name_map)
    compactness_table["cluster_profile"] = compactness_table["cluster_id"].map(
        cluster_name_map
    )

    print("Step 7: Picking exemplar patients and 2D projection...")
    exemplar_table = pick_exemplar_patients(
        feature_table, scaled_matrix, original_cluster_labels
    )
    exemplar_table["cluster_profile"] = exemplar_table["cluster_id"].map(cluster_name_map)

    projected_matrix = calculate_pca_projection(scaled_matrix)
    projection_table = pd.DataFrame(
        {
            "person_id": feature_table["person_id"],
            "cluster_id": feature_table["cluster_id"],
            "cluster_profile": feature_table["cluster_profile"],
            "pc1": projected_matrix[:, 0],
            "pc2": projected_matrix[:, 1],
        }
    )

    print("Step 8: Saving validation outputs...")
    create_embedding_svg(projection_table, output_dir / "cluster_embedding.svg")
    validation_table.to_csv(output_dir / "validation_summary.csv", index=False)
    compactness_table.to_csv(output_dir / "cluster_compactness.csv", index=False)
    top_feature_table.to_csv(output_dir / "top_cluster_features.csv", index=False)
    exemplar_table.to_csv(output_dir / "cluster_exemplars.csv", index=False)
    projection_table.to_csv(output_dir / "cluster_projection.csv", index=False)

    write_enhanced_report(
        validation_table=validation_table,
        compactness_table=compactness_table,
        top_feature_table=top_feature_table,
        profile_counts=feature_table["cluster_profile"].value_counts().sort_index(),
        output_path=output_dir / "enhanced_project_report.md",
    )

    print(f"Finished. Enhanced outputs were written to: {output_dir}")


if __name__ == "__main__":
    main()
