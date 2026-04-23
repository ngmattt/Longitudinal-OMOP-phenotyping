import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MEASUREMENT_CSV = Path(r"C:\Users\Matthew Ng\AppData\Local\Temp\metadata.csv")
OUTPUT_DIR = PROJECT_FOLDER / "outputs"
N_CLUSTERS = 3


# I am keeping the measurement list near the top because it is one of the first
# things a reader usually wants to know: which labs are included in the project?
TARGET_MEASUREMENTS = {
    "50912": "Creatinine",
    "51006": "Urea Nitrogen",
    "50983": "Sodium",
    "50971": "Potassium",
    "50902": "Chloride",
}


def load_measurement_data(measurement_csv):
    """
    Load the measurement table and keep only the columns needed for this project.

    The original CSV has many columns, but for a first project we only need:
    - patient ID
    - date/time of the measurement
    - which lab test it was
    - the numeric result
    - the units
    """
    measurements = pd.read_csv(measurement_csv, dtype={"measurement_source_value": str})

    columns_to_keep = [
        "person_id",
        "measurement_date",
        "measurement_datetime",
        "measurement_source_value",
        "value_as_number",
        "unit_source_value",
    ]
    measurements = measurements[columns_to_keep].copy()

    measurements["measurement_source_value"] = measurements["measurement_source_value"].fillna("")
    measurements = measurements[
        measurements["measurement_source_value"].isin(TARGET_MEASUREMENTS.keys())
    ].copy()

    measurements["measurement_name"] = measurements["measurement_source_value"].map(
        TARGET_MEASUREMENTS
    )

    # Convert text dates into actual datetime objects.
    measurements["measurement_datetime"] = pd.to_datetime(
        measurements["measurement_datetime"], errors="coerce"
    )
    measurements["measurement_date"] = pd.to_datetime(
        measurements["measurement_date"], errors="coerce"
    )

    # Some rows have a full datetime and some only have a date, so we combine them.
    measurements["event_time"] = measurements["measurement_datetime"].fillna(
        measurements["measurement_date"]
    )

    measurements["value_as_number"] = pd.to_numeric(
        measurements["value_as_number"], errors="coerce"
    )

    measurements = measurements.dropna(
        subset=["person_id", "event_time", "value_as_number"]
    ).copy()

    measurements = measurements.sort_values(
        ["person_id", "measurement_name", "event_time"]
    ).reset_index(drop=True)

    return measurements


def add_time_from_first_measurement(measurements):
    """
    For each patient, calculate how many hours have passed since their first
    measurement in the dataset.
    """
    measurements = measurements.copy()
    first_event_per_patient = measurements.groupby("person_id")["event_time"].transform("min")
    measurements["hours_since_first_measurement"] = (
        measurements["event_time"] - first_event_per_patient
    ).dt.total_seconds() / 3600.0
    return measurements


def find_most_common_unit(values):
    """Return the most common non-empty unit string."""
    cleaned_values = values.dropna().astype(str)
    cleaned_values = cleaned_values[cleaned_values != ""]
    if cleaned_values.empty:
        return ""
    return str(cleaned_values.mode().iloc[0])


def build_measurement_summary(measurements):
    """
    Summarize how much data we have for each lab measurement.

    This is useful because it tells us whether a measurement has enough repeated
    observations to support a trajectory project.
    """
    summary_rows = []

    for measurement_name, measurement_group in measurements.groupby("measurement_name"):
        per_patient_counts = measurement_group.groupby("person_id").size()

        time_span_table = (
            measurement_group.groupby("person_id")["event_time"]
            .agg(["min", "max"])
            .reset_index()
        )
        time_span_table["span_hours"] = (
            time_span_table["max"] - time_span_table["min"]
        ).dt.total_seconds() / 3600.0

        one_row = {
            "measurement_name": measurement_name,
            "measurement_source_value": measurement_group["measurement_source_value"].iloc[0],
            "rows": int(len(measurement_group)),
            "unique_patients": int(measurement_group["person_id"].nunique()),
            "median_measurements_per_patient": float(per_patient_counts.median()),
            "median_time_span_hours": float(time_span_table["span_hours"].median()),
            "common_unit": find_most_common_unit(measurement_group["unit_source_value"]),
        }
        summary_rows.append(one_row)

    summary_table = pd.DataFrame(summary_rows)
    summary_table = summary_table.sort_values("rows", ascending=False).reset_index(drop=True)
    return summary_table


def calculate_slope(time_values, lab_values):
    """
    Estimate a simple linear slope.

    This is not meant to be a perfect model. It is just a simple summary of the
    overall trend for one patient's lab values over time.
    """
    if len(time_values) < 2:
        return 0.0

    centered_time = time_values - np.mean(time_values)
    centered_lab = lab_values - np.mean(lab_values)
    denominator = np.sum(centered_time ** 2)

    if denominator == 0:
        return 0.0

    slope = np.sum(centered_time * centered_lab) / denominator
    return float(slope)


def calculate_abnormal_fraction(measurement_name, lab_values):
    """
    Count the fraction of values outside a simple reference range.

    These are rough educational thresholds, not clinical-grade decision rules.
    """
    normal_ranges = {
        "Creatinine": (0.6, 1.3),
        "Urea Nitrogen": (7.0, 20.0),
        "Sodium": (135.0, 145.0),
        "Potassium": (3.5, 5.1),
        "Chloride": (98.0, 107.0),
    }

    low_value, high_value = normal_ranges[measurement_name]
    is_abnormal = (lab_values < low_value) | (lab_values > high_value)
    return float(is_abnormal.mean())


def build_feature_table_for_each_patient(measurements):
    """
    Convert repeated lab rows into patient-level trajectory features.

    The long table has one row per patient per measurement.
    The wide table has one row per patient total.
    """
    long_feature_rows = []

    grouped = measurements.groupby(["person_id", "measurement_name"])
    for (person_id, measurement_name), patient_measurement_group in grouped:
        patient_measurement_group = patient_measurement_group.sort_values("event_time")

        lab_values = patient_measurement_group["value_as_number"].to_numpy(dtype=float)
        time_values = patient_measurement_group["hours_since_first_measurement"].to_numpy(
            dtype=float
        )

        feature_row = {
            "person_id": person_id,
            "measurement_name": measurement_name,
            "measurement_count": int(len(patient_measurement_group)),
            "first_value": float(lab_values[0]),
            "last_value": float(lab_values[-1]),
            "delta_value": float(lab_values[-1] - lab_values[0]),
            "min_value": float(np.min(lab_values)),
            "max_value": float(np.max(lab_values)),
            "mean_value": float(np.mean(lab_values)),
            "std_value": float(np.std(lab_values)),
            "slope_per_hour": calculate_slope(time_values, lab_values),
            "time_span_hours": float(np.max(time_values) - np.min(time_values)),
            "abnormal_fraction": calculate_abnormal_fraction(
                measurement_name, patient_measurement_group["value_as_number"]
            ),
        }
        long_feature_rows.append(feature_row)

    long_feature_table = pd.DataFrame(long_feature_rows)

    wide_feature_table = long_feature_table.pivot(
        index="person_id", columns="measurement_name"
    )

    renamed_columns = []
    for feature_name, measurement_name in wide_feature_table.columns:
        clean_measurement_name = measurement_name.lower().replace(" ", "_")
        renamed_columns.append(f"{clean_measurement_name}_{feature_name}")

    wide_feature_table.columns = renamed_columns
    wide_feature_table = wide_feature_table.reset_index()

    # Fill missing values with the column median so every patient can remain in
    # the clustering step.
    for column_name in wide_feature_table.columns:
        if column_name == "person_id":
            continue
        wide_feature_table[column_name] = pd.to_numeric(
            wide_feature_table[column_name], errors="coerce"
        )
        median_value = wide_feature_table[column_name].median()
        wide_feature_table[column_name] = wide_feature_table[column_name].fillna(
            median_value
        )

    return long_feature_table, wide_feature_table


def standardize_feature_table(feature_table):
    """
    Standardize features so they are on a similar scale before clustering.
    """
    feature_columns = []
    for column_name in feature_table.columns:
        if column_name != "person_id":
            feature_columns.append(column_name)

    feature_matrix = feature_table[feature_columns].to_numpy(dtype=float)

    column_means = np.mean(feature_matrix, axis=0)
    column_stds = np.std(feature_matrix, axis=0)
    column_stds[column_stds == 0] = 1.0

    scaled_matrix = (feature_matrix - column_means) / column_stds
    return scaled_matrix, feature_columns


def run_simple_kmeans(feature_matrix, n_clusters, random_seed=42):
    """
    A small k-means implementation written in a direct, readable style.
    """
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
                replacement_centroid = feature_matrix[random_index]
                new_centroids.append(replacement_centroid)
            else:
                new_centroids.append(cluster_members.mean(axis=0))

        new_centroids = np.vstack(new_centroids)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels


def summarize_clusters(clustered_feature_table):
    """
    Create a short summary table describing each cluster.
    """
    cluster_summary = (
        clustered_feature_table.groupby("cluster_id")
        .agg(
            patients=("person_id", "count"),
            creatinine_mean_value_mean=("creatinine_mean_value", "mean"),
            creatinine_delta_value_mean=("creatinine_delta_value", "mean"),
            urea_nitrogen_mean_value_mean=("urea_nitrogen_mean_value", "mean"),
            sodium_mean_value_mean=("sodium_mean_value", "mean"),
            potassium_mean_value_mean=("potassium_mean_value", "mean"),
            chloride_mean_value_mean=("chloride_mean_value", "mean"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )
    return cluster_summary


def name_clusters(cluster_summary):
    """
    Give each cluster a simple label based on kidney-related features.
    """
    label_map = {}

    for row in cluster_summary.itertuples(index=False):
        if row.creatinine_mean_value_mean >= 1.8 or row.urea_nitrogen_mean_value_mean >= 35:
            label_map[row.cluster_id] = "higher-kidney-burden"
        elif row.creatinine_delta_value_mean < -0.05:
            label_map[row.cluster_id] = "improving-labs"
        else:
            label_map[row.cluster_id] = "lower-kidney-burden"

    return label_map


def create_svg_report(measurements, clustered_feature_table, output_path):
    """
    Draw a simple SVG figure that shows mean trajectories by cluster.

    I kept this as plain SVG text instead of using plotting packages because the
    runtime here is very minimal.
    """
    width = 1000
    height = 650
    margin_left = 70
    margin_right = 30
    margin_top = 40
    row_height = 110
    plot_width = width - margin_left - margin_right

    measurement_names = list(TARGET_MEASUREMENTS.values())
    cluster_ids = sorted(clustered_feature_table["cluster_id"].unique())

    color_palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    cluster_colors = {}
    for i, cluster_id in enumerate(cluster_ids):
        cluster_colors[cluster_id] = color_palette[i % len(color_palette)]

    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Arial, sans-serif;font-size:12px;} .title{font-size:20px;font-weight:bold;} .axis{stroke:#555;stroke-width:1;} .grid{stroke:#ddd;stroke-width:1;} .label{font-size:13px;font-weight:bold;}</style>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<text x='30' y='28' class='title'>Cluster Mean Trajectories</text>",
    ]

    for row_number, measurement_name in enumerate(measurement_names):
        y_top = margin_top + row_number * row_height
        y_bottom = y_top + 70

        measurement_rows = measurements[measurements["measurement_name"] == measurement_name]
        merged_rows = measurement_rows.merge(
            clustered_feature_table[["person_id", "cluster_id"]],
            on="person_id",
            how="inner",
        )

        if merged_rows.empty:
            continue

        max_hours = max(merged_rows["hours_since_first_measurement"].max(), 1.0)
        min_value = merged_rows["value_as_number"].min()
        max_value = merged_rows["value_as_number"].max()
        if math.isclose(min_value, max_value):
            max_value = min_value + 1.0

        svg_parts.append(
            f"<text x='30' y='{y_top + 15}' class='label'>{measurement_name}</text>"
        )
        svg_parts.append(
            f"<line x1='{margin_left}' y1='{y_bottom}' x2='{margin_left + plot_width}' y2='{y_bottom}' class='axis' />"
        )
        svg_parts.append(
            f"<line x1='{margin_left}' y1='{y_top}' x2='{margin_left}' y2='{y_bottom}' class='axis' />"
        )

        for fraction in [0.25, 0.5, 0.75]:
            x_position = margin_left + plot_width * fraction
            svg_parts.append(
                f"<line x1='{x_position:.1f}' y1='{y_top}' x2='{x_position:.1f}' y2='{y_bottom}' class='grid' />"
            )

        for cluster_id in cluster_ids:
            cluster_rows = merged_rows[merged_rows["cluster_id"] == cluster_id].copy()
            if cluster_rows.empty:
                continue

            cluster_rows["hour_bin"] = (
                cluster_rows["hours_since_first_measurement"] / 12.0
            ).round() * 12.0

            mean_curve = (
                cluster_rows.groupby("hour_bin")["value_as_number"]
                .mean()
                .reset_index()
                .sort_values("hour_bin")
            )

            point_strings = []
            for curve_row in mean_curve.itertuples(index=False):
                x_position = margin_left + (curve_row.hour_bin / max_hours) * plot_width
                y_fraction = (curve_row.value_as_number - min_value) / (max_value - min_value)
                y_position = y_bottom - y_fraction * (y_bottom - y_top)
                point_strings.append(f"{x_position:.1f},{y_position:.1f}")

            if len(point_strings) >= 2:
                joined_points = " ".join(point_strings)
                svg_parts.append(
                    f"<polyline fill='none' stroke='{cluster_colors[cluster_id]}' stroke-width='2' points='{joined_points}' />"
                )

        common_unit = find_most_common_unit(merged_rows["unit_source_value"])
        svg_parts.append(
            f"<text x='{margin_left + plot_width + 5}' y='{y_top + 12}'>{common_unit}</text>"
        )

    legend_y = height - 25
    legend_x = 40
    for cluster_id in cluster_ids:
        svg_parts.append(
            f"<line x1='{legend_x}' y1='{legend_y}' x2='{legend_x + 18}' y2='{legend_y}' stroke='{cluster_colors[cluster_id]}' stroke-width='3' />"
        )
        svg_parts.append(
            f"<text x='{legend_x + 24}' y='{legend_y + 4}'>Cluster {cluster_id}</text>"
        )
        legend_x += 120

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def dataframe_to_markdown(dataframe):
    """A small helper to turn a DataFrame into a markdown table."""
    headers = list(dataframe.columns)
    markdown_lines = []
    markdown_lines.append("| " + " | ".join(headers) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in dataframe.iterrows():
        row_values = []
        for header in headers:
            row_values.append(str(row[header]))
        markdown_lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join(markdown_lines)


def write_project_report(
    measurement_summary,
    clustered_feature_table,
    cluster_summary,
    output_path,
):
    """
    Write a plain-language report that explains the project outputs.
    """
    cluster_sizes = clustered_feature_table["cluster_label"].value_counts().sort_index()

    lines = [
        "# OMOP Kidney Trajectory Project Report",
        "",
        "## What The Script Did",
        "",
        "1. Loaded the OMOP measurement export from `metadata.csv`.",
        "2. Kept five repeated lab measurements: Creatinine, Urea Nitrogen, Sodium, Potassium, and Chloride.",
        "3. Built patient-level trajectory features such as baseline, final value, change over time, spread, slope, and abnormal fraction.",
        "4. Grouped patients into simple clusters using a lightweight k-means implementation.",
        "",
        "## Measurement Coverage",
        "",
        dataframe_to_markdown(measurement_summary),
        "",
        "## Cluster Sizes",
        "",
    ]

    for cluster_label, count in cluster_sizes.items():
        lines.append(f"- `{cluster_label}`: {int(count)} patients")

    lines.extend(
        [
            "",
            "## Cluster Summary",
            "",
            dataframe_to_markdown(cluster_summary),
            "",
            "## Beginner Interpretation",
            "",
            "A trajectory feature is just a summary of how a measurement changes over time for one patient.",
            "For example, if creatinine starts high and stays high, that patient may land in a higher-burden cluster.",
            "If creatinine drops over time, that patient may land in a more improving cluster.",
            "",
            "This is not a finished clinical model. It is a portfolio project that shows you can transform raw OMOP repeated measurements into an analysis-ready phenotype pipeline.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_main_outputs(
    output_dir,
    measurement_summary,
    measurements,
    long_feature_table,
    clustered_feature_table,
    cluster_summary,
):
    """Save the core output files."""
    measurement_summary.to_csv(output_dir / "measurement_summary.csv", index=False)
    measurements.to_csv(output_dir / "longitudinal_labs.csv", index=False)
    long_feature_table.to_csv(output_dir / "patient_features_long.csv", index=False)
    clustered_feature_table.to_csv(output_dir / "patient_clusters.csv", index=False)
    clustered_feature_table.to_csv(output_dir / "patient_features_wide.csv", index=False)
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading measurement data...")
    measurements = load_measurement_data(str(MEASUREMENT_CSV))

    print("Step 2: Calculating time from first measurement for each patient...")
    measurements = add_time_from_first_measurement(measurements)

    print("Step 3: Summarizing measurement coverage...")
    measurement_summary = build_measurement_summary(measurements)

    print("Step 4: Building patient-level trajectory features...")
    long_feature_table, wide_feature_table = build_feature_table_for_each_patient(
        measurements
    )

    print("Step 5: Standardizing features before clustering...")
    scaled_matrix, _feature_names = standardize_feature_table(wide_feature_table)

    print("Step 6: Running k-means clustering...")
    cluster_labels = run_simple_kmeans(
        scaled_matrix, n_clusters=N_CLUSTERS, random_seed=42
    )

    clustered_feature_table = wide_feature_table.copy()
    clustered_feature_table["cluster_id"] = cluster_labels

    print("Step 7: Summarizing and naming clusters...")
    cluster_summary = summarize_clusters(clustered_feature_table)
    cluster_name_map = name_clusters(cluster_summary)
    clustered_feature_table["cluster_label"] = clustered_feature_table["cluster_id"].map(
        cluster_name_map
    )
    cluster_summary["cluster_label"] = cluster_summary["cluster_id"].map(cluster_name_map)

    print("Step 8: Saving tables and reports...")
    save_main_outputs(
        output_dir=output_dir,
        measurement_summary=measurement_summary,
        measurements=measurements,
        long_feature_table=long_feature_table,
        clustered_feature_table=clustered_feature_table,
        cluster_summary=cluster_summary,
    )

    create_svg_report(
        measurements=measurements,
        clustered_feature_table=clustered_feature_table,
        output_path=output_dir / "cluster_mean_trajectories.svg",
    )

    write_project_report(
        measurement_summary=measurement_summary,
        clustered_feature_table=clustered_feature_table,
        cluster_summary=cluster_summary,
        output_path=output_dir / "project_report.md",
    )

    print(f"Finished. Outputs were written to: {output_dir}")


if __name__ == "__main__":
    main()
