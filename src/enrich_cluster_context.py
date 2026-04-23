from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
CLUSTERS_CSV = PROJECT_FOLDER / "outputs" / "patient_clusters.csv"
LABS_CSV = PROJECT_FOLDER / "outputs" / "longitudinal_labs.csv"
PERSON_CSV = Path(r"C:\Users\Matthew Ng\AppData\Local\Temp\procedure_occurrence.csv")
CONDITION_CSV = Path(r"C:\Users\Matthew Ng\AppData\Local\Temp\cost.csv")
OUTPUT_DIR = PROJECT_FOLDER / "outputs"


def mode_or_unknown(values):
    """Return the most common non-empty value, or 'Unknown'."""
    cleaned_values = values.fillna("").astype(str).str.strip()
    cleaned_values = cleaned_values[cleaned_values != ""]
    if cleaned_values.empty:
        return "Unknown"
    return str(cleaned_values.mode().iloc[0])


def dataframe_to_markdown(dataframe):
    """Small helper for writing markdown tables."""
    headers = list(dataframe.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in dataframe.iterrows():
        row_values = []
        for header in headers:
            row_values.append(str(row[header]))
        lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join(lines)


def build_patient_context_table(cluster_table, lab_table, person_table):
    """
    Add age and demographic information to each clustered patient.
    """
    first_measurement_table = (
        lab_table.groupby("person_id")["event_time"]
        .min()
        .reset_index()
        .rename(columns={"event_time": "first_measurement_time"})
    )
    first_measurement_table["first_measurement_time"] = pd.to_datetime(
        first_measurement_table["first_measurement_time"], errors="coerce"
    )
    first_measurement_table["first_measurement_year"] = (
        first_measurement_table["first_measurement_time"].dt.year
    )

    person_columns_to_keep = [
        "person_id",
        "year_of_birth",
        "gender_source_value",
        "race_source_value",
        "ethnicity_source_value",
    ]
    person_table = person_table[person_columns_to_keep].copy()

    patient_context = cluster_table.merge(
        first_measurement_table, on="person_id", how="left"
    )
    patient_context = patient_context.merge(person_table, on="person_id", how="left")

    patient_context["year_of_birth"] = pd.to_numeric(
        patient_context["year_of_birth"], errors="coerce"
    )
    patient_context["age_at_first_measurement"] = (
        patient_context["first_measurement_year"] - patient_context["year_of_birth"]
    )

    patient_context["gender_source_value"] = patient_context["gender_source_value"].fillna(
        "Unknown"
    )
    patient_context["race_source_value"] = patient_context["race_source_value"].fillna(
        "Unknown"
    )
    patient_context["ethnicity_source_value"] = patient_context[
        "ethnicity_source_value"
    ].fillna("Unknown")

    return patient_context


def summarize_demographics_by_cluster(patient_context):
    """Create one summary row per cluster."""
    summary_rows = []

    for (cluster_id, cluster_label), cluster_group in patient_context.groupby(
        ["cluster_id", "cluster_label"]
    ):
        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": cluster_label,
                "patients": int(len(cluster_group)),
                "mean_age_at_first_measurement": round(
                    float(cluster_group["age_at_first_measurement"].mean()), 1
                ),
                "median_age_at_first_measurement": round(
                    float(cluster_group["age_at_first_measurement"].median()), 1
                ),
                "most_common_sex": mode_or_unknown(cluster_group["gender_source_value"]),
                "most_common_race": mode_or_unknown(cluster_group["race_source_value"]),
                "most_common_ethnicity": mode_or_unknown(
                    cluster_group["ethnicity_source_value"]
                ),
            }
        )

    summary_table = pd.DataFrame(summary_rows)
    summary_table = summary_table.sort_values("cluster_id").reset_index(drop=True)
    return summary_table


def get_condition_display_map():
    """Map a few common source labels and ICD codes to nicer display text."""
    return {
        "SR (Sinus Rhythm)": "Sinus rhythm",
        "ST (Sinus Tachycardia)": "Sinus tachycardia",
        "SB (Sinus Bradycardia)": "Sinus bradycardia",
        "AF (Atrial Fibrillation)": "Atrial fibrillation",
        "A Paced": "Atrial paced rhythm",
        "V Paced": "Ventricular paced rhythm",
        "1st AV (First degree AV Block)": "First-degree AV block",
        "4019": "Hypertension (ICD-9 401.9)",
        "I10": "Essential hypertension (ICD-10 I10)",
        "2724": "Hyperlipidemia (ICD-9 272.4)",
        "E785": "Hyperlipidemia (ICD-10 E78.5)",
        "42731": "Atrial fibrillation (ICD-9 427.31)",
        "I4891": "Atrial fibrillation (ICD-10 I48.91)",
        "5849": "Acute kidney failure, unspecified (ICD-9 584.9)",
        "N179": "Acute kidney failure, unspecified (ICD-10 N17.9)",
        "5856": "End stage renal disease (ICD-9 585.6)",
        "N186": "End stage renal disease (ICD-10 N18.6)",
    }


def summarize_top_conditions_by_cluster(patient_context, condition_table, top_n=8):
    """
    Find the most common diagnosis source labels inside each cluster.
    """
    cluster_people = patient_context[
        ["person_id", "cluster_id", "cluster_label"]
    ].drop_duplicates()

    merged_condition_table = condition_table.merge(
        cluster_people, on="person_id", how="inner"
    ).copy()
    merged_condition_table["condition_source_value"] = (
        merged_condition_table["condition_source_value"].fillna("").astype(str).str.strip()
    )

    grouped_counts = (
        merged_condition_table.groupby(
            ["cluster_id", "cluster_label", "condition_concept_id", "condition_source_value"]
        )
        .size()
        .reset_index(name="occurrence_count")
        .sort_values(["cluster_id", "occurrence_count"], ascending=[True, False])
    )

    top_condition_rows = grouped_counts.groupby("cluster_id").head(top_n).copy()

    display_map = get_condition_display_map()
    top_condition_rows["condition_display"] = top_condition_rows[
        "condition_source_value"
    ].map(display_map).fillna(top_condition_rows["condition_source_value"])

    return top_condition_rows.reset_index(drop=True)


def write_application_summary(
    demographic_summary,
    top_conditions,
    output_path,
):
    """Write a report that is easy to reuse in interviews and applications."""
    lines = [
        "# Application-Focused Project Summary",
        "",
        "## What This Adds",
        "",
        "This report adds patient context to the kidney trajectory clusters.",
        "That matters because phenotype clusters are much easier to explain when you can describe who is in each group and what clinical conditions appear around them.",
        "",
        "## Demographic Summary By Cluster",
        "",
        dataframe_to_markdown(demographic_summary),
        "",
        "## Top Diagnosis Signals By Cluster",
        "",
    ]

    for (cluster_id, cluster_label), cluster_group in top_conditions.groupby(
        ["cluster_id", "cluster_label"]
    ):
        lines.append(f"### Cluster {cluster_id}: {cluster_label}")
        for row in cluster_group.itertuples(index=False):
            lines.append(f"- `{row.condition_display}`: {int(row.occurrence_count)} condition rows")
        lines.append("")

    lines.extend(
        [
            "## How To Explain This In An Interview",
            "",
            "I started with raw OMOP repeated lab measurements, engineered patient-level trajectory features, and used unsupervised clustering to identify kidney-related subgroups.",
            "I then validated cluster stability, examined which features drove separation, and added demographic and diagnosis context from OMOP person and condition tables to make the phenotypes interpretable.",
            "",
            "## Honest Limitation",
            "",
            "This is a de-identified ICU demo-style dataset, so the trajectories are short-term inpatient patterns rather than long-term outpatient disease progression.",
            "That limitation is fine for a portfolio project as long as you say it clearly.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading cluster results and OMOP context tables...")
    cluster_table = pd.read_csv(CLUSTERS_CSV)
    lab_table = pd.read_csv(LABS_CSV)
    person_table = pd.read_csv(PERSON_CSV)
    condition_table = pd.read_csv(
        CONDITION_CSV,
        usecols=["person_id", "condition_concept_id", "condition_source_value"],
    )

    lab_table["event_time"] = pd.to_datetime(lab_table["event_time"], errors="coerce")

    print("Step 2: Building patient-level context table...")
    patient_context = build_patient_context_table(
        cluster_table=cluster_table,
        lab_table=lab_table,
        person_table=person_table,
    )

    print("Step 3: Summarizing demographics by cluster...")
    demographic_summary = summarize_demographics_by_cluster(patient_context)

    print("Step 4: Finding top diagnosis signals in each cluster...")
    top_conditions = summarize_top_conditions_by_cluster(
        patient_context=patient_context,
        condition_table=condition_table,
        top_n=8,
    )

    print("Step 5: Saving enriched outputs...")
    patient_context.to_csv(output_dir / "patient_cluster_context.csv", index=False)
    demographic_summary.to_csv(
        output_dir / "demographic_summary_by_cluster.csv", index=False
    )
    top_conditions.to_csv(output_dir / "top_conditions_by_cluster.csv", index=False)

    write_application_summary(
        demographic_summary=demographic_summary,
        top_conditions=top_conditions,
        output_path=output_dir / "application_project_summary.md",
    )

    print(f"Finished. Enriched context outputs were written to: {output_dir}")


if __name__ == "__main__":
    main()
