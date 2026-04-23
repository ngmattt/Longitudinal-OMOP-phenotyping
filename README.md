# OMOP Kidney Trajectory Phenotyping

This project uses longitudinal OMOP clinical data to discover patient subgroups based on repeated kidney-related laboratory measurements.

The core idea is simple: instead of looking at one lab value at one time point, this project asks whether patients can be grouped by how their labs change over time.

## Research Question

Can repeated kidney-related laboratory measurements from OMOP clinical data be used to identify distinct patient trajectory groups?

More specifically:

- can repeated creatinine, urea nitrogen, sodium, potassium, and chloride measurements be turned into patient-level trajectory features?
- can those features be used to discover subgroups with unsupervised learning?
- are the resulting groups stable and clinically interpretable?

## Project Summary

This repository contains a beginner-friendly longitudinal phenotyping workflow with three stages:

1. `omop_kidney_trajectory_pipeline.py`
   Builds trajectory features from repeated measurements and clusters patients into subgroups.
2. `enhance_cluster_analysis.py`
   Evaluates cluster stability and identifies the features that drive separation.
3. `enrich_cluster_context.py`
   Adds demographic and diagnosis context from OMOP `person` and `condition_occurrence` tables.

Together, these scripts create a small end-to-end phenotype discovery project from OMOP data.

## What The Project Does

The workflow:

1. loads repeated laboratory measurements from the OMOP `measurement` table
2. keeps five kidney/electrolyte-related labs
3. summarizes each patient's longitudinal pattern using features such as:
   - first value
   - last value
   - change over time
   - mean
   - standard deviation
   - slope
   - abnormal fraction
4. applies k-means clustering to discover patient subgroups
5. evaluates cluster quality and stability
6. adds age, sex, race, ethnicity, and diagnosis context
7. writes figures, tables, and markdown summaries for interpretation

## Measurements Used

The project focuses on these OMOP `measurement_source_value` codes:

- `50912`: Creatinine
- `51006`: Urea Nitrogen
- `50983`: Sodium
- `50971`: Potassium
- `50902`: Chloride

These were chosen because they were repeated frequently in the available dataset and form a clinically coherent kidney/electrolyte phenotype story.

## Why This Uses Unsupervised Learning

The project uses unsupervised learning in the clustering step.

Each patient is converted into a row of trajectory features. For example:

- creatinine mean
- creatinine slope
- urea nitrogen maximum
- sodium variability
- potassium abnormal fraction

Then k-means clustering groups similar patients together without being told the correct labels ahead of time.

That means the patient subgroups are discovered from the data rather than predefined manually.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- src
|   |-- omop_kidney_trajectory_pipeline.py
|   |-- enhance_cluster_analysis.py
|   |-- enrich_cluster_context.py
|-- outputs
    |-- measurement_summary.csv
    |-- patient_features_wide.csv
    |-- patient_clusters.csv
    |-- cluster_summary.csv
    |-- validation_summary.csv
    |-- top_cluster_features.csv
    |-- demographic_summary_by_cluster.csv
    |-- top_conditions_by_cluster.csv
    |-- cluster_mean_trajectories.svg
    |-- cluster_embedding.svg
    |-- project_report.md
    |-- enhanced_project_report.md
    |-- application_project_summary.md
```

## How To Run

The scripts were written in a beginner-friendly style and use simple config variables at the top of each file instead of command-line arguments.

Before running:

1. open each script in `src`
2. check the file paths near the top
3. update them if your OMOP CSVs are in a different location

Then run the scripts in this order:

```powershell
python .\src\omop_kidney_trajectory_pipeline.py
python .\src\enhance_cluster_analysis.py
python .\src\enrich_cluster_context.py
```

If `python` is not on your path, use your local Python executable instead.

## Input Data Notes

This project was built on a local OMOP CSV export.

Important details:

- `metadata.csv` was treated as the OMOP `measurement` table
- `procedure_occurrence.csv` was treated as the OMOP `person` table
- `cost.csv` was treated as the OMOP `condition_occurrence` table

This naming mismatch came from the local export, so you may need to adjust file paths if your OMOP tables are named normally.

## Main Outputs

Useful output files include:

- `outputs/measurement_summary.csv`
- `outputs/patient_features_wide.csv`
- `outputs/patient_clusters.csv`
- `outputs/cluster_summary.csv`
- `outputs/validation_summary.csv`
- `outputs/top_cluster_features.csv`
- `outputs/demographic_summary_by_cluster.csv`
- `outputs/top_conditions_by_cluster.csv`
- `outputs/cluster_mean_trajectories.svg`
- `outputs/cluster_embedding.svg`
- `outputs/project_report.md`
- `outputs/enhanced_project_report.md`
- `outputs/application_project_summary.md`

## Limitations

- this is a de-identified ICU-style dataset, so the trajectories are short-term inpatient patterns rather than long-term outpatient disease progression
- the clustering is a simple first-pass unsupervised learning approach, not a definitive clinical subtype model
- some diagnosis labels remain source-coded because of the structure of the local OMOP export
