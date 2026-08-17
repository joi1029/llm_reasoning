# llm_reasoning

This repository contains the analysis code, data-processing scripts, and supporting files for the paper: AI Reasoning Traces Can Undermine Human Verification of AI Responses

## Project overview

Main analysis code are in Python (version 3.11.9) and R version 4.4.1 (2024-06-14 ucrt).

## Repository structure

- `read_reasoning_data.py` — survey cleaning script
- `filter_demographics_to_cleaned.py` — filters Prolific demographic exports to the final cleaned sample
- `analyze_demographics.py` — descriptive demographic comparisons by experimental condition
- `analysis_bayesianSDT.rmd` — Bayesian SDT analysis report in R Markdown
- `data/` — raw and processed survey and demographic CSV files
- `models/` — saved fitted model objects (`.RData`) from the analyses
- `cleaned_llm_answers.csv` and `cleaned_llm_reasoning.csv` — cleaned participant-level outputs
- `df_final_llm_combined_long.csv` — combined long-form dataset used in downstream analysis

## Data pipeline


1. Load raw Qualtrics export data.
2. Dataframe cleaning (Restrict to respondents with sufficient progress and valid Prolific IDs, rename dataframe). (`read_reasoning_data.py`)
3. Data analysis and regression. (`analysis_bayesian_SDT.rmd`)


## Script descriptions

### `read_reasoning_data.py`

This is the primary data-cleaning script. It reads the raw survey export and performs the following tasks:

- removes spreadsheet-style header rows mixed into the data
- keeps only participants with adequate progress and valid Prolific IDs
- renames survey items into domain-specific features such as `math_*`, `legal_*`, `finance_*`, `medical_*`, and `misinfo_*`
- converts judgment variables into consistent boolean or numeric types
- reverses coding for confidence, knowledge, and trust scales so higher values reflect more positive perception
- creates derived metrics such as `accuracy`, `correct_answers`, and `avg_confidence`
- returns a cleaned DataFrame for analysis

This script is the starting point for most modeling work in the repository.

### `filter_demographics_to_cleaned.py`

This script filters the Prolific demographic export to the subset of participants retained in the cleaned survey data. It ensures that the final demographic sample matches the analytic sample used for the study.

It writes outputs:

- `data/filtered_demographics_answer.csv`
- `data/filtered_demographics_reasoning.csv`
- `data/filtered_demographics_combined.csv`

These files are useful for demographic comparisons between conditions and for analyses that require participant-level background information.

### `analyze_demographics.py`

This script compares demographic composition across conditions using descriptive statistics and inferential tests. It focuses on:

- age distributions
- sex composition
- primary language
- ethnicity

It reports sample sizes, descriptive summaries, contingency tables.

### `analysis_bayesianSDT.rmd`

This R Markdown document contains the Bayesian signal detection theory analysis used in the paper. It formalizes the relationship between evidence quality, verification behavior, and AI reasoning traces.


## Folders

The `data/` directory contains cleaned survey outputs. Key files include:

- cleaned demographic CSV files used for analysis
- processed reasoning/answer datasets used in the paper
- justification response files used for labeling and qualitative analysis

These files connect participant survey responses, background information, and the final analytic outputs.

`data/justification_responses` contains data from justification labeling