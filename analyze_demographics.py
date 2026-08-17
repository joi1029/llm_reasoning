"""
Demographic Analysis Script
Analyzes respondent demographics across conditions.
Focuses on: Primary language, Age, Sex, Ethnicity simplified
"""

import pandas as pd
import numpy as np
from scipy import stats

# load demographic data for both surveys
demo_answer = pd.read_csv('data/reasoning_answer_prolific_demographic_export_68f6f19d4d84e6271577570a_approved_in_qualtrics.csv')
demo_reasoning = pd.read_csv('data/reasoning_prolific_demographic_export_68f46dab021946cc658677df_approved_in_qualtrics.csv')

# add condition labels
demo_answer['condition'] = 'answer'
demo_reasoning['condition'] = 'reasoning'

# combine datasets
demo_all = pd.concat([demo_answer, demo_reasoning], ignore_index=True)

print("=" * 60)
print("DEMOGRAPHIC ANALYSIS BY CONDITION")
print("=" * 60)

# columns of interest
focus_cols = ['Primary language', 'Age', 'Sex', 'Ethnicity simplified']

# sample sizes
print("\n--- SAMPLE SIZES ---")
print(demo_all['condition'].value_counts())
print(f"\nTotal participants: {len(demo_all)}")

print("\n" + "=" * 60)
print("AGE ANALYSIS")
print("=" * 60)

# descriptive stats by condition
age_stats = demo_all.groupby('condition')['Age'].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
print("\nAge descriptive statistics by condition:")
print(age_stats.round(2))

# independent samples t-test
age_answer = demo_all[demo_all['condition'] == 'answer']['Age'].dropna()
age_reasoning = demo_all[demo_all['condition'] == 'reasoning']['Age'].dropna()

t_stat, p_value = stats.ttest_ind(age_answer, age_reasoning)
print(f"\nIndependent samples t-test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")

# mann-whitney u test (non-parametric alternative)
u_stat, u_pvalue = stats.mannwhitneyu(age_answer, age_reasoning, alternative='two-sided')
print(f"\nMann-Whitney U test:")
print(f"  U-statistic: {u_stat:.1f}")
print(f"  p-value: {u_pvalue:.4f}")

# sex
print("\n" + "=" * 60)
print("SEX ANALYSIS")
print("=" * 60)

# frequency table
sex_crosstab = pd.crosstab(demo_all['condition'], demo_all['Sex'])
print("\nSex distribution by condition (counts):")
print(sex_crosstab)

# proportions
sex_props = pd.crosstab(demo_all['condition'], demo_all['Sex'], normalize='index') * 100
print("\nSex distribution by condition (percentages):")
print(sex_props.round(1))

# chi-square test
chi2, p_chi, dof, expected = stats.chi2_contingency(sex_crosstab)
print(f"\nChi-square test:")
print(f"  Chi-square statistic: {chi2:.3f}")
print(f"  Degrees of freedom: {dof}")
print(f"  p-value: {p_chi:.4f}")

# primary language
print("\n" + "=" * 60)
print("PRIMARY LANGUAGE ANALYSIS")
print("=" * 60)

# create binary variable for English vs non-English primary language
demo_all['English_primary'] = demo_all['Primary language'].apply(
    lambda x: 'English' if pd.notna(x) and 'English' in str(x).split(',')[0].strip() else 'Other'
)

# frequency table
lang_crosstab = pd.crosstab(demo_all['condition'], demo_all['English_primary'])
print("\nPrimary language (English vs Other) by condition (counts):")
print(lang_crosstab)

# proportions
lang_props = pd.crosstab(demo_all['condition'], demo_all['English_primary'], normalize='index') * 100
print("\nPrimary language by condition (percentages):")
print(lang_props.round(1))

# chi-square test (or fisher's exact if small cell counts)
chi2_lang, p_lang, dof_lang, expected_lang = stats.chi2_contingency(lang_crosstab)
print(f"\nChi-square test:")
print(f"  Chi-square statistic: {chi2_lang:.3f}")
print(f"  Degrees of freedom: {dof_lang}")
print(f"  p-value: {p_lang:.4f}")

# detailed language breakdown
print("\nDetailed primary language distribution:")
lang_detail = demo_all.groupby('condition')['Primary language'].value_counts()
print(lang_detail)

# ethnicity
print("\n" + "=" * 60)
print("ETHNICITY ANALYSIS")
print("=" * 60)

# frequency table
eth_crosstab = pd.crosstab(demo_all['condition'], demo_all['Ethnicity simplified'])
print("\nEthnicity distribution by condition (counts):")
print(eth_crosstab)

# proportions
eth_props = pd.crosstab(demo_all['condition'], demo_all['Ethnicity simplified'], normalize='index') * 100
print("\nEthnicity distribution by condition (percentages):")
print(eth_props.round(1))

# chi-square test
# filter out DATA_EXPIRED for valid test
eth_valid = demo_all[demo_all['Ethnicity simplified'] != 'DATA_EXPIRED']
eth_crosstab_valid = pd.crosstab(eth_valid['condition'], eth_valid['Ethnicity simplified'])

if eth_crosstab_valid.size > 0:
    chi2_eth, p_eth, dof_eth, expected_eth = stats.chi2_contingency(eth_crosstab_valid)
    print(f"\nChi-square test (excluding DATA_EXPIRED):")
    print(f"  Chi-square statistic: {chi2_eth:.3f}")
    print(f"  Degrees of freedom: {dof_eth}")
    print(f"  p-value: {p_eth:.4f}")
    
    # check expected cell counts
    min_expected = expected_eth.min()
    print(f"  Minimum expected count: {min_expected:.2f}")
    if min_expected < 5:
        print("  Note: Some expected counts < 5, chi-square may not be reliable.")

# summary
print("\n" + "=" * 60)
print("SUMMARY: DEMOGRAPHIC COMPARISON")
print("=" * 60)

summary_data = []

# age
summary_data.append({
    'Variable': 'Age (mean ± SD)',
    'Answer Condition': f"{age_answer.mean():.1f} ± {age_answer.std():.1f}",
    'Reasoning Condition': f"{age_reasoning.mean():.1f} ± {age_reasoning.std():.1f}",
    'Test': 't-test',
    'p-value': f"{p_value:.4f}"
})

# sex (% female)
pct_female_answer = (demo_all[demo_all['condition'] == 'answer']['Sex'] == 'Female').mean() * 100
pct_female_reasoning = (demo_all[demo_all['condition'] == 'reasoning']['Sex'] == 'Female').mean() * 100
summary_data.append({
    'Variable': 'Sex (% Female)',
    'Answer Condition': f"{pct_female_answer:.1f}%",
    'Reasoning Condition': f"{pct_female_reasoning:.1f}%",
    'Test': 'Chi-square',
    'p-value': f"{p_chi:.4f}"
})

# primary language (% English)
pct_eng_answer = (demo_all[demo_all['condition'] == 'answer']['English_primary'] == 'English').mean() * 100
pct_eng_reasoning = (demo_all[demo_all['condition'] == 'reasoning']['English_primary'] == 'English').mean() * 100
summary_data.append({
    'Variable': 'Primary Language (% English)',
    'Answer Condition': f"{pct_eng_answer:.1f}%",
    'Reasoning Condition': f"{pct_eng_reasoning:.1f}%",
    'Test': 'Chi-square',
    'p-value': f"{p_lang:.4f}"
})

# ethnicity (% White, excluding DATA_EXPIRED)
eth_answer_valid = demo_all[(demo_all['condition'] == 'answer') & (demo_all['Ethnicity simplified'] != 'DATA_EXPIRED')]
eth_reasoning_valid = demo_all[(demo_all['condition'] == 'reasoning') & (demo_all['Ethnicity simplified'] != 'DATA_EXPIRED')]
pct_white_answer = (eth_answer_valid['Ethnicity simplified'] == 'White').mean() * 100
pct_white_reasoning = (eth_reasoning_valid['Ethnicity simplified'] == 'White').mean() * 100
summary_data.append({
    'Variable': 'Ethnicity (% White)',
    'Answer Condition': f"{pct_white_answer:.1f}%",
    'Reasoning Condition': f"{pct_white_reasoning:.1f}%",
    'Test': 'Chi-square',
    'p-value': f"{p_eth:.4f}"
})

summary_df = pd.DataFrame(summary_data)
print("\n")
print(summary_df.to_string(index=False))