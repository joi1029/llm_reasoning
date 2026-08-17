import pandas as pd
import numpy as np
from typing import Dict, List, cast


def clean_llm_reasoning_survey(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the LLM reasoning survey dataframe.
    Filter for valid responses and extract relevant columns for analysis.
    Args:
        df: Raw survey dataframe from Qualtrics
    Returns:
        Cleaned dataframe with renamed columns and filtered data
    """
    print(f"Initial number of rows: {len(df)}")
    
    # Remove header rows that got mixed in with data
    # Look for rows where key columns contain their own column names or import IDs
    header_indicators = ['{"ImportId"', 'Please enter', 'Finished', 'Progress', 'Response Type']
    mask = True
    for col in ['Finished', 'Progress', 'Q2']:
        if col in df.columns:
            col_mask = ~df[col].astype(str).str.contains('|'.join(header_indicators), na=False)
            mask = mask & col_mask
    
    df = df[mask].copy()
    print(f"After removing header rows: {len(df)}")
    
    #Only keep finished responses
    # if "Finished" in df.columns:
    #      df["Finished"] = pd.to_numeric(df["Finished"], errors='coerce')
    #      df = df[df["Finished"] == 1]
    #      print(f"After filtering for finished responses: {len(df)}")
    
    # Only keep responses with Progress
    if "Progress" in df.columns:
        df["Progress"] = pd.to_numeric(df["Progress"], errors='coerce')
        df = df[df["Progress"] >= 95]
        print(f"After filtering for 95% progress: {len(df)}")
    
    #Filter for valid Prolific IDs
    if "Q2" in df.columns:  # Q2 is the Prolific ID column
        df["Q2"] = df["Q2"].astype("string").str.strip()
        df = df[df["Q2"].str.len() >= 24]
        print(f"After filtering for valid Prolific IDs (24 characters): {len(df)}")


    basic_columns = {
        "StartDate": "start_date",
        "EndDate": "end_date", 
        "Status": "status",
        "Progress": "progress",
        "Duration (in seconds)": "duration_seconds",
        "Finished": "finished",
        "ResponseId": "response_id",
        "Q2": "prolific_id",
        "Q4": "consent",
        "Q6": "age",
        "Q7": "education_level", 
        "Q8": "occupation",
        "Q9": "llm_familiarity",
        "Q10": "llm_knowledge",
        "Q12": "ai_truthful1",
        "Q13": "ai_helpful1", 
        "Q14": "ai_confident1"
    }
    
    # Create mapping for all reasoning questions
    # Pattern: [question_number]_Q[question_type] where:
    # Q17 = correctness judgment (1=correct, 2=incorrect)
    # Q18 = confidence (1-5 scale, recoded: 5=very confident, 1=not confident)
    # Q19 = reasoning text
    # Q20 = knowledge level (1-5 scale)
    reasoning_columns = {}
    
    # Legal questions (22-31)
    for q_num in range(22, 32):
        reasoning_columns.update({
            f"{q_num}_Q17": f"legal_{q_num}_correctness",
            f"{q_num}_Q18": f"legal_{q_num}_confidence", 
            f"{q_num}_Q19": f"legal_{q_num}_reasoning",
            f"{q_num}_Q20": f"legal_{q_num}_knowledge",
            f"{q_num}_Q21_Page Submit": f"legal_{q_num}_time"
        })
    
    # Math questions (14-23) 
    for q_num in range(14, 24):
        reasoning_columns.update({
            f"{q_num}_Q23": f"math_{q_num}_correctness",
            f"{q_num}_Q24": f"math_{q_num}_confidence",
            f"{q_num}_Q25": f"math_{q_num}_reasoning", 
            f"{q_num}_Q26": f"math_{q_num}_knowledge",
            f"{q_num}_Q27_Page Submit": f"math_{q_num}_time"
        })
    
    # Finance questions (21-30)
    for q_num in range(21, 31):
        reasoning_columns.update({
            f"{q_num}_Q29": f"finance_{q_num}_correctness",
            f"{q_num}_Q30": f"finance_{q_num}_confidence",
            f"{q_num}_Q31": f"finance_{q_num}_reasoning",
            f"{q_num}_Q32": f"finance_{q_num}_knowledge", 
            f"{q_num}_Q33_Page Submit": f"finance_{q_num}_time"
        })
    
    # Medical questions (100-109)
    for q_num in range(100, 110):
        reasoning_columns.update({
            f"{q_num}_Q35": f"medical_{q_num}_correctness",
            f"{q_num}_Q36": f"medical_{q_num}_confidence",
            f"{q_num}_Q37": f"medical_{q_num}_reasoning",
            f"{q_num}_Q38": f"medical_{q_num}_knowledge",
            f"{q_num}_Q39_Page Submit": f"medical_{q_num}_time"
        })
    
    # Misinformation questions (100-109 with different Q pattern)
    for q_num in range(100, 110):
        reasoning_columns.update({
            f"{q_num}_Q41": f"misinfo_{q_num}_correctness", 
            f"{q_num}_Q42": f"misinfo_{q_num}_confidence",
            f"{q_num}_Q43": f"misinfo_{q_num}_reasoning",
            f"{q_num}_Q44": f"misinfo_{q_num}_knowledge",
            f"{q_num}_Q45_Page Submit": f"misinfo_{q_num}_time"
        })
    
    # other post-srvey questions
    post_columns = {
        "Q47": "crt_bat_ball",
        "Q48": "crt_machines", 
        "Q49": "crt_lilies",
        "Q51": "ai_truthful2",
        "Q52": "ai_helpful2",
        "Q53": "ai_confident2"
    }
    
    all_columns = {**basic_columns, **reasoning_columns, **post_columns}
    
    # rename columns that exist in the dataframe
    existing_columns = {k: v for k, v in all_columns.items() if k in df.columns}
    df = df.rename(columns=existing_columns)
    print(f"Renamed {len(existing_columns)} columns")
    
    columns_to_keep = list(existing_columns.values())
    
    # keep any additional columns that might be useful
    additional_cols = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in ['worktime', 'tasktime', 'offtask', 'ontask']):
            additional_cols.append(col)
    
    columns_to_keep.extend(additional_cols)
    

    final_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[final_columns].copy()
    
    # Convert correctness judgments to boolean 
    correctness_cols = [col for col in df.columns if 'correctness' in col]
    for col in correctness_cols:
        # First try numeric conversion (1=correct, 2=incorrect)
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        # If numeric values exist, convert them
        if not numeric_vals.isna().all():
            df[col] = numeric_vals.map({1: True, 2: False})

# Reverse coding: confidence, knowledge, trust in AI
    # Convert confidence and knowledge ratings to numeric
    rating_cols = [col for col in df.columns if any(x in col for x in ['confidence', 'knowledge'])]
    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Recode confidence: Original survey had 1=very confident, 5=not confident
    # Reverse so that 5=very confident, 1=not confident, which is more intuitive
    confidence_cols_to_recode = [col for col in df.columns if 'confidence' in col]
    for col in confidence_cols_to_recode:
        df[col] = 6 - df[col]  # Reverse
    

    # Recode knowledge: Original survey had 1=very knowledgeable, 5=not knowledgeable
    # Reverse so that 5=very knowledgeable, 1=not knowledgeable
    knowledge_cols_to_recode = [col for col in df.columns if 'knowledge' in col]
    for col in knowledge_cols_to_recode:
        df[col] = 6 - df[col]  # Reverse
    

    # Recode trust questions: Original survey had 1=strongly agree (high trust), 5=strongly disagree (low trust)
    # Reverse so that 5=high trust, 1=low trust
    trust_cols_to_recode = [col for col in df.columns if col.startswith('ai_truthful') or 
                           col.startswith('ai_helpful') or col.startswith('ai_confident')]
    for col in trust_cols_to_recode:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = 6 - df[col]  # Reverse
    
    # Convert timing columns to numeric
    timing_cols = [col for col in df.columns if 'time' in col]
    for col in timing_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert duration to numeric
    if 'duration_seconds' in df.columns:
        df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce')
    
    # Create a fresh copy to avoid fragmentation warnings
    df = df.copy()
    
    # Create summary statistics
    df['total_questions_answered'] = df[correctness_cols].notna().sum(axis=1)
    df['correct_answers'] = df[correctness_cols].sum(axis=1)
    df['accuracy'] = df['correct_answers'] / df['total_questions_answered']
    
    # Calculate average confidence and knowledge across domains
    confidence_cols = [col for col in df.columns if 'confidence' in col]
    knowledge_cols = [col for col in df.columns if 'knowledge' in col]
    
    if confidence_cols:
        df['avg_confidence'] = df[confidence_cols].mean(axis=1, skipna=True)
    if knowledge_cols:
        df['avg_knowledge'] = df[knowledge_cols].mean(axis=1, skipna=True)
    
    print(f"Final dataframe shape: {df.shape}")
    print(f"Columns in final dataframe: {len(df.columns)}")
    
    return df

def add_ground_truth_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ground truth comparison to determine if participants' judgments are correct.
    Loads reasoning results and compares participant responses with ground truth.
    
    For each question, creates a new column ending in '_got_correct' that indicates
    whether the participant's correctness judgment matched the ground truth.
    
    Also adds summary columns:
    - participant_accuracy_vs_ground_truth: proportion of questions answered correctly
    - total_questions_correct: count of questions answered correctly
    - total_questions_attempted: count of questions with responses
    """
    # Load ground truth data
    try:
        ground_truth = pd.read_csv(r'c:/Users/joice/MSRA/reasoning/data/reasoning_results_processed0728.csv', 
                                 encoding='latin1')
    except:
        print("Warning: Could not load ground truth file. Skipping ground truth comparison.")
        return df
    
    print(f"Loaded ground truth with {len(ground_truth)} questions")
    
    # Convert ground truth correctness to boolean (in case it's stored as string "TRUE"/"FALSE")
    if 'correctness' in ground_truth.columns:
        ground_truth['correctness'] = ground_truth['correctness'].map({
            'TRUE': True, 
            'FALSE': False,
            True: True,
            False: False,
            1: True,
            0: False
        })
    
    # Create ground truth lookup using question_id instead of id
    gt_lookup = dict(zip(ground_truth['question_id'], ground_truth['correctness']))
    
    # Add correctness evaluation columns
    participant_correct_count = 0
    total_comparisons = 0
    
    print("\nComparing participant responses with ground truth...")
    print(f"Available ground truth questions: {len(gt_lookup)}")
    
    # Create reverse mapping to get original question IDs from renamed columns
    # Legal questions: legal_XX_correctness -> XX_Q7  
    # Math questions: math_XX_correctness -> XX_Q13
    # Finance questions: finance_XX_correctness -> XX_Q19
    # Medical questions: medical_XXX_correctness -> XXX_Q25
    # Misinformation questions: misinfo_XXX_correctness -> XXX_Q31
    
    for col in df.columns:
        if 'correctness' in col:
            original_question_id = None
            
            # Extract the question ID based on column name pattern
            if col.startswith('legal_') and '_correctness' in col:
                # legal_22_correctness -> 22_Q7
                q_num = col.replace('legal_', '').replace('_correctness', '')
                original_question_id = f"{q_num}_Q7"
            elif col.startswith('math_') and '_correctness' in col:
                # math_14_correctness -> 14_Q13  
                q_num = col.replace('math_', '').replace('_correctness', '')
                original_question_id = f"{q_num}_Q13"
            elif col.startswith('finance_') and '_correctness' in col:
                # finance_21_correctness -> 21_Q22 (based on the ground truth data)
                q_num = col.replace('finance_', '').replace('_correctness', '')
                original_question_id = f"{q_num}_Q22"
            elif col.startswith('medical_') and '_correctness' in col:
                # medical_100_correctness -> 100_Q25
                q_num = col.replace('medical_', '').replace('_correctness', '')
                original_question_id = f"{q_num}_Q25"
            elif col.startswith('misinfo_') and '_correctness' in col:
                # misinfo_100_correctness -> 100_Q31
                q_num = col.replace('misinfo_', '').replace('_correctness', '')
                original_question_id = f"{q_num}_Q31"
            
            if original_question_id and original_question_id in gt_lookup:
                # Create column name that clearly indicates if participant got the question correct
                new_col = col.replace('correctness', 'got_correct')
                
                # Compare participant's judgment with ground truth
                # Only evaluate questions where participants actually provided an answer
                ground_truth_val = gt_lookup[original_question_id]
                
                # Create a mask for participants who actually answered this question
                answered_mask = df[col].notna()
                
                # Initialize the got_correct column with NaN (missing data)
                df[new_col] = pd.NA
                
                # Only compare for participants who actually answered the question
                df.loc[answered_mask, new_col] = df.loc[answered_mask, col] == ground_truth_val
                
                # Count correct judgments (only among those who answered)
                correct_judgments = df[new_col].sum()
                valid_responses = df[new_col].notna().sum()
                participant_correct_count += correct_judgments
                total_comparisons += valid_responses
                
                # Debug output for first few questions
                if total_comparisons <= 30:  # Only show first 30 to avoid clutter
                    print(f"Question {original_question_id} ({col}): Ground truth = {ground_truth_val}, "
                          f"Correct judgments = {correct_judgments}/{valid_responses}")
            elif original_question_id:
                print(f"Warning: No ground truth found for question {original_question_id} ({col})")
    
    # Find all columns that indicate if participant got questions correct
    got_correct_cols = [col for col in df.columns if 'got_correct' in col]
    if got_correct_cols:
        # Calculate accuracy: proportion of questions participant got correct
        df['participant_accuracy_vs_ground_truth'] = df[got_correct_cols].mean(axis=1, skipna=True)
        df['total_questions_correct'] = df[got_correct_cols].sum(axis=1, skipna=True)
        df['total_questions_attempted'] = df[got_correct_cols].notna().sum(axis=1)
        
        print(f"Added ground truth comparison for {len(got_correct_cols)} questions")
        print(f"Overall participant accuracy vs ground truth: {participant_correct_count}/{total_comparisons} = {participant_correct_count/total_comparisons:.1%}" if total_comparisons > 0 else "No comparisons available")
    else:
        print("Warning: No ground truth comparison columns were created")
    
    # Evaluate CRT questions
    print("\nEvaluating CRT (Cognitive Reflection Test) questions...")
    
    # Define correct answers with alternatives
    crt_correct_answers = {
        'crt_bat_ball': ['0.05', '.05', '0.05', '5', 'five cents', '5 cents', 'five', '$0.05'],
        'crt_machines': ['5', 'five', '5 minutes', 'five minutes'],
        'crt_lilies': ['47', 'forty seven', 'forty-seven', 'fortyseven', '47 days']
    }
    
    crt_correct_count = 0
    for crt_col, correct_vals in crt_correct_answers.items():
        if crt_col in df.columns:
            # Create got_correct column for this CRT question
            got_correct_col = crt_col + '_correct'
            df[got_correct_col] = pd.Series(False, index=df.index, dtype="boolean")
            
            # Check each possible correct answer (case-insensitive)
            for correct_val in correct_vals:
                df.loc[df[crt_col].astype(str).str.lower().str.strip() == correct_val.lower(), got_correct_col] = True
            
            # Set to NaN if original answer was NaN
            df.loc[df[crt_col].isna(), got_correct_col] = pd.NA
            
            correct = df[got_correct_col].sum()
            total = df[got_correct_col].notna().sum()
            print(f"  {crt_col}: {correct}/{total} correct ({correct/total:.1%})" if total > 0 else f"  {crt_col}: No responses")
            crt_correct_count += 1
    
    # Calculate total CRT score (0-3)
    crt_cols = [col for col in df.columns if col.endswith('_correct') and col.startswith('crt_')]
    if crt_cols:
        df['crt_score'] = df[crt_cols].sum(axis=1, skipna=True)
        print(f"\nAdded CRT score column (range 0-3)")
        print(f"Mean CRT score: {df['crt_score'].mean():.2f}")
    
    return df

def filter_to_demographic_participants(
    df: pd.DataFrame,
    demographic_file: str,
    condition_name: str,
    approved_only: bool = True
) -> pd.DataFrame:
    """
    Keep only survey participants whose Prolific ID is present in the matching
    demographic export.
    """
    if "prolific_id" not in df.columns:
        print(f"Warning: No prolific_id column found for {condition_name}. Skipping demographic filter.")
        return df

    demographics = pd.read_csv(demographic_file)
    if approved_only and "Status" in demographics.columns:
        demographics = demographics[
            demographics["Status"].astype("string").str.lower() == "approved"
        ].copy()

    valid_ids = set(
        demographics["Participant id"]
        .astype("string")
        .str.strip()
        .dropna()
    )
    survey_ids = set(
        df["prolific_id"]
        .astype("string")
        .str.strip()
        .dropna()
    )

    survey_not_in_demographics = sorted(survey_ids - valid_ids)
    demographics_not_in_survey = sorted(valid_ids - survey_ids)

    if survey_not_in_demographics:
        print(
            f"{condition_name} IDs in cleaned Qualtrics but not in "
            f"{'approved ' if approved_only else ''}demographic export:"
        )
        for prolific_id in survey_not_in_demographics:
            print(f"  {prolific_id}")

    if demographics_not_in_survey:
        print(
            f"{condition_name} IDs in {'approved ' if approved_only else ''}"
            "demographic export but not in cleaned Qualtrics:"
        )
        for prolific_id in demographics_not_in_survey:
            print(f"  {prolific_id}")

    if (
        len(survey_not_in_demographics) == 1
        and len(demographics_not_in_survey) == 1
    ):
        print(
            f"{condition_name} has one unmatched ID on each side; inspect whether "
            f"Qualtrics ID {survey_not_in_demographics[0]} should be corrected to "
            f"demographic ID {demographics_not_in_survey[0]}."
        )

    before = len(df)
    filtered = df[
        df["prolific_id"].astype("string").str.strip().isin(valid_ids)
    ].copy()
    removed = before - len(filtered)

    status_label = "approved " if approved_only else ""
    print(
        f"After filtering {condition_name} for IDs in {status_label}"
        f"demographic export: {len(filtered)}"
        f" (removed {removed})"
    )

    return filtered

def analyze_survey_completion(df: pd.DataFrame) -> Dict:
    stats = {}
    
    # Analyze completion by domain
    domains = ['legal', 'math', 'finance', 'medical', 'misinfo']
    
    for domain in domains:
        correctness_cols = [col for col in df.columns if f'{domain}_' in col and 'correctness' in col]
        if correctness_cols:
            completion_rate = df[correctness_cols].notna().mean().mean()
            stats[f'{domain}_completion_rate'] = completion_rate
            stats[f'{domain}_questions_available'] = len(correctness_cols)
    
    # Overall completion statistics
    all_correctness_cols = [col for col in df.columns if 'correctness' in col]
    if all_correctness_cols:
        stats['overall_completion_rate'] = df[all_correctness_cols].notna().mean().mean()
        stats['total_questions_available'] = len(all_correctness_cols)
    
    # Response time statistics
    timing_cols = [col for col in df.columns if 'time' in col and col != 'duration_seconds']
    if timing_cols:
        stats['avg_response_time'] = df[timing_cols].mean().mean()
        stats['median_response_time'] = df[timing_cols].median().median()
    
    return stats

if __name__ == "__main__":
    # Read the CSV file
    input_file1 = r"c:\Users\joice\MSRA\reasoning\data\LLM-Reasoning_Reasoning_August 4, 2026_19.18.csv"
    input_file2 = r"C:\Users\joice\MSRA\reasoning\data\LLM-Reasoning_Answer_August 4, 2026_19.24.csv"
    demographic_file1 = r"c:\Users\joice\MSRA\reasoning\data\reasoning_prolific_demographic_export_68f46dab021946cc658677df.csv"
    demographic_file2 = r"c:\Users\joice\MSRA\reasoning\data\reasoning_answer_prolific_demographic_export_68f6f19d4d84e6271577570a.csv"
    
    print("Reading CSV file...")
    df1 = pd.read_csv(input_file1)
    df2 = pd.read_csv(input_file2)

    # Clean the data
    print("\nCleaning data...")
    cleaned_df1 = clean_llm_reasoning_survey(df1)
    cleaned_df2 = clean_llm_reasoning_survey(df2)

    # Keep only participants present in the matching approved demographic export.
    cleaned_df1 = filter_to_demographic_participants(
        cleaned_df1,
        demographic_file1,
        "reasoning"
    )
    cleaned_df2 = filter_to_demographic_participants(
        cleaned_df2,
        demographic_file2,
        "answer"
    )

    # Add ground truth comparison
    print("\nAdding ground truth comparison...")
    cleaned_df1 = add_ground_truth_comparison(cleaned_df1)
    cleaned_df2 = add_ground_truth_comparison(cleaned_df2)

    # Remove a specific respondent (by prolific_id) from the cleaned answers dataframe
    # remove_pid = "672bc3ce315f709fe35a3392"
    # if "prolific_id" in cleaned_df2.columns:
    #     cleaned_df2 = cleaned_df2[
    #         cleaned_df2["prolific_id"].astype("string").str.strip() != remove_pid
    #     ].copy()

    # Analyze completion patterns
    print("\nAnalyzing completion patterns...")
    stats1 = analyze_survey_completion(cleaned_df1)
    stats2 = analyze_survey_completion(cleaned_df2)

    print("\nCompletion Statistics:")
    for key, value in stats1.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Save the cleaned data
    output_file1 = r"c:\Users\joice\MSRA\reasoning\cleaned_llm_reasoning.csv"
    output_file2 = r"c:\Users\joice\MSRA\reasoning\cleaned_llm_answers.csv"
    cleaned_df1.to_csv(output_file1, index=False)
    cleaned_df2.to_csv(output_file2, index=False)
    print(f"\nCleaned data saved to: {output_file1}")
    print(f"\nCleaned data saved to: {output_file2}")
