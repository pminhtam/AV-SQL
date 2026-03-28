"""

From exact list in .csv file ,
group result by level: easy, medium, hard
using toks data from spider2snow_toks.json from script get_toks_spider2.py


"""

import json
import pandas as pd


if __name__ == '__main__':
    toks_path = "spider2snow_toks.json"
    # list_correct_csv_path = "./logs_correct/spider2_snow_gemini3pro_preview/submit_0_revise/.csv"
    # list_correct_csv_path = "./logs_correct_henrygpu/spider2_snow_gpt5mini/submit_0_revise/.csv"
    # list_correct_csv_path = "./logs_correct_henrygpu/spider2_snow_qwen25/submit_0_revise/.csv"
    list_correct_csv_path = "./logs_correct_henrygpu/spider2_snow_llama33/submit_0_revise/.csv"
    df = pd.read_csv(list_correct_csv_path)
    with open(toks_path, 'r', encoding='utf-8') as f_toks:
        toks_data = json.load(f_toks)

    total_easy = 0
    total_medium = 0
    total_hard = 0
    for k, v in toks_data.items():
        toks = int(v)
        if toks < 80:
            total_easy += 1
        elif 80 <= toks < 160:
            total_medium += 1
        else:
            total_hard += 1
    print(f"Total easy: {total_easy}, medium: {total_medium}, hard: {total_hard}")
    level_groups = {
        'easy': [],
        'medium': [],
        'hard': []
    }
    true_questions = df['output'].tolist()

    for instance_id in true_questions:
        if instance_id in toks_data:
            toks = int(toks_data[instance_id])
            if toks < 80:
                level_groups['easy'].append(instance_id)
            elif 80 <= toks < 160:
                level_groups['medium'].append(instance_id)
            else:
                level_groups['hard'].append(instance_id)
        else:
            print(f"Warning: {instance_id} not found in toks data.")

    for level, instances in level_groups.items():
        print(f"{level.capitalize()} ({len(instances)} instances):")
        # print(instances)
        print()
"""
python evaluate/spider2snow_group_by_level.py
"""