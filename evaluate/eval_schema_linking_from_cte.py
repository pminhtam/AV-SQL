"""

From cte in log .json files, extract filtered schema linking information.
Eval the extracted schema linking against the gold schema linking.

Metric :
- Table level: ( Spider2.0/Spider1.0/BIRD)
    - Precision
    - Recall
- Column level : (Spider1.0/BIRD)
    - Precision
    - Recall


"""


import os
import sys
import json
import random
import os.path as osp

proj_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
print(proj_dir)
sys.path = [osp.join(proj_dir, "../")] + sys.path   #
# Đỡ phải chạy python -m

from av_sql.database_schema_manager import DatabaseSchemaManager, TextInforManager


proj_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
print(proj_dir)
sys.path = [osp.join(proj_dir, "../")] + sys.path   #
# Đỡ phải chạy python -m
from av_sql.utils import read_jsonl
from preprocess_data.spider2.step1_load_schema_infor import group_names_by_pattern

def cvt_list_to_dict(data_list):
    data_dict = {}
    for item in data_list:
        data_dict[item['instance_id']] = item['gold_tables']
    return data_dict

def cvt_list_to_dict_question(data_list):
    data_dict = {}
    for item in data_list:
        data_dict[item['instance_id']] = item['db_id']
    return data_dict

if __name__ == '__main__':
    dataset_name = "spider2_snow"
    # dataset_name = "spider2_lite"
    if dataset_name == "spider2_snow":
        data_file_path =  '../Spider2/spider2-snow/spider2-snow.jsonl'
        ground_truth_file_path = '../Spider2/methods/gold-tables/spider2-snow-gold-tables.jsonl'
        table_file_path = "spider2_schema_processing/preprocessed_data_compress/spider2-snow/tables_preprocessed_step2_group_columns_with_example_values.json"
        # predict_file_path = 'spider2_snow_table_used_gemini3pro_preview.json'
        # predict_file_path = 'spider2_snow_table_used_gemini3pro_preview_recall_1.json'
        # predict_file_path = 'spider2_snow_table_used_gemini3pro_preview_recall_all.json'
        # predict_file_path = 'spider2_snow_table_used_gemini3pro_preview_recall_all_2.json'
        # predict_file_path = 'spider2_snow_table_used_qwen25_recall_all.json'
        # predict_file_path = 'spider2_snow_table_used_llama33_recall_all.json'
        predict_file_path = 'spider2_snow_table_used_gpt5mini_recall_all.json'

        database_schema_manager_instance = DatabaseSchemaManager(table_file_path=table_file_path, dataset_name=dataset_name,
                                                             tokenizer_name="openai/gpt-oss-20b")
        data_questions = read_jsonl(data_file_path)
        data_questions_dict = cvt_list_to_dict_question(data_questions)

    elif dataset_name == "spider2_lite":
        data_file_path =  '../Spider2/spider2-lite/spider2-lite.jsonl'
        predict_file_path = ""

    predict_data = json.load(open(predict_file_path, 'r', encoding='utf-8'))
    ground_truth_data_list = read_jsonl(ground_truth_file_path)
    ground_truth_data_dict = cvt_list_to_dict(ground_truth_data_list)
    recall_item_levels = []
    precision_item_levels = []
    num_predict_tables = []
    compress_ratio = []
    for instance_id in predict_data:
        db_id = data_questions_dict[instance_id]
        schema_dict_dbid_ori = database_schema_manager_instance.db_schema_dict_all[db_id]
        # import pdb; pdb.set_trace()
        if instance_id not in ground_truth_data_dict:
            print(f"Warning: instance_id {instance_id} not in ground truth data")
        else:
            predict_table = [tbl.lower().split(".")[-1] for tbl in list(predict_data[instance_id].keys())]
            ground_truth_table = [tbl.lower().split(".")[-1] for tbl in ground_truth_data_dict[instance_id]]
            # Eval table level
            predict_group_table_dict = group_names_by_pattern(predict_table)
            ground_truth_group_table_dict = group_names_by_pattern(ground_truth_table)
            predict_group_table_list = list(predict_group_table_dict.keys())
            ground_truth_group_table_list = list(ground_truth_group_table_dict.keys())

            # tp_table = len(set(predict_table) & set(ground_truth_table))
            # precision_table = tp_table / len(predict_table) if len(predict_table) > 0 else 0.0
            # recall_table = tp_table / len(ground_truth_table) if len(ground_truth_table) > 0 else 0.0

            tp_table = len(set(predict_group_table_list) & set(ground_truth_group_table_list))
            precision_table = tp_table / len(predict_group_table_list) if len(predict_group_table_list) > 0 else 0.0
            recall_table = tp_table / len(ground_truth_group_table_list) if len(ground_truth_group_table_list) > 0 else 0.0

            recall_item_levels.append(recall_table)
            precision_item_levels.append(precision_table)
            num_predict_tables.append(len(predict_table))
            compress_ratio.append(len(predict_table)/ len(schema_dict_dbid_ori))
            # import pdb; pdb.set_trace()
            if len(predict_table) == 0:
                print(f"Instance ID: {instance_id}")
            # print(f"  Predicted Tables: {predict_table}")
            # print(f"  Ground Truth Tables: {ground_truth_table}")
            # print(f"  Table Level - Precision: {precision_table:.4f}, Recall: {recall_table:.4f}")

    print("Overall Evaluation:")
    overall_precision = sum(precision_item_levels) / len(precision_item_levels) if len(precision_item_levels) > 0 else 0.0
    overall_recall = sum(recall_item_levels) / len(recall_item_levels) if len(recall_item_levels) > 0 else 0.0
    print(f"  Overall Table Level - Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}")
    print(f"  Average Number of Predicted Tables per Instance: {sum(num_predict_tables) / len(num_predict_tables):.2f}")
    print(f"  Average Compress Ratio per Instance: {sum(compress_ratio) / len(compress_ratio):.4f}")
    """
    question level : just count if all tables are correct
    """
    recall_question_levels = [1.0 if r == 1.0 else 0.0 for r in recall_item_levels]
    overall_recall_question_level = sum(recall_question_levels) / len(recall_question_levels) if len(
        recall_question_levels) > 0 else 0.0
    print(f"  Overall Question Level Recall: {overall_recall_question_level:.4f}")
    precision_question_levels = [1.0 if p == 1.0 else 0.0 for p in precision_item_levels]
    overall_precision_question_level = sum(precision_question_levels) / len(precision_question_levels) if len(
        precision_question_levels) > 0 else 0.0
    print(f"  Overall Question Level Precision: {overall_precision_question_level:.4f}")


"""
python evaluate/eval_schema_linking_from_cte.py
"""