"""



"""

import os
import json
import copy
import random
import itertools
import psutil
from collections import defaultdict
from typing import Dict, Union, Any
import sqlite3
import pandas as pd

import func_timeout
from func_timeout import func_set_timeout

def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024**2) # Memory in MiB

###################################################################################################
########  Eval hardness ###########################################################################################

# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True


CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}

def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])
def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count

def eval_hardness(sql):
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
            (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
        return "medium"
    elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
            (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
            (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
        return "hard"
    else:
        return "extra"
###################################################################################################
###################################################################################################
@func_set_timeout(20)
def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all") -> Any:
    """
    Executes an SQL query on a database and fetches results.

    Args:
        db_path (str): The path to the database file.
        sql (str): The SQL query to execute.
        fetch (Union[str, int]): How to fetch the results. Options are "all", "one", "random", or an integer.

    Returns:
        Any: The fetched results based on the fetch argument.

    Raises:
        Exception: If an error occurs during SQL execution.
    """
    conn = None  # Initialize connection to None
    cursor = None
    # result = None
    try:
        conn =  sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        if fetch == "all":
            result = cursor.fetchall()
        elif fetch == "one":
            result = cursor.fetchone()
        elif fetch == "random":
            samples = cursor.fetchmany(10)
            result = random.choice(samples) if samples else []
        elif isinstance(fetch, int):
            result = cursor.fetchmany(fetch)
        else:
            result = "Invalid fetch argument. Must be 'all', 'one', 'random', or an integer."
    except Exception as e:
        # print(f"Error in execute_sql: {e}\nSQL: {sql}")
        result = f"Error in execute_sql: {e}\nSQL: {sql}"
    finally:
        # Ensure the connection is closed no matter what
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
            # print("Database connection closed.")
    try:
        if cursor is not None:
            cursor.close()
    except sqlite3.ProgrammingError as e:
        # print(f"Cursor already closed: {e}")
        pass
    if conn is not None:
        conn.close()
    del cursor, conn  # Clean up references
    return result
def read_jsonl(file_path):
    json_list_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
    # with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            json_list_data.append(json.loads(line))
        # return [json.loads(line) for line in file]
    return json_list_data
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _compare_sqls_outcomes(db_path: str, predicted_sql: str, ground_truth_sql: str) -> int:
    """
    Compares the outcomes of two SQL queries to check for equivalence.
    Copy code from RAQS-SQL/eval/evaluate.py
    Args:
        db_path (str): The path to the database file.
        predicted_sql (str): The predicted SQL query.
        ground_truth_sql (str): The ground truth SQL query.

    Returns:
        int: 1 if the outcomes are equivalent, 0 otherwise.

    Raises:
        Exception: If an error occurs during SQL execution.
    """
    try:
        ground_truth_res_ori = execute_sql(db_path, ground_truth_sql)
        predicted_res_ori = execute_sql(db_path, predicted_sql)
        num_row_gt = len(ground_truth_res_ori) if isinstance(ground_truth_res_ori, list) else 0
        num_row_pred = len(predicted_res_ori) if isinstance(predicted_res_ori, list) else 0
        ground_truth_res = list(itertools.chain.from_iterable(ground_truth_res_ori))
        # ground_truth_res = [round(item, 2) if is_number(item) else item for item in ground_truth_res]
        ground_truth_res = set(ground_truth_res)
        predicted_res = list(itertools.chain.from_iterable(predicted_res_ori))
        # predicted_res = [round(item, 2) if is_number(item) else item for item in predicted_res]
        predicted_res = set(predicted_res)
        # import pdb; pdb.set_trace()
    except func_timeout.exceptions.FunctionTimedOut as e:
        return 0, [], []
    except:
        return 0, [], []
    # if int(set(ground_truth_res).issubset(set(predicted_res))) and not int(set(predicted_res) == set(ground_truth_res)):
    #     import pdb; pdb.set_trace()
    # result = (
    #           int(set(predicted_res) == set(ground_truth_res))
    #           or
    #           (int(set(ground_truth_res).issubset(set(predicted_res))) and len(ground_truth_res) > 0)  and num_row_pred == num_row_gt)
    ########## recallEX <<<<<<<<<<<
    # result = True
    # if num_row_gt > 0:
    #     result  &= (
    #             (
    #                     int(set(predicted_res) == set(ground_truth_res))
    #               or
    #               (
    #                       len(ground_truth_res) > 0
    #                       and len(set(ground_truth_res).intersection(set(predicted_res))) >= num_row_gt
    #                       and (len(set(ground_truth_res).intersection(set(predicted_res))) % num_row_gt ==0)
    #                       and num_row_pred == num_row_gt
    #               )
    #             )
    #
    #     )
    #     # if result and not(int(set(predicted_res) == set(ground_truth_res))):
    #     #     print(f"ground true {ground_truth_res} , predicted {predicted_res}")
    #     #     import pdb; pdb.set_trace()
    # else:
    #     result  &= int(set(predicted_res) == set(ground_truth_res))
    ########## recallEX >>>>>>>>>>
    result = int(set(predicted_res) == set(ground_truth_res))
    # if int(set(predicted_res) != set(ground_truth_res)) and \
    #         (int(set(ground_truth_res).issubset(set(predicted_res)))) and \
    #         (len(ground_truth_res) > 0) and num_row_pred == num_row_gt:
    #     print(f"ground true {ground_truth_res} , predicted {predicted_res}")
    # return int(set(predicted_res) == set(ground_truth_res)), predicted_res, ground_truth_res
    # return int(set(predicted_res).issubset(set(ground_truth_res))), predicted_res, ground_truth_res
    # return int(set(ground_truth_res).issubset(set(predicted_res))), predicted_res_ori, ground_truth_res_ori
    return result, predicted_res_ori, ground_truth_res_ori



if __name__ == '__main__':
    ########## Spider
    # data_file_path = 'preprocessed_data/spider/dev.jsonl'
    # databases_dir = "../data_text2sql/spider_data/database/"

    # predict_log_dir = "logs/spider_gemini3pro_preview_cot_7_3"
    # predict_log_dir = "logs_correct_henrygpu/spider1_gemini3pro/"
    # predict_log_dir = "logs_correct_henrygpu/spider1_qwen25/"
    # predict_log_dir = "logs_correct_henrygpu/spider1_llama33/"

    # predict_log_dir = "logs/spider_gemini3pro_preview_final_nocte/"
    # predict_log_dir = "logs/spider_gemini3pro_preview_final_noexefeedback/"
    # predict_log_dir = "logs/spider_gemini3pro_preview_final_noexefeedback_1/"
    # predict_log_dir = "logs/spider_gemini3pro_preview_final_noplanner/"
    # predict_log_dir = "logs/spider_gemini3pro_preview_final_norephrase/"
    # predict_log_dir = "logs/spider_gemini3pro_preview_final_noschema/"

    # predict_log_dir = "logs/spider_llama33_final_nocte/"
    # predict_log_dir = "logs/spider_llama33_final_noexefeedback/"
    # predict_log_dir = "logs/spider_llama33_final_noexefeedback_1/"
    # predict_log_dir = "logs/spider_llama33_final_noplanner/"
    # predict_log_dir = "logs/spider_llama33_final_norephrase/"
    # predict_log_dir = "logs/spider_llama33_final_noschema/"


    ########### kaggleDBQA
    data_file_path = 'preprocessed_data/kaggleDBQA/kaggle_dbqa_test_questions.jsonl'
    databases_dir = "../data_text2sql/kaggle_dbqa/databases/"

    # predict_log_dir = "logs/kaggleDBQA_gpt5mini_cot_7_3/"
    # predict_log_dir = "logs/kaggleDBQA_gemini3pro_preview_cot_7_3/"
    # predict_log_dir = "logs/kaggleDBQA_llama33_cot_7_3/"
    predict_log_dir = "logs_correct/kaggledbqa_llama33"
    # predict_log_dir = "logs_correct/kaggledbqa_gemini3pro"
    data_questions = read_jsonl(data_file_path)
    # data_questions_old = read_jsonl(data_file_path_old)
    total_items = 0
    total_correct_items = 0
    df = None
    correct_list = []
    idx_run = 0
    total_question_by_level = {"easy":0, "medium":0, "hard":0, "extra":0}
    correct_by_level = {"easy":0, "medium":0, "hard":0, "extra":0}
    for idx_ques, question_item in enumerate(data_questions):
        instance_id = question_item["instance_id"]
        print(f"Processing instance_id: {instance_id}")
        print(f"Current process memory usage: {memory_usage_psutil():.2f} MiB")
        db_id = question_item["db_id"]
        gt_sql = question_item.get("SQL",question_item.get("query",""))
        # predict_sql_path = f"{predict_log_dir}/{instance_id}/{instance_id}.sql"
        # predict_sql_path = f"{predict_log_dir}/{instance_id}/{idx_run}/{instance_id}.sql"
        predict_sql_path = f"{predict_log_dir}/{instance_id}/{idx_run}/{instance_id}_revise.sql"
        if os.path.exists(predict_sql_path):
            with open(predict_sql_path, 'r+', encoding='utf-8') as predict_sql_file:
                predict_sql = predict_sql_file.read()
        else:
            print(f"Predict sql file not found: {predict_sql_path}")
            continue
        db_path = os.path.join(databases_dir, db_id, db_id + ".sqlite")
        result, predicted_res, ground_truth_res = _compare_sqls_outcomes(
            db_path=db_path,
            predicted_sql=predict_sql,
            ground_truth_sql=gt_sql
        )
        question_item["predict_sql"] = str(predict_sql)
        question_item["result"] = str(result)
        question_item["predicted_res"] = str(predicted_res)
        question_item["ground_truth_res"] = str(ground_truth_res)

        for key in question_item.keys():    #
            """
            Bộ spider có ấy field là list, excel ko lưu đc list nên chuyển thành str
            """
            if not isinstance(question_item[key], str):
                question_item[key] = str(question_item[key])
        # import pdb; pdb.set_trace()
        # print(question_item['difficulty'])
        total_items += 1
        ########### Spider hardness level <<<<<<<<<
        # import pdb; pdb.set_trace()
        question_item_sql = eval(question_item['sql'])
        hardness = eval_hardness(question_item_sql)
        total_question_by_level[hardness] += 1
        ########### Spider hardness level >>>>>>>>
        if result == 1:
            total_correct_items += 1
            correct_list.append(instance_id)
            correct_by_level[hardness] += 1
        else:
            pass
            # print("predicted_res :", predicted_res, "  |  ground_truth_res: ", ground_truth_res)
        # print(len(question_item))
        if df is not None:
            df = pd.concat([df, pd.DataFrame(
                question_item, index=[0])])
        else:
            df = pd.DataFrame(question_item, index=[0])
    print(f"Total items: {total_items}, Total correct items: {total_correct_items}")
    print(f"Question by level: {total_question_by_level}")
    print(f"Correct by level: {correct_by_level}")
    # df_correct_list = pd.DataFrame({"output": correct_list})
    # df_correct_list.to_csv(os.path.join(predict_log_dir, f"correct_predictions_{idx_run}.csv"), index=False)
    # df_correct_list.to_csv(os.path.join(predict_log_dir, f"correct_predictions_{idx_run}_revise.csv"), index=False)
    # df_correct_list.to_csv(os.path.join(predict_log_dir, f"correct_predictions_{idx_run}_revise_recallEX.csv"), index=False)
    # df_correct_list.to_csv(os.path.join(predict_log_dir, f"correct_predictions_{idx_run}_revise_recallEX_intersection.csv"), index=False)
    # df.to_excel(os.path.join(predict_log_dir, f"predictions_{idx_run}_norevise.xlsx"))
    # df.to_excel(os.path.join(predict_log_dir, f"predictions_{idx_run}.xlsx"))
    # df.to_excel(os.path.join(predict_log_dir, f"predictions_{idx_run}_recallEX.xlsx"))
    # df.to_excel(os.path.join(predict_log_dir, f"predictions_{idx_run}_recallEX_intersection.xlsx"))




"""
python oop_cot/evaluate/eval_spider_hardness.py

"""





