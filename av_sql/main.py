"""

Main file to run the pipeline with the specified configuration.

"""





import os
import shutil
import sys
import json
import os.path as osp
from functools import partial
import random
import gc

import yaml
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
# import torch.multiprocessing as mp


proj_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
print(proj_dir)
sys.path = [osp.join(proj_dir)] + sys.path   # Thêm thư mục cha vào sys.path để import module từ đó



from av_sql.question import QuestionInstance
from av_sql.database_schema_manager import DatabaseSchemaManager
from av_sql.sql_exec_env import SqlExecEnv
from av_sql.lsh_index import ValueManager

def read_jsonl(file_path):
    json_list_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
    # with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            json_list_data.append(json.loads(line))
        # return [json.loads(line) for line in file]
    return json_list_data

def run_one_question(question_item: Dict, config: Dict):
    """
    question_item : will pass in map function
    config : pass by using partial function
    CAN not change the position of two argument. It causes error.
    Because executor.map function always pass first arg

    :param question_item:
    :param config:
    :return:
    """
    db_id = question_item.get("db_id", question_item.get("db", None))
    # get question first when spider/bird . if not exist, instruction in spider2
    question_text = question_item.get("question", question_item.get("instruction", None))
    instance_id = question_item.get("instance_id", question_item.get("id", "unknown_id"))
    #  : external_knowledge in BIRD is None. -> need get evidence first
    ext_knowledge = question_item.get("evidence", question_item.get("external_knowledge", ""))
    ext_knowledge = str(ext_knowledge)
    if db_id is None:
        print("No database ID found for question:", question_item)
    elif question_text is None:
        print("No question text found for question:", question_item)
    else:
        # import pdb; pdb.set_trace()
        # if instance_id[:2] == "bq" or instance_id[:2] == "ga":
        #     print("Ignore Bigquery question : ", instance_id)
        #     return
        for idx_run in range(config.get("n_runs",1)):
            question_instance = QuestionInstance(
            config=config,
            question_id=instance_id,
            question_text=question_text,
            ext_knowledge=ext_knowledge,
            db_id=db_id, idx_run = idx_run,
            database_schema_manager=DatabaseSchemaManager.get_instace(),
            )
            try:
                question_instance.run()
            except Exception as e:
                print(f"Error when running question {instance_id}: {e}")
            # BUG13122025 : out of memory. So need to clean RAM
            del question_instance
            gc.collect()
        print("Finished running question:", instance_id)
    return 0

def run_main(config: Dict[str, Any]):
    data_questions = read_jsonl(config['data_file_path'])
    ## Initialize singleton instances
    """
    singleton instances : instances that will be shared among multiple processes
    1. DatabaseSchemaManager
    2. SqlExecEnv
    3. ValueManager
    """
    database_schema_manager_instance = DatabaseSchemaManager(table_file_path=config['table_file_path'],
                                                             dataset_name=config.get('dataset_name', 'spider2'),
                                                             tokenizer_name=config.get('tokenizer_name', "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
                                                             )
    sql_env = SqlExecEnv(snowflake_credential_path=config.get('snowflake_credential_path','configs/snowflake_credential.json'),
                         bigquery_credential_path=config.get('bigquery_credential_path','configs/bigquery_credential.json'),
                         sqlite_root_dir=config['sqlite_root_dir'],
                         mysql_env=config.get('mysql_env',{}),
                         )
    if "value_manager" in config:
        value_manager_instance = ValueManager(config)
    ##############
    # log_dir = "log/test_run_main_av_sql"
    n_processes = config.get("n_processes", 1)
    if n_processes == 1 :
        for question_item in data_questions:
            instance_id = question_item.get("instance_id", question_item.get("id", "unknown_id"))
            # if instance_id != "bq010": # load sample value
            #     continue
            run_one_question(question_item, config)
    else:

        partial_run_one_question = partial(run_one_question, config=config) #
        # import pdb; pdb.set_trace()
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            start = 0
            cnt = len(data_questions)
            # list(tqdm(executor.map(partial_run_one_question, data_questions), total=len(data_questions), desc="Solving tasks"))
            while start < cnt:
                list(executor.map(partial_run_one_question, data_questions[start:start + n_processes]))
                start += n_processes
        # print(question_instance)
    sql_env.close_db()  # close all db

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="av_sql/configs/spider/qwen_config_1.yaml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    config_file = args.config_file

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    seed = config.get("seed", 17)
    random.seed(seed)
    np.random.seed(seed)
    log_dir = config.get('log_dir', "log/test_log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    shutil.copy2(config_file, log_dir)
    run_main(config)

"""
python av_sql/main.py --config_file configs/spider/qwen_config_1.yaml
"""