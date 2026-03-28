"""

This file :
- Copy all .sql file into other folder


"""


import os
import json
import shutil
import re
import argparse

def read_jsonl(file_path):
    json_list_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
    # with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            json_list_data.append(json.loads(line))
    return json_list_data

def copy_log_files(src_log_dir, dest_log_dir):

    # Ensure the destination folder exists (create it if not)
    os.makedirs(dest_log_dir, exist_ok=True)

    # Iterate over all files in the source directory
    for file_name in os.listdir(src_log_dir):
        # Construct full file paths
        source_path = os.path.join(src_log_dir, file_name)
        destination_path = os.path.join(dest_log_dir, file_name)
        if ".json" in source_path:
            data_json = json.load(open(source_path, 'r', encoding='utf-8'))
            with open(destination_path, 'w', encoding='utf-8') as f:
                json.dump(data_json, f, ensure_ascii=False, indent=4)
        # Copy only files (not subdirectories)
        elif "log.txt" in source_path:
            content = open(source_path, 'r', encoding='utf-8').read()
            with open(destination_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            if os.path.isfile(source_path):
                shutil.copy2(source_path, destination_path)  # copy2 preserves file metadata
                print(f"Copied: {file_name}")
    print("All files copied successfully.")

if __name__ == '__main__':
    ########## Spider2-snow
    data_file_path = '../Spider2/spider2-snow/spider2-snow.jsonl'
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_log_dir", type=str, default="spider")
    args = parser.parse_args()
    predict_log_dir = args.predict_log_dir

    # idx_run = 0

    # submit_spider2_path = os.path.join(predict_log_dir, f"submit_{idx_run}/")
    # submit_log_spider2_path = os.path.join(predict_log_dir, f"submit_log_{idx_run}/")
    submit_spider2_path = os.path.join(predict_log_dir, f"submit/")
    submit_log_spider2_path = os.path.join(predict_log_dir, f"submit_log/")
    # path contain all .sql file for submit on spider2
    # assert
    if not os.path.exists(submit_spider2_path):
        os.makedirs(submit_spider2_path)
    data_questions = read_jsonl(data_file_path)
    total_items = 0
    total_correct_items = 0

    for question_item in data_questions:
        instance_id = question_item["instance_id"]
        print(instance_id)
        # predict_sql_path = f"{predict_log_dir}/{instance_id}/{idx_run}/{instance_id}.sql"
        predict_sql_path = f"{predict_log_dir}/{instance_id}/0/{instance_id}_revise.sql"
        if os.path.exists(predict_sql_path):

            shutil.copy2(predict_sql_path, submit_spider2_path + f"/{instance_id}.sql")
            copy_log_files(os.path.join(predict_log_dir, instance_id, str(0)), os.path.join(submit_log_spider2_path, instance_id))


