"""


- instance_id
- instruction
- db_id



"""



import os
import json
import glob

if __name__ == "__main__":
    dataset = "kaggleDBQA"
    kaggle_dbqa_json_folder = "/mnt/disk2/tampm/data_text2sql/kaggle_dbqa/KaggleDBQA/examples/"
    json_name = "kaggle_dbqa_test_questions"
    output_file = f"preprocessed_data/{dataset}/{json_name}.jsonl"
    list_all_test_json_file = glob.glob(os.path.join(kaggle_dbqa_json_folder, "*_test.json"))
    idx = 0

    for json_file in list_all_test_json_file:
        with open(json_file, "r") as f:
            data = json.load(f)
        for item in data:
            idx += 1
            item["instance_id"] = f"local_kaggledbqa_{idx}"
            item["instruction"] = item["question"]
            with open(output_file, "a") as f:
                f.write(json.dumps(item) + "\n")


"""

python database/kaggleDBQA/step2_preprocess_questionjson.py
 
"""

