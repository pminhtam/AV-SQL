"""

add field :
- instance_id : local_{số thứ tự câu hỏi}
- "external_knowledge": null
- instruction : question

"""

import os
import json



if __name__ == "__main__":
    # dataset = 'spider'
    dataset = 'bird'
    if dataset == 'spider':
        spider_root_path = "/mnt/disk2/tampm/data_text2sql/spider_data/"
        # data_json_path = os.path.join(spider_root_path, "dev.json")
        data_json_path = os.path.join(spider_root_path, "train_spider.json")
        json_name = os.path.basename(data_json_path).split(".")[0]
    elif dataset == 'bird':
        # bird_root_path = "/mnt/disk2/tampm/data_text2sql/bird/dev_20240627/"
        # data_json_path = os.path.join(bird_root_path, "dev.json")
        bird_root_path = "/mnt/disk2/tampm/data_text2sql/bird/dev_20240627/"
        data_json_path = os.path.join("/mnt/disk2/tampm/data_text2sql/bird/bird_sql_dev_20251106/data/dev_20251106-00000-of-00001.json")

        # bird_root_path = "/mnt/disk2/tampm/data_text2sql/bird/train/"
        # data_json_path = os.path.join(bird_root_path, "train.json")

        json_name = os.path.basename(data_json_path).split(".")[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    output_file = f"preprocessed_data/{dataset}/{json_name}.jsonl"
    full_data = json.load(open(data_json_path, 'r'))
    idx = 0
    with open(output_file, 'w') as f_out:
        for item in full_data:
            # import pdb; pdb.set_trace()
            idx += 1
            item["instance_id"] = f"local_{dataset}_{idx}"
            item["external_knowledge"] = None
            item["instruction"] = item["question"] # Spider
            # item["instruction"] = item["question"] + "\n with information to help answer question: " + item.get("evidence", "")
            f_out.write(json.dumps(item) + "\n")


    print(f"Re-id question jsonl saved to {output_file}")


"""

python database/SpiderBIRD/re_id_question_jsonl.py
"""