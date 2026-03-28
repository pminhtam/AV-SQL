"""

easy : toks < 80
medium : 80 <= toks < 160
hard : toks >= 160



"""

import json
import os

if __name__ == '__main__':
    spider2_lite_path = "../Spider2/spider2-lite/evaluation_suite/gold/spider2lite_eval.jsonl"

    output_path = "spider2snow_toks.json"
    with open(spider2_lite_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        result = {}
        for line in f_in:
            data = json.loads(line)
            instance_id = data['instance_id']
            toks = data['toks']

            # Map to spider2-snow instance_id
            # if instance_id.startswith('bq') or instance_id.startswith('local'):
            if "sf" not in instance_id:
                instance_id = 'sf_' + instance_id

            result[instance_id] = toks

        json.dump(result, f_out, indent=4)
    print(f"Toks data has been written to {output_path}")

"""
python evaluate/get_toks_spider2.py
"""