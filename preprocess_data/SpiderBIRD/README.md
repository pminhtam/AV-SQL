## Spider/BIRD  Schema Preprocess data


Code preprocessing for Spider and BIRD dataset.
Why need schema preprocess?
- To make data similar structure with Spider2.0 dataset.
- Easier run with chain-of-agents pipeline

There just have 2 steps:
- Preprocess table file : 
  - Add sample rows for each table
  - Add list of example_values for each column

### Step 1: 

Run 
```shell
python database/SpiderBIRD/schema_processing/step1_remake_tablesjsonl.py

```

### Step 2:

- Preprocess question json file:
  - Add 'instance_id' field to each question object to make it similar with Spider2.0 question format
  - 'instance_id' have format: f"{db_id}_{datáet}_{index}"

  - Run 
```shell
python database/SpiderBIRD/re_id_question_jsonl.py

```
