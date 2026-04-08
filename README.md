![visitors](https://visitor-badge.laobi.icu/badge?page_id=pminhtam.AV-SQL)

# AV-SQL

# Environment Setup

```shell
conda create -y -n av_sql python=3.10
conda activate av_sql
pip install -r requirements.txt
```

# Project Structure


Agents:
- Rewriter : [question.py](av_sql/question.py#L276) 
- View generator : [cte_agent.py](av_sql/cte_agent.py)
- Planner: [sql_agent.py](av_sql/sql_agent.py#L455)
- SQL generator: [sql_agent.py](av_sql/sql_agent.py)
- Revisor: [sql_agent.py](av_sql/sql_agent.py#L569)

# Preprocess data

## Reformat data

Preprocess all datasets (Spider, BIRD, KaggleDBQA, Spider2.0) into a unified format,

Run following : [README.md](preprocess_data/README.md)

Or you can directly download the preprocessed data from huggingface: https://huggingface.co/datasets/griffith-bigdata/av_sql_preprocessed_data

```shell
hf download griffith-bigdata/av_sql_preprocessed_data --repo-type=dataset --local-dir ./av_sql_preprocessed_data/
```


## Indexing data

Offline pre-process data in Database Value Preprocessing.
Using LSH to index the database values.

Run 
```shell
python av_sql/lsh_index.py --dataset_name spider
python av_sql/lsh_index.py --dataset_name bird
python av_sql/lsh_index.py --dataset_name kaggleDBQA
```


# Run 


```shell
python av_sql/main.py --config_file configs/spider/llama33_config.yaml
```

with `--config_file` to specify the config file for different datasets and models.

# Evaluate

### Eval Spider/BIRD/KaggleDBQA

```shell
python evaluate/eval_spiderbird.py --dataset_name spider --predict_log_dir ./logs/spider_llama33/
```

with `--dataset_name` to specify the dataset name (spider, bird, kaggleDBQA) and `--predict_log_dir` to specify the log dir of the prediction results.

### Eval Spider2.0

Contains two steps: 1) postprocess to get .sql files, 2) run eval script from Spider2.0 repo.

Step 1: postprocess to get .sql files

```shell
python evaluate/postprocess_spider20.py --predict_log_dir [predict_log_dir]
```

The .sql files will be saved in the `[predict_log_dir]/submit` directory.

Step 2: run eval script from Spider2.0 repo

```shell
cd ../Spider2/spider2-snow/evaluation_suite
python evaluate.py --result_dir [predict_log_dir]/submit --mode sql --timeout 600
``` 



# Acknowledgements
We would like to acknowledge the following open-source projects for their valuable contributions and inspiration to this work, particularly in terms of prompt design and codebase structure:
- CHESS : https://github.com/ShayanTalaei/CHESS/
- Alpha-SQL : https://github.com/HKUSTDial/Alpha-SQL
- ReFoRCE : https://github.com/Snowflake-Labs/ReFoRCE/
- MAC-SQL : https://github.com/wbbeyourself/MAC-SQL/
- DSR-SQL : https://github.com/DMIRLAB-Group/DSR-SQL

-----------
**Backup Statistics**

![Visitors](https://margherita-gustatory-zane.ngrok-free.dev/badge/pminhtam%2FAV-SQL.svg?ngrok-skip-browser-warning=true)
