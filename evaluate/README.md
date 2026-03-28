## Evaluateing 

## Eval :

#### Eval Spider1.0/BIRD
```shell
python evaluate/eval_spiderbird.py
```

#### Eval Spider2.0
For Spider2.0, need to preprocess by copy final .sql files to a separate folder first. Then run eval script from Spider2.0 repo.

Prepare .sql folders:
```shell
python evaluate/postprocess_spider20.py
```
Then move to Spider2.0 repo and run eval:
```shell
cd ../Spider2/spider2-snow/evaluation_suite
python evaluate.py --result_dir ../../../Spider2DecomposeQuestion2SQL/logs/spider2_lite_gemini3pro_preview_cot_7_3/submit --mode sql --timeout 600
```



### Some other useful scripts

Eval schema linking precision/recall:
```shell
python evaluate/eval_schema_linking_from_cte.py
```


Get EX by toks length on Spider2.0 dataset:
Follow Spider2.0 paper, questions are divided into 3 difficulty levels based on the number of tokens:
- Easy: questions with < 80 tokens
- Medium: questions with 80-160 tokens
- Hard: questions with > 160 tokens
But the toks information just have in Spider2.0-lite, so need map from Spider2.0-lite question_id to Spider2.0-snow question_id first.

```shell
python evaluate/get_toks_spider2.py
````
Then get results by difficulty level:
```shell
python evaluate/spider2snow_group_by_level.py
```
