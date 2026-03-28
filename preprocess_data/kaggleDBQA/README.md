## Preprocess data for KaggleDBQA dataset



Why need schema preprocess?
- To make data similar structure with Spider2.0 dataset.
- Easier run with chain-of-agents pipeline


## Setup database 

```shell
git clone git@github.com:Chia-Hsuan-Lee/KaggleDBQA.git

```

Download KaggleDBQA dataset from : https://drive.google.com/drive/folders/1g-Y9Up2_mtQijqUXBNcTWmsFIvsFnV7f





## Install python requirement 

## Run preprocess


Step1 : 

```shell
python database/kaggleDBQA/step1_preprocess_tablesjson.py

```

Step2: file question 


```shell
python database/kaggleDBQA/step2_preprocess_questionjson.py
```




