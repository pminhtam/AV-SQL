## Schema Preprocess data


Process database schema for Spider2.0 dataset.


Spider2.0 schema have some special attributes:
- Each table stored in on json file -> need to combine these json files into one json file for each database
- foreign_keys and primary_keys : not explicitly defined in the schema, need to extract from create table statements
- table names : a lot of table names have shared prefix and similar column names -> need to group these tables to reduce the number of tables in the schema
  - Notice that some table have similar prefix but have different column names : could more or less some columns, need to compare column names to decide whether to group these tables or not
- column names : in bigquery column names are organized in nested structure, we can :
  - Spilit these table into multiple tables with flat structure column names to reduce prompting size
  - Or keep the nested structure and use dot notation to refer to these columns

#### Usage

#### Step 1: 

Load all json files and combine them into one json file for each database

```shell
python database/spider2/schema_processing/step1_load_schema_infor.py --dev spider2-lite
```

This script load all json files and group them into one json file for each database, and save the combined json files.

Also group table of each database based on table name prefix and column names similarity.

Sau khi chạy xong thu được file `tables_preprocessed.json` giống như file `tables.json` trong Spider/BIRD dataset.

#### Step 2: 

Group columns name based on patterns , column types and description similarity

```shell
python database/spider2/schema_processing/step2_group_column_by_pattern.py --table_file_path spider2_schema_processing/preprocessed_data_compress/spider2-lite/tables_preprocessed.json
```

This script process:
- Group columns based on patterns
- Check grouped columns with conditions
  - All columns in the same group should have the same type
  - Description of columns in the same group should be similar 


### Step 3: 

From json file generate prompting to input to LLM

```shell
python database/spider2/schema_processing/step3_make_prompt_text.py --table_file_path spider2_schema_processing/preprocessed_data_compress/spider2-lite/tables_preprocessed_step2_group_columns.json
```

There are 5 prompting strategies implemented:
- Original : original schema without any processing
- Grouped tables : group tables based on name prefix
- Grouped columns : group columns based on name patterns
- Grouped tables + grouped columns : group tables and columns
- Gold : use gold schema provided in spider2.0 dataset


### Step 4:

Spilit database schema into multiple smaller schemas based on table relations

```shell
python -m database.spider2.schema_processing.step4_split_schema_into_small_part --table_file_path "spider2_schema_processing/preprocessed_data_compress/spider2-lite/tables_preprocessed_step2_group_columns.json"
```

There are 3 strategies to split the database schema:
- One table per part
- Connected components based on foreign key relations
- Size limited : each part has size less than a limit (e.g., 1500 tokens)

