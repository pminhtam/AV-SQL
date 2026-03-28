"""


"""


import json
import os
import random
import sqlite3

from typing import Dict, Union, Any
from func_timeout import func_set_timeout


"""
"""

###################################################################################################
@func_set_timeout(600)
def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all") -> Any:
    """
    Executes an SQL query on a database and fetches results.
    Source copy code từ file :
    - src/utils/get_dbschema_context.py : exec sql
    - oop_cot/sql_exec_env.py  : code exec sql của Spider2.0 : lấy cả column_info
    Args:
        db_path (str): The path to the database file.
        sql (str): The SQL query to execute.
        fetch (Union[str, int]): How to fetch the results. Options are "all", "one", "random", or an integer.

    Returns:
        Any: The fetched results based on the fetch argument.

    Raises:
        Exception: If an error occurs during SQL execution.
    """
    conn = None  # Initialize connection to None
    cursor = None
    result = []
    column_info = None
    try:
        conn =  sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        column_info = cursor.description
        if fetch == "all":
            result = cursor.fetchall()
        elif fetch == "one":
            result = cursor.fetchone()
        elif fetch == "random":
            samples = cursor.fetchmany(10)
            result = random.choice(samples) if samples else []
        elif isinstance(fetch, int):
            result = cursor.fetchmany(fetch)
        else:
            err_msg = "Invalid fetch argument. Must be 'all', 'one', 'random', or an integer."
            print(err_msg)
    except Exception as e:
        # print(f"Error in execute_sql: {e}\nSQL: {sql}")
        err_msg = f"Error in execute_sql: {e}\nSQL: {sql}"
        print(err_msg)
    finally:
        # Ensure the connection is closed no matter what
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
            # print("Database connection closed.")
    try:
        if cursor is not None:
            cursor.close()
    except sqlite3.ProgrammingError as e:
        # print(f"Cursor already closed: {e}")
        pass
    if conn is not None:
        conn.close()
    del cursor, conn  # Clean up references
    return result, column_info

if __name__ == "__main__":
    dataset = "kaggleDBQA"
    kaggleDBQA_root_path = "/mnt/disk2/tampm/data_text2sql/kaggle_dbqa/"
    database_dir = os.path.join(kaggleDBQA_root_path, "databases")
    tables_json_path = os.path.join(kaggleDBQA_root_path, "KaggleDBQA/KaggleDBQA_tables.json")
    tables_json_name = os.path.basename(tables_json_path).split(".")[0]

    output_tables_json_path = f"preprocessed_data/{dataset}/{tables_json_name}_example_values.json"
    os.makedirs(os.path.dirname(output_tables_json_path), exist_ok=True)
    tables_list = json.load(open(tables_json_path, 'r'))
    for idx, table_info in enumerate(tables_list):
        db_id = table_info["db_id"]
        db_path = os.path.join(database_dir, f"{db_id}/{db_id}.sqlite")
        table_info["db_path"] = db_path
        print("Processing database:", db_id, f"({idx+1}/{len(tables_list)})")
        # Connect to the SQLite database and fetch sample rows for each table
        sample_rows = {}
        sql_query_get_row = "SELECT * FROM `{table_name}` LIMIT 5;"
        sql_query_get_distinct_values = "SELECT DISTINCT `{column_name}` FROM `{table_name}`  ORDER BY RANDOM();"
        for table_name in table_info["table_names_original"]:
            # try:
            rows_results, column_info = execute_sql(db_path, sql_query_get_row.format(table_name=table_name), fetch=5)
            if len(rows_results) == 0:
                continue
            columns = [desc[0] for desc in column_info]
            assert len(rows_results[0]) == len(columns), f"Column length mismatch in table {table_name} of database {db_id}"
            # except:
            #     import pdb; pdb.set_trace()
            for row in rows_results:
                row_dict = {columns[i]: row[i] for i in range(len(columns))}
                if table_name not in sample_rows:
                    sample_rows[table_name] = []
                sample_rows[table_name].append(row_dict)

            # import pdb ; pdb.set_trace()
        # TODO-DONE add example_values
        example_values = []
        max_table_id = 0
        for table_id, col in table_info["column_names_original"]:
            max_table_id = max(max_table_id, table_id)
            if table_id == -1:
                rows_results = []   #
            else:
                table_name = table_info["table_names_original"][table_id]
                rows_results, _ = execute_sql(db_path, sql_query_get_distinct_values.format(column_name=col,table_name=table_name), fetch=20)
            example_values.append([row[0] for row in rows_results])

        # Add important value infor
        table_info["db_stats"] = {
            "No. of tables": len(table_info["table_names_original"]),
            "No. of columns": len(table_info["column_names_original"])
        }
        table_info["sample_rows"] = sample_rows #  { table_name: [ {col1: val1, col2: val2}, {col1: val1_2, col2: val2_2}, ... ] }
        table_info["example_values"] = example_values
        # Add other fields as empty or default values
        table_info["table_to_projDataset"] = {}
        table_info["db_type"] = "sqlite"
        table_info["group_of_tables"] = []
        table_info["column_descriptions"] = table_info["column_descriptions"]

        # import pdb; pdb.set_trace()
        assert len(table_info["example_values"]) == len(table_info["column_types"]), f"Example values length mismatch in database {db_id}"
        assert len(table_info["example_values"]) == len(table_info["column_names_original"]), f"Example values length mismatch in database {db_id}"
        assert len(table_info["table_names_original"]) == max_table_id + 1, f"Table ID mismatch in database {db_id}"
    with open(output_tables_json_path, 'w') as f_out:
        json.dump(tables_list, f_out, indent=4)


"""
python database/kaggleDBQA/step1_preprocess_tablesjson.py
"""


