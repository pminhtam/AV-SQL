"""


From jsons schema, make prompt text for LLM to generate Text-to-SQL task.

This script just make database schema

The prompt should include:
- Task description
- Database schema information
    + Table names and their columns
    + Column types and descriptions (if available)
    + Relevant examples from the schema (if available)

- Examples of input questions and corresponding SQL queries (if available)

Prompt type :
- Full Schema Prompt: Includes complete database schema information.
- Gold Schema Prompt: Includes only the gold schema information relevant
- Compact Schema Prompt: A concise version of the schema
    + Compress with table patterns (From output of step1)
    + Compress with column patterns (From output of step2)





"""


import json
import os
import argparse
from collections import defaultdict
import numpy as np

def full_schema_prompt(schema_dict : dict) -> str:
    """
    Full schema prompt including all details.
    :param schema_dict:
    :return:
    """
    prompt_text = ""
    for table_name in schema_dict:
        table_name_fullname = schema_dict[table_name].get("table_to_tablefullname",
                                                          table_name)
        prompt_text += f"Table: {table_name_fullname}\n"
        # prompt_text += f"Table: {table_name}\n"
        for idx, col_name in enumerate(schema_dict[table_name]["columns_name"]):
            col_type = schema_dict[table_name]["columns_type"][idx]
            col_desc = schema_dict[table_name]["columns_description"][idx] if idx < len(schema_dict[table_name]["columns_description"]) else ""
            exp_values = schema_dict[table_name].get("example_values", [])

            prompt_text += f"  Column: {col_name} (Type: {col_type}"
            if col_desc:
                prompt_text += f", Description: {col_desc}"
            prompt_text += ")\n"
        if schema_dict[table_name]["primary_key"]:
            pk_str = ", ".join(schema_dict[table_name]["primary_key"])
            prompt_text += f"  Primary Key: {pk_str}\n"
        if schema_dict[table_name]["foreign_keys"]:
            fk_strs = []
            for fk in schema_dict[table_name]["foreign_keys"]:
                fk_strs.append(f"{fk[0][0]}.{fk[0][1]} -> {fk[1][0]}.{fk[1][1]}")
            fk_str = "; ".join(fk_strs)
            prompt_text += f"  Foreign Key: {fk_str}\n"
        prompt_text += "\n"
    return prompt_text

def gold_schema_prompt(schema_dict : dict) -> str:
    """
    Gold schema prompt including only relevant schema details.

    :param schema_dict:
    :return:
    """

    pass

def compress_table_pattern_prompt(schema_dict : dict, group_of_tables: dict) -> str:
    """
    Compress schema using table patterns.
    Group tables by their patterns to reduce redundancy.
    Still keep column details within each table.

    :param schema_dict:
    :return:
    """
    # From group_of_table : get schema for each group
    # If a group has multiple tables, only keep 1 table's details
    # If a group has single table, keep that table's details
    if group_of_tables is None or len(group_of_tables) == 0:
        """
        Nếu không có group_of_table thì trả về full schema luôn
        Cho bộ spider/bird
        """
        return full_schema_prompt(schema_dict)
    prompt_text = ""
    for group_pattern in group_of_tables:
        tables_in_group = group_of_tables[group_pattern]
        # Keep only 1 table's details
        representative_table = tables_in_group[0]
        if representative_table not in schema_dict:
            # Just skip those table
            # BUG20102025PM1747: Some table in group_of_tables not in schema_dict (spider2 local sqlite databases)
            continue
        # Build prompt for representative_table
        if len(tables_in_group) == 1:
            # Single table in group
            representative_table_fullname = schema_dict[representative_table].get("table_to_tablefullname",
                                                                                  representative_table)
            prompt_text += f"Table: {representative_table_fullname}\n"

            # prompt_text += f"Table: {representative_table}\n"
        else:
            # Multiple tables in group
            group_pattern_str = group_pattern.split("_cluster")[0]
            prompt_text += f"List of tables have same pattern {group_pattern_str} : {', '.join(tables_in_group)}\n"
            representative_table_fullname = schema_dict[representative_table].get("table_to_tablefullname",
                                                                                  representative_table)
            prompt_text += f"with representative Table for pattern {group_pattern_str} : {representative_table_fullname}\n"
            # prompt_text += f"with representative Table for pattern {group_pattern_str} : {representative_table}\n"
        for idx, col_name in enumerate(schema_dict[representative_table]["columns_name"]):
            col_type = schema_dict[representative_table]["columns_type"][idx]
            col_desc = schema_dict[representative_table]["columns_description"][idx] if idx < len(
                schema_dict[representative_table]["columns_description"]) else ""
            prompt_text += f"  Column: {col_name} (Type: {col_type}"
            if col_desc:
                prompt_text += f", Description: {col_desc}"
            prompt_text += ")\n"
        if schema_dict[representative_table]["primary_key"]:
            pk_str = ", ".join(schema_dict[representative_table]["primary_key"])
            prompt_text += f"  Primary Key: {pk_str}\n"
        if schema_dict[representative_table]["foreign_keys"]:
            fk_strs = []
            for fk in schema_dict[representative_table]["foreign_keys"]:
                fk_strs.append(f"{fk[0][0]}.{fk[0][1]} -> {fk[1][0]}.{fk[1][1]}")
            fk_str = "; ".join(fk_strs)
            prompt_text += f"  Foreign Key: {fk_str}\n"
        prompt_text += "\n"
    return prompt_text

def compress_column_pattern_prompt(schema_dict : dict, group_of_columns: dict) -> str:
    """
    Compress schema using column patterns.
    Group columns by their patterns to reduce redundancy.
    Still keep table details.

    :param schema_dict:
    :return:
    """
    prompt_text = ""
    # print(group_of_columns)
    # import pdb; pdb.set_trace()
    for table_name in schema_dict:
        table_name_fullname = schema_dict[table_name].get("table_to_tablefullname",
                                                                              table_name)

        prompt_text += f"Table: {table_name_fullname}\n"
        # prompt_text += f"Table: {table_name}\n"
        # Step 1: Build a reverse mapping from column name to group pattern
        colname_to_pattern = {}
        # print(grouped_of_columns.get(table_name,{}))
        for pattern, col_names in group_of_columns.get(table_name,{}).items():
            for col_name in col_names:
                colname_to_pattern[col_name] = pattern
        already_processed_cols_pattern = set()
        # Step 2: Process columns in the table
        # print(colname_to_pattern)
        # if len(colname_to_pattern) > 1:
        #     import pdb; pdb.set_trace()
        for idx, col_name in enumerate(schema_dict[table_name]["columns_name"]):
            cols_pattern = colname_to_pattern.get(col_name, col_name)
            if cols_pattern in already_processed_cols_pattern:
                continue
            already_processed_cols_pattern.add(cols_pattern)
            col_type = schema_dict[table_name]["columns_type"][idx]
            col_desc = schema_dict[table_name]["columns_description"][idx] if idx < len(
                schema_dict[table_name]["columns_description"]) else ""
            exp_values = schema_dict[table_name].get("example_values", [])
            if cols_pattern in group_of_columns.get(table_name, {}):
                # This is a grouped column pattern
                col_names_in_pattern = group_of_columns[table_name][cols_pattern]
                prompt_text += f"  Columns with pattern '{cols_pattern}': {', '.join(col_names_in_pattern)}\n"
                prompt_text += f"   with representative Column for pattern '{cols_pattern}' : {col_name} (Type: {col_type}"
                if col_desc:
                    prompt_text += f", Description: {col_desc}"
                prompt_text += ")\n"
            else:
                # This is a single column (not grouped)
                prompt_text += f"  Column: {col_name} (Type: {col_type}"
                if col_desc:
                    prompt_text += f", Description: {col_desc}"
                prompt_text += ")\n"

        if schema_dict[table_name]["primary_key"]:
            pk_str = ", ".join(schema_dict[table_name]["primary_key"])
            prompt_text += f"  Primary Key: {pk_str}\n"
        if schema_dict[table_name]["foreign_keys"]:
            fk_strs = []
            for fk in schema_dict[table_name]["foreign_keys"]:
                fk_strs.append(f"{fk[0][0]}.{fk[0][1]} -> {fk[1][0]}.{fk[1][1]}")
            fk_str = "; ".join(fk_strs)
            prompt_text += f"  Foreign Key: {fk_str}\n"
        prompt_text += "\n"
    return prompt_text
def compact_schema_prompt(schema_dict : dict,  group_of_tables: dict , group_of_columns: dict) -> str:
    """
    Compact schema prompt combining table and column pattern compression.
    Group both tables and columns by their patterns.

    Combine  compress_table_pattern_prompt + compress_column_pattern_prompt
    :param schema_dict:
    :return:
    """
    if group_of_tables is None or len(group_of_tables) == 0:
        """
        Nếu không có group_of_table thì trả về  compress_column_pattern_prompt
        Cho bộ spider/bird
        """
        return compress_column_pattern_prompt(schema_dict, group_of_columns)
    prompt_text = ""
    #
    for group_pattern in group_of_tables:
        tables_in_group = group_of_tables[group_pattern]
        # Keep only 1 table's details
        representative_table = tables_in_group[0]
        if representative_table not in schema_dict:
            # Just skip those table
            # : Some table in group_of_tables not in schema_dict (spider2 local sqlite databases)
            continue
        ##########################################################################################
        ### COMPRESS TABLES WITH PATTERN ### Like compress_table_pattern_prompt()
        ##########################################################################################
        # Build prompt for representative_table
        if len(tables_in_group) == 1:
            # Single table in group
            representative_table_fullname = schema_dict[representative_table].get("table_to_tablefullname", representative_table)
            prompt_text += f"Table: {representative_table_fullname}\n"
            # prompt_text += f"Table: {representative_table}\n"
        else:
            # Multiple tables in group
            group_pattern_str = group_pattern.split("_cluster")[0]
            prompt_text += f"List of tables have same pattern {group_pattern_str} : {', '.join(tables_in_group)}\n"
            representative_table_fullname = schema_dict[representative_table].get("table_to_tablefullname",
                                                                                  representative_table)
            prompt_text += f"with representative Table for pattern {group_pattern_str} : {representative_table_fullname}\n"
            # prompt_text += f"with representative Table for pattern {group_pattern_str} : {representative_table}\n"

        ##########################################################################################
        ### COMPRESS COLUMNS WITH PATTERN  ### like compress_column_pattern_prompt()
        ##########################################################################################
        # Step 1: Build a reverse mapping from column name to group pattern
        colname_to_pattern = {}
        # print(grouped_of_columns.get(table_name,{}))
        for pattern, col_names in group_of_columns.get(representative_table, {}).items():
            for col_name in col_names:
                colname_to_pattern[col_name] = pattern
        already_processed_cols_pattern = set()
        # Step 2: Process columns in the table
        # print(colname_to_pattern)
        # if len(colname_to_pattern) > 1:
        #     import pdb; pdb.set_trace()
        for idx, col_name in enumerate(schema_dict[representative_table]["columns_name"]):
            cols_pattern = colname_to_pattern.get(col_name, col_name)
            if cols_pattern in already_processed_cols_pattern:
                continue
            already_processed_cols_pattern.add(cols_pattern)
            col_type = schema_dict[representative_table]["columns_type"][idx]
            col_desc = schema_dict[representative_table]["columns_description"][idx] if idx < len(
                schema_dict[representative_table]["columns_description"]) else ""
            exp_values = schema_dict[representative_table].get("example_values", [])
            if cols_pattern in group_of_columns.get(representative_table, {}):
                # This is a grouped column pattern
                col_names_in_pattern = group_of_columns[representative_table][cols_pattern]
                prompt_text += f"  Columns with pattern '{cols_pattern}': {', '.join(col_names_in_pattern)}\n"
                prompt_text += f"   with representative Column for pattern '{cols_pattern}' : {col_name} (Type: {col_type}"
                if col_desc:
                    prompt_text += f", Description: {col_desc}"
                prompt_text += ")\n"
            else:
                # This is a single column (not grouped)
                prompt_text += f"  Column: {col_name} (Type: {col_type}"
                if col_desc:
                    prompt_text += f", Description: {col_desc}"
                prompt_text += ")\n"

        if schema_dict[representative_table]["primary_key"]:
            pk_str = ", ".join(schema_dict[representative_table]["primary_key"])
            prompt_text += f"  Primary Key: {pk_str}\n"
        if schema_dict[representative_table]["foreign_keys"]:
            fk_strs = []
            for fk in schema_dict[representative_table]["foreign_keys"]:
                fk_strs.append(f"{fk[0][0]}.{fk[0][1]} -> {fk[1][0]}.{fk[1][1]}")
            fk_str = "; ".join(fk_strs)
            prompt_text += f"  Foreign Key: {fk_str}\n"
        prompt_text += "\n"
    return prompt_text


def convert_db_schema_to_dict(db_schema_item_ori : dict, db_type="spider2") -> dict:
    """
    Convert db schema from json to dict format for easier processing.
    :param schema_item_ori:
    :return:
    dict of table infor
    schema_dict = {
        table_name : {
            "columns_name" : [col1, col2, ...],
            "columns_type" : [type1, type2, ...],
            "columns_description" : [desc1, desc2, ...],
            "example_values" : [[val1, val2, ...], ...]  # list of example values for each column
            "primary_key" : [col1, col2, ...],
            "foreign_keys" : [[col1, [ref_table1,ref_col1]], [col2, ref_table2,ref_col2], ...],
            "sample_rows" : [{col1: val1, col2: val2, ...}, ...]  # list of sample rows as dict
    }

    """
    if db_type == "spider2":
        column_names_original_key, column_names_key, column_types_key = "nested_column_names_original", "nested_column_names", "nested_column_types"
    else:
        column_names_original_key , table_names_original_key,column_names_key, table_names_original_key, column_types_key, primary_keys_key, foreign_keys_key = \
            "column_names_original", "table_names_original", "column_names", "table_names_original", "column_types", "primary_keys", "foreign_keys"

    schema_dict = {}
    table_to_cols = defaultdict(list)
    table_to_explaincols = defaultdict(list)
    table_to_typecols = defaultdict(list)
    table_to_tablefullname = defaultdict(list)
    for table_id, table_name in enumerate(db_schema_item_ori["table_names_original"]):
        table_to_tablefullname[table_name] = db_schema_item_ori["table_names"][table_id]
    for table_id, col in db_schema_item_ori[column_names_original_key]:
        if table_id == -1:
            continue
        table_to_cols[db_schema_item_ori["table_names_original"][table_id]].append(col)
    idx_col_type = 0
    for table_id, col in db_schema_item_ori[column_names_key]:
        if table_id == -1:
            continue
        table_to_explaincols[db_schema_item_ori["table_names_original"][table_id]].append(str(col))
        table_to_typecols[db_schema_item_ori["table_names_original"][table_id]].append(db_schema_item_ori[column_types_key][idx_col_type])
        idx_col_type += 1
    assert len(table_to_cols[db_schema_item_ori["table_names_original"][0]]) == len(table_to_explaincols[db_schema_item_ori["table_names_original"][0]])
    assert len(table_to_cols[db_schema_item_ori["table_names_original"][0]]) == len(table_to_typecols[db_schema_item_ori["table_names_original"][0]])
    primary_keys_list = []
    for pk_idx in db_schema_item_ori.get("primary_keys", []):
        if type(pk_idx) == int:
            table_id, col_name = db_schema_item_ori[column_names_original_key][pk_idx]
            primary_keys_list.append((db_schema_item_ori["table_names_original"][table_id], col_name))
        elif type(pk_idx) == list:
            for pk in pk_idx:
                table_id, col_name = db_schema_item_ori[column_names_original_key][pk]
                primary_keys_list.append((db_schema_item_ori["table_names_original"][table_id], col_name))
        else:
            raise ValueError(f"Unknown primary key format: {pk_idx}")
    fkeys_list = []
    for k1, k2 in db_schema_item_ori["foreign_keys"]:
        table_id_1, col_1 = db_schema_item_ori[column_names_original_key][k1]
        table_id_2, col_2 = db_schema_item_ori[column_names_original_key][k2]
        fkeys_list.append(
            [[db_schema_item_ori['table_names_original'][table_id_1], col_1], [db_schema_item_ori['table_names_original'][table_id_2], col_2]]
        )
    # example_value_list = []
    sample_rows_all = db_schema_item_ori.get("sample_rows", {})
    for table in table_to_cols:
        schema_dict[table] = {
            "columns_name" : table_to_cols[table],
            "columns_type" : table_to_typecols[table],
            "columns_description" : table_to_explaincols[table],
            "primary_key" : [col for tbl, col in primary_keys_list if tbl == table],
            "foreign_keys" : [fk for fk in fkeys_list if fk[0][0] == table or fk[1][0] == table],
            "table_to_tablefullname" : table_to_tablefullname[table],
            "sample_rows" : sample_rows_all.get(table, []),
        }

    return schema_dict


def make_str_from_item_dict(schema_item_ori : dict, prompt_type : str) -> str:
    # schema_dict = convert_db_schema_to_dict(schema_item_ori, db_type="spider2")
    schema_dict = convert_db_schema_to_dict(schema_item_ori, db_type="spider")
    group_of_tables = schema_item_ori.get("group_of_tables", {}) # Group of table

    group_of_columns = schema_item_ori.get("group_of_columns", {})

    # import pdb; pdb.set_trace()
    if prompt_type == "full":
        prompt_text = full_schema_prompt(schema_dict)
    elif prompt_type == "gold":
        prompt_text = gold_schema_prompt(schema_dict)
    elif prompt_type == "compress_table":
        prompt_text = compress_table_pattern_prompt(schema_dict, group_of_tables)
    elif prompt_type == "compress_column":
        prompt_text = compress_column_pattern_prompt(schema_dict, group_of_columns)
    elif prompt_type == "compact":
        prompt_text = compact_schema_prompt(schema_dict, group_of_tables, group_of_columns)
    else:
        prompt_text = ""
    return prompt_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--table_file_path', default='spider2_schema_processing/preprocessed_data_compress/spider2-lite/tables_preprocessed_step2_group_columns.json', type=str, help='the table file from step 1')
    args = parser.parse_args()
    db_schema_list = json.load(open(args.table_file_path, 'r'))
    prompt_length = []
    for item in db_schema_list:
        prompt = make_str_from_item_dict(item, prompt_type = "full")
        # prompt = make_str_from_item_dict(item, prompt_type = "compress_table")
        # prompt = make_str_from_item_dict(item, prompt_type = "compress_column")
        # prompt = make_str_from_item_dict(item, prompt_type = "compact")
        # import pdb; pdb.set_trace()
        # if len(item.get("group_of_columns", {})) > 3:
        #     max_col_in_group = 0
        #     for table in item.get("group_of_columns", {}):
        #         for pattern, col_names in item["group_of_columns"][table].items():
        #             max_col_in_group = max(max_col_in_group, len(col_names))
        #     if max_col_in_group > 10:
        #         import pdb; pdb.set_trace()
        prompt_length.append(len(prompt.split()))
        # if item['db_id'].lower() == 'google_dei':
        #     import pdb; pdb.set_trace()
        if len(prompt.split())>50000:
            print("DB ID: ", item['db_id'] , " prompt length: ", len(prompt.split()))
    print("AVG prompt length: ", sum(prompt_length)/len(prompt_length))
    print("MAX prompt length: ", max(prompt_length))
    prompt_length = np.array(prompt_length)
    # import pdb; pdb.set_trace()
    print("Num long prompt : ", sum(prompt_length>50000))


