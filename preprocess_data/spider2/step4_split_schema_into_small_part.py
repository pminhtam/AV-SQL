"""


Split a large schema into small parts for easier processing and analysis.

Strategy:
- one_table : Each part contains a single table and its columns.
- base on foreign key connections: Each part contains tables connected through foreign keys.
- prompting size limit: Each part is created to fit within a specified token limit for prompting LLMs.




"""

import json
import os
import random
import argparse
from collections import defaultdict
from .step3_make_prompt_text import full_schema_prompt, compress_table_pattern_prompt, compress_column_pattern_prompt, \
    compact_schema_prompt



def split_one_table(schema_dict : dict) -> list:
    """
    Split schema into small parts, each part contains one table and its columns.
    :param schema_dict:
    :return:
    list of schema parts
    Each part is a dict of table infor
    schema_part = {
        table_name : {
            "columns_name" : [col1, col2, ...],
            "columns_type" : [type1, type2, ...],
            "columns_description" : [desc1, desc2, ...],
            "example_values" : [[val1, val2, ...], ...]  # list of example values for each column
            "primary_key" : [col1, col2, ...],
            "foreign_keys" : [[col1, [ref_table1,ref_col1]], [col2, ref_table2,ref_col2], ...]
    }
    """
    schema_parts = []
    for table_name, table_info in schema_dict.items():
        schema_part = {
            table_name: table_info
        }
        schema_parts.append(schema_part)
    return schema_parts


def split_by_foreign_keys(schema_dict : dict) -> list:
    """
    Split schema into small parts, each part contains tables connected through foreign keys.
    :param schema_dict:
    :return:
    list of schema parts
    Each part is a dict of table infor
    schema_part = {
        table_name : {
            "columns_name" : [col1, col2, ...],
            "columns_type" : [type1, type2, ...],
            "columns_description" : [desc1, desc2, ...],
            "example_values" : [[val1, val2, ...], ...]  # list of example values for each column
            "primary_key" : [col1, col2, ...],
            "foreign_keys" : [[col1, [ref_table1,ref_col1]], [col2, ref_table2,ref_col2], ...]
    }
    """
    schema_parts = []
    visited_tables = set()

    # Build adjacency list based on foreign keys
    adjacency_list = defaultdict(set)
    for table_name, table_info in schema_dict.items():
        for fk in table_info.get("foreign_keys", []):
            ref_table = fk[1][0]
            adjacency_list[table_name].add(ref_table)
            adjacency_list[ref_table].add(table_name)

    def dfs(table, current_part):
        visited_tables.add(table)
        current_part[table] = schema_dict[table]
        for neighbor in adjacency_list[table]:
            if neighbor not in visited_tables:
                dfs(neighbor, current_part)

    for table_name in schema_dict.keys():
        if table_name not in visited_tables:
            current_part = {}
            dfs(table_name, current_part)
            schema_parts.append(current_part)

    return schema_parts


def merge_parts_by_token_limit(schema_parts : list, token_limit : int = 50000) -> list:
    """
    Merge small schema parts into larger parts based on token limit.
    Input là list các part nhỏ : chính là output của các hàm split ở trên : split_one_table, split_by_foreign_keys, split_by_prompting_size_limit
    :param schema_parts:
    :param token_limit: maximum number of tokens for each part
    :return:
    list of merged schema parts
    Each part is a dict of table infor
    schema_part = {
        table_name : {
            "columns_name" : [col1, col2, ...],
            "columns_type" : [type1, type2, ...],
            "columns_description" : [desc1, desc2, ...],
            "example_values" : [[val1, val2, ...], ...]  # list of example values for each column
            "primary_key" : [col1, col2, ...],
            "foreign_keys" : [[col1, [ref_table1,ref_col1]], [col2, ref_table2,ref_col2], ...]
    }
    """
    merged_parts = []
    current_part = {}
    current_token_count = 0

    def estimate_token_count(table_info):
        # Simple estimation: number of columns * average tokens per column
        avg_tokens_per_text = 10  # This is a rough estimate
        return len(table_info["columns_name"]) * avg_tokens_per_text

    for schema_part in schema_parts:
        part_token_count = sum(estimate_token_count(table_info) for table_info in schema_part.values())
        if current_token_count + part_token_count > token_limit:
            # Start a new merged part
            if current_part:
                merged_parts.append(current_part)
            current_part = schema_part
            current_token_count = part_token_count
        else:
            current_part.update(schema_part)
            current_token_count += part_token_count

    if current_part:
        merged_parts.append(current_part)

    return merged_parts


def split_by_prompting_size_limit(schema_dict : dict, token_limit : int = 50000) -> list:
    """
    Split schema into small parts based on prompting size limit.
    :param schema_dict:
    :param token_limit: maximum number of tokens for each part
    :return:
    list of schema parts
    Each part is a dict of table infor
    schema_part = {
        table_name : {
            "columns_name" : [col1, col2, ...],
            "columns_type" : [type1, type2, ...],
            "columns_description" : [desc1, desc2, ...],
            "example_values" : [[val1, val2, ...], ...]  # list of example values for each column
            "primary_key" : [col1, col2, ...],
            "foreign_keys" : [[col1, [ref_table1,ref_col1]], [col2, ref_table2,ref_col2], ...]
    }
    """
    schema_parts = []
    current_part = {}
    current_token_count = 0
    def estimate_token_count(table_info):
        # Simple estimation: number of columns * average tokens per column
        # avg_tokens_per_text = 10  # This is a rough estimate
        avg_tokens_per_text = 2  # This is a rough estimate
        # return len(table_info["columns_name"]) * avg_tokens_per_column
        return len(str(table_info["columns_name"])) + len(str(table_info["columns_type"])) + len(str(table_info["columns_description"])) * avg_tokens_per_text

    estimate_token_table2token_count = {table_name: estimate_token_count(table_info) for table_name, table_info in schema_dict.items()}
    estimate_token_table2token_count_sorted = dict(sorted(estimate_token_table2token_count.items(), key=lambda item: item[1], reverse=False))

    for table_name, table_token_count in estimate_token_table2token_count_sorted.items():
        table_info = schema_dict[table_name]
        if current_token_count + table_token_count > token_limit:
            # Start a new part
            if current_part:
                schema_parts.append(current_part)
            current_part = {table_name: table_info}
            current_token_count = table_token_count
        else:
            current_part[table_name] = table_info
            current_token_count += table_token_count

    if current_part:
        schema_parts.append(current_part)
    # random.shuffle(schema_parts)
    return schema_parts


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
            "foreign_keys" : [[col1, [ref_table1,ref_col1]], [col2, ref_table2,ref_col2], ...]
    }

    Hàm này cũng giống như hàm convert_db_schema_to_dict ở bước step2 nhưng tổng quát hơn cho cả Spider/Bird dataset
    Copy từ script step3
    step3_make_prompt_text.py
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
    example_value_list = []

    for table in table_to_cols:
        schema_dict[table] = {
            "columns_name" : table_to_cols[table],
            "columns_type" : table_to_typecols[table],
            "columns_description" : table_to_explaincols[table],
            "primary_key" : [col for tbl, col in primary_keys_list if tbl == table],
            "foreign_keys" : [fk for fk in fkeys_list if fk[0][0] == table or fk[1][0] == table]
        }

    return schema_dict


def split_schema(schema_dict : dict, split_type : str , token_limit : int = 50000) -> list:
    if split_type == "one_table":
        schema_parts = split_one_table(schema_dict)
    elif split_type == "foreign_key":
        schema_parts = split_by_foreign_keys(schema_dict)
    elif split_type == "prompting_size_limit":
        schema_parts = split_by_prompting_size_limit(schema_dict, token_limit=token_limit)
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    return schema_parts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--table_file_path', default='spider2_schema_processing/preprocessed_data_compress/spider2-lite/tables_preprocessed_step2_group_columns.json', type=str, help='the table file from step 1')
    args = parser.parse_args()
    db_schema_list = json.load(open(args.table_file_path, 'r'))

    for item in db_schema_list:
        #  item : chứa thông tin của 1 database : table name, column name, column type, column description, example values
        # schema_parts = split_schema(item, split_type="one_table")
        # schema_parts = split_schema(item, split_type="foreign_key")
        schema_dict = convert_db_schema_to_dict(item, db_type="spider2")
        # schema_dict = convert_db_schema_to_dict(schema_item_ori, db_type="spider")
        schema_parts = split_schema(schema_dict, split_type="prompting_size_limit")
        print(len(schema_parts))
        # group_of_tables = item.get("group_of_tables",
        #                                       {})  # Group of table chua tat ca cac table. Neu table khong co pattern thi group cho 1 phan tu
        # group_of_columns = item.get("group_of_columns", {})
        # for idx, part in enumerate(schema_parts):
        #     # prompt_text = compact_schema_prompt(part, group_of_tables, group_of_columns)
        #     prompt_text = full_schema_prompt(part)
        #     print(len(prompt_text)) # nhiều prompt == 0 Vì group of tables chỉ dùng table[0] thôi, những table còn lại trong pattern không được tạo prompt

"""
python -m database.spider2.schema_processing.step4_split_schema_into_small_part --table_file_path "spider2_schema_processing/preprocessed_data_compress/spider2-lite/tables_preprocessed_step2_group_columns.json"

"""
