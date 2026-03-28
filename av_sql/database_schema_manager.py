"""





"""

import random
import json
import pandas as pd

from collections import defaultdict
from transformers import AutoTokenizer


def is_json_str(mystr):
    """
    Check if a string is a valid JSON format.
    Source : ReFoRCE/utils.py
    :param mystr:
    :return:
    """
    try:
        json_object = json.loads(mystr)
    except ValueError as e:
        return False
    return True
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class DatabaseSchemaManager:

    _instance = None  # Class variable to hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # If no instance exists, create a new one
            cls._instance = super(DatabaseSchemaManager, cls).__new__(cls)
            # Optional: Add any initialization logic here if __init__ is not used
            # For example, if you need to set attributes immediately after creation
        return cls._instance  # Return the existing or newly created instance

    def __init__(self, table_file_path: str= "",dataset_name : str='spider2', tokenizer_name: str="Qwen/Qwen2.5-Coder-1.5B-Instruct"):
        # __init__ will be called every time an instance is "created"
        # but the actual object creation is controlled by __new__
        if not hasattr(self, '_initialized'): # Prevent re-initialization
            self.table_file_path = table_file_path
            self.dataset_name = dataset_name
            self._initialized = True
            self.preprocess_schema_data()
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

        else:
            print(f"Instance already initialized with value: {self.table_file_path}")
    @staticmethod
    def get_instace():
        if DatabaseSchemaManager._instance is None:
            raise ValueError("Instance not created yet. Please create an instance first.")
        return DatabaseSchemaManager._instance

    def preprocess_schema_data(self):
        self.db_schema_list_all = json.load(open(self.table_file_path, 'r'))
        self.db_schema_dict_all = {item['db_id']: self.convert_db_schema_to_dict(item,item['db_id'],dataset_name=self.dataset_name) for item in self.db_schema_list_all}
        self.group_of_tables_dict_all = {item['db_id']: item.get("group_of_tables", {}) for item in self.db_schema_list_all}
        self.group_of_columns_dict_all = {item['db_id']: item.get("group_of_columns", {}) for item in self.db_schema_list_all}
    def make_str_from_item_dict(self):
        pass

    @classmethod
    def convert_db_schema_to_dict(cls, db_schema_item_ori : dict, db_id: str, dataset_name="spider2") -> dict:
        """
            Convert db schema from json to dict format for easier processing.
            :param db_schema_item_ori:
            :return:
            dict of table infor
            schema_dict = {
                table_name : {
                    "db_type": db_type, # nó chính là TextInforManager.api_type cảu từng question.
                    "db_id": db_id,
                    "columns_name" : [col1, col2, ...],
                    "columns_type" : [type1, type2, ...],
                    "columns_description" : [desc1, desc2, ...],
                    "example_values" : [[val1, val2, ...], ...]  # list of example values for each column
                    "primary_key" : [col1, col2, ...],
                    "foreign_keys" : [[col1, [ref_table1,ref_col1]], [col2, ref_table2,ref_col2], ...],
                    "sample_rows" : [{col1: val1, col2: val2, ...}, ...]  # list of sample rows as dict
            }


            """
        if dataset_name == "spider2" or "spider2" in dataset_name:
            column_names_original_key, column_names_key, column_types_key, table_full_names_key = \
                "nested_column_names_original", "nested_column_names", "nested_column_types", "table_names"

        else:
            column_names_original_key, table_names_original_key, column_names_key, table_names_original_key, column_types_key, primary_keys_key, foreign_keys_key = \
                "column_names_original", "table_names_original", "column_names", "table_names_original", "column_types", "primary_keys", "foreign_keys"
            table_full_names_key = "table_names_original"    #

        schema_dict = {}
        table_to_cols = defaultdict(list)
        table_to_explaincols = defaultdict(list)
        table_to_typecols = defaultdict(list)
        table_to_examplevals = defaultdict(list)
        table_to_tablefullname = defaultdict(list)
        for table_id, table_name in enumerate(db_schema_item_ori["table_names_original"]):
            table_to_tablefullname[table_name] = db_schema_item_ori[table_full_names_key][table_id]
        for table_id, col in db_schema_item_ori[column_names_original_key]:
            if table_id == -1:
                continue
            table_to_cols[db_schema_item_ori["table_names_original"][table_id]].append(col)
        idx_col_type = -1  # BUG11112025 :
        assert len(db_schema_item_ori[column_types_key]) == len(db_schema_item_ori[column_names_key])
        for table_id, col in db_schema_item_ori[column_names_key]:
            idx_col_type += 1   # FIX BUG11112025 : di chuyển lên đây
            if table_id == -1:
                continue
            table_to_explaincols[db_schema_item_ori["table_names_original"][table_id]].append(str(col))
            table_to_typecols[db_schema_item_ori["table_names_original"][table_id]].append(
                db_schema_item_ori[column_types_key][idx_col_type])
            if "example_values" in db_schema_item_ori:
                table_to_examplevals[db_schema_item_ori["table_names_original"][table_id]].append(
                    db_schema_item_ori["example_values"][idx_col_type])
            else:
                table_to_examplevals[db_schema_item_ori["table_names_original"][table_id]].append([])

        assert len(table_to_cols[db_schema_item_ori["table_names_original"][0]]) == len(
            table_to_explaincols[db_schema_item_ori["table_names_original"][0]])
        assert len(table_to_cols[db_schema_item_ori["table_names_original"][-1]]) == len(
            table_to_typecols[db_schema_item_ori["table_names_original"][-1]])
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
                [[db_schema_item_ori['table_names_original'][table_id_1], col_1],
                 [db_schema_item_ori['table_names_original'][table_id_2], col_2]]
            )
        # example_value_list = []
        sample_rows_all = db_schema_item_ori.get("sample_rows", {})
        db_type = db_schema_item_ori.get("db_type", "") #

        for table in table_to_cols:
            schema_dict[table] = {
                "db_type": db_type, #
                "db_id": db_id,
                "columns_name": table_to_cols[table],
                "columns_type": table_to_typecols[table],
                "columns_description": table_to_explaincols[table],
                "example_values": table_to_examplevals[table],
                "primary_key": [col for tbl, col in primary_keys_list if tbl == table],
                "foreign_keys": [fk for fk in fkeys_list if fk[0][0] == table or fk[1][0] == table],
                "table_to_tablefullname": table_to_tablefullname[table],
                "sample_rows" : sample_rows_all.get(table, []),
            }

        return schema_dict

    @classmethod
    def get_sample_rows_prompt(cls, sample_rows: list, max_size: int = 3000) -> str:
        """
        Get sample rows prompt text.
        Sample row cho table
        :param sample_rows:
        :param max_size: Maximum number of words in the prompt.

        :return:
        Sample Rows of this table ( symbol ; separate value in a row ) :
            Row Names: col1 ;  col2 ; col3
            value11 ; value12 ; value13
            value21 ; value22 ; value23
            value31 ; value32 ; value33 ...
        """
        if len(sample_rows) == 0:   # BUG20112025 : when sample_rows = []
            return ""
        prompt_text = " Sample Rows of this table ( symbol ; separate value in a row ) :\n"
        row_name = [f"{col}" for col, _ in sample_rows[0].items()]
        row_str = "Row Names: "
        row_str += " ;\t".join(row_name)
        len_row_str = len(row_str)
        prompt_text += f" {row_str}\n"
        seperate_column_name = ["-----"]*len(row_name)
        prompt_text += " ;\t".join(seperate_column_name) + "\n"
        for row in sample_rows[:3]: # Only include up to 3 sample rows
            row_items = []
            for _, val in row.items():
                if (type(val) == dict or is_json_str(str(val))) and (not is_number(str(val))):
                    # import pdb; pdb.set_trace()
                    if type(val) is dict:
                        json_val = val
                    else:
                        json_val = json.loads(str(val))  #
                    if type(json_val) is list:

                        json_val_input = json_val[:1]
                    else:
                        json_val_input = json_val

                    json_val_str = json.dumps(json_val_input, separators=(',', ':'))


                    if len(json_val_str) > 2002:
                        row_items.append(f"{str(json_val_str)[:2000]} ...")
                    else:
                        row_items.append(f"{str(json_val_str)}")
                else:
                    if "{" in str(val) and "}" in str(val): # value là json nhưng parsing lỗi
                        if len(str(val)) > 602:
                            row_items.append(f"{str(val)[:600]} ...")
                        else:
                            row_items.append(f"{str(val)}")
                    elif len(str(val)) > 52:
                        row_items.append(f"{str(val)[:50]} ...")
                    else:
                        row_items.append(f"{str(val)}")

            row_str = " ;\t".join(row_items)
            prompt_text += f" {row_str}\n"
            if len(prompt_text) > 2*len_row_str + max_size: #
                prompt_text = prompt_text[:2*len_row_str + max_size*5] #

                prompt_text += " ...\n"
                return prompt_text
        return prompt_text

    @classmethod
    def get_example_value_text(cls, idx, exp_values, columns_name):
        """
         example

        :param idx:
        :param exp_values:
        :param columns_name:
        :return:
              : "value1 ; value2 ; value3 ; ... "
        """
        col_desc = ""
        if exp_values and idx < len(exp_values) and len(exp_values) == len(columns_name):

            if len(exp_values[idx]) > 0:
                example_vals_str = ""
                exp_values_list = exp_values[idx]
                random.shuffle(exp_values_list)  #
                for val in exp_values_list[:3]:
                    if (type(val) == dict  or is_json_str(str(val))) and (not is_number(str(val))):
                        # nested type
                        if type(val) is dict:
                            val = json.dumps(json.loads(val))
                        else:
                            val = str(val)
                        if len(str(val)) > 1002:
                            val = str(val)[:1000] + " ..."
                    else:
                        if "{" in str(val) and "}" in str(val):  #
                            if len(str(val)) > 602:
                                val = str(val)[:600] + " ..."
                            else:
                                val = str(val)
                        elif len(str(val)) > 52:
                            val = str(val)[:50] + " ..."    #
                    if len(example_vals_str) < 50:
                        if len(example_vals_str) > 0: #
                            example_vals_str += " ; "
                        example_vals_str += str(val)
                    else:   #
                        example_vals_str += " ;  ..."
                        break
                col_desc += f", Example Values:[ {example_vals_str} ]"
        return col_desc
    @classmethod
    def full_schema_prompt(cls,schema_dict: dict, is_use_sample_rows: bool=False, is_use_col_desc: bool=True) -> str:
        """
        Full schema prompt including all details.
        :param schema_dict:
        :return:
        """
        prompt_text = ""
        for table_name in schema_dict:
            table_name_fullname = schema_dict[table_name].get("table_to_tablefullname",
                                                              table_name)
            db_id = schema_dict[table_name]["db_id"]
            prompt_text += f"DATABASE_NAME : {db_id} . Table: {table_name_fullname}\n"
            # prompt_text += f"Table: {table_name}\n"
            exp_values = schema_dict[table_name].get("example_values", [])

            for idx, col_name in enumerate(schema_dict[table_name]["columns_name"]):
                col_type = schema_dict[table_name]["columns_type"][idx]
                col_desc = schema_dict[table_name]["columns_description"][idx] if idx < len(
                    schema_dict[table_name]["columns_description"]) else ""

                prompt_text += f"  Column: {col_name} (Type: {col_type}"
                col_desc += cls.get_example_value_text(idx, exp_values,
                                                       schema_dict[table_name][
                                                           "columns_name"])
                # schema_dict[table_name]["columns_name"]
                if col_desc and is_use_col_desc:
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
            if is_use_sample_rows:
                prompt_text += cls.get_sample_rows_prompt(schema_dict[table_name]["sample_rows"])
            prompt_text += "\n"
        return prompt_text


    @classmethod
    def compress_table_pattern_prompt(cls, schema_dict: dict, group_of_tables: dict, is_use_sample_rows: bool=False, is_use_col_desc: bool=True) -> str:
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

            return cls.full_schema_prompt(schema_dict, is_use_sample_rows=is_use_sample_rows)
        prompt_text = ""
        for group_pattern in group_of_tables:
            tables_in_group = group_of_tables[group_pattern]
            # Keep only 1 table's details
            representative_table = tables_in_group[0]
            if representative_table not in schema_dict:
                # Just skip those table
                continue
            # Build prompt for representative_table
            if len(tables_in_group) == 1:
                # Single table in group
                representative_table_fullname = schema_dict[representative_table].get("table_to_tablefullname",
                                                                                      representative_table)
                db_id = schema_dict[representative_table]["db_id"]
                prompt_text += f"DATABASE_NAME : {db_id} . Table: {representative_table_fullname}\n"

                # prompt_text += f"Table: {representative_table}\n"
            else:
                # Multiple tables in group
                group_pattern_str = group_pattern.split("_cluster")[0]
                prompt_text += f"List of tables have same pattern {group_pattern_str} : {', '.join(tables_in_group)}\n"
                representative_table_fullname = schema_dict[representative_table].get("table_to_tablefullname",
                                                                                      representative_table)
                prompt_text += f"with representative Table for pattern {group_pattern_str} : {representative_table_fullname}\n"
                # prompt_text += f"with representative Table for pattern {group_pattern_str} : {representative_table}\n"
            exp_values = schema_dict[representative_table].get("example_values", [])
            for idx, col_name in enumerate(schema_dict[representative_table]["columns_name"]):
                col_type = schema_dict[representative_table]["columns_type"][idx]
                col_desc = schema_dict[representative_table]["columns_description"][idx] if idx < len(
                    schema_dict[representative_table]["columns_description"]) else ""
                prompt_text += f"  Column: {col_name} (Type: {col_type}"
                col_desc += cls.get_example_value_text(idx, exp_values,
                                                       schema_dict[representative_table][
                                                           "columns_name"])
                if col_desc and is_use_col_desc:
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
            if is_use_sample_rows:
                prompt_text += cls.get_sample_rows_prompt(schema_dict[representative_table]["sample_rows"])
            prompt_text += "\n"
        return prompt_text

    @classmethod
    def compress_column_pattern_prompt(cls, schema_dict: dict, group_of_columns: dict, is_use_sample_rows: bool=False, is_use_col_desc: bool=True) -> str:
        """
        Compress schema using column patterns.
        Group columns by their patterns to reduce redundancy.
        Still keep table details.

        :param schema_dict:
        :return:
        """
        prompt_text = ""
        for table_name in schema_dict:
            table_name_fullname = schema_dict[table_name].get("table_to_tablefullname",
                                                              table_name)
            db_id = schema_dict[table_name]["db_id"]

            prompt_text += f"DATABASE_NAME : {db_id} . Table: {table_name_fullname}\n"
            # prompt_text += f"Table: {table_name}\n"
            # Step 1: Build a reverse mapping from column name to group pattern
            colname_to_pattern = {}
            # print(grouped_of_columns.get(table_name,{}))
            for pattern, col_names in group_of_columns.get(table_name, {}).items():
                for col_name in col_names:
                    colname_to_pattern[col_name] = pattern
            already_processed_cols_pattern = set()
            # Step 2: Process columns in the table

            exp_values = schema_dict[table_name].get("example_values", [])
            for idx, col_name in enumerate(schema_dict[table_name]["columns_name"]):
                cols_pattern = colname_to_pattern.get(col_name, col_name)
                if cols_pattern in already_processed_cols_pattern:
                    continue
                already_processed_cols_pattern.add(cols_pattern)
                col_type = schema_dict[table_name]["columns_type"][idx]
                col_desc = schema_dict[table_name]["columns_description"][idx] if idx < len(
                    schema_dict[table_name]["columns_description"]) else ""
                col_desc += cls.get_example_value_text(idx, exp_values,
                                                       schema_dict[table_name][
                                                           "columns_name"])
                if cols_pattern in group_of_columns.get(table_name, {}):
                    # This is a grouped column pattern
                    col_names_in_pattern = group_of_columns[table_name][cols_pattern]
                    prompt_text += f"  Columns with pattern '{cols_pattern}': {', '.join(col_names_in_pattern)}\n"
                    prompt_text += f"   with representative Column for pattern '{cols_pattern}' : {col_name} (Type: {col_type}"
                    if col_desc and is_use_col_desc:
                        prompt_text += f", Description: {col_desc}"
                    prompt_text += ")\n"
                else:
                    # This is a single column (not grouped)
                    prompt_text += f"  Column: {col_name} (Type: {col_type}"
                    if col_desc and is_use_col_desc:
                        prompt_text += f", Description: {col_desc}"
                    prompt_text += ")\n"

            # import pdb; pdb.set_trace()
            if schema_dict[table_name]["primary_key"]:
                pk_str = ", ".join(schema_dict[table_name]["primary_key"])
                prompt_text += f"  Primary Key: {pk_str}\n"
            if schema_dict[table_name]["foreign_keys"]:
                fk_strs = []
                for fk in schema_dict[table_name]["foreign_keys"]:
                    fk_strs.append(f"{fk[0][0]}.{fk[0][1]} -> {fk[1][0]}.{fk[1][1]}")
                fk_str = "; ".join(fk_strs)
                prompt_text += f"  Foreign Key: {fk_str}\n"
            if is_use_sample_rows:
                prompt_text += cls.get_sample_rows_prompt(schema_dict[table_name]["sample_rows"])
            prompt_text += "\n"
        return prompt_text

    @classmethod
    def compact_schema_prompt(cls, schema_dict: dict, group_of_tables: dict, group_of_columns: dict, is_use_sample_rows: bool=False, is_use_col_desc: bool=True) -> str:
        """
        Compact schema prompt combining table and column pattern compression.
        Group both tables and columns by their patterns.

        Combine  compress_table_pattern_prompt + compress_column_pattern_prompt
        :param schema_dict:
        :return:
        """
        if group_of_tables is None or len(group_of_tables) == 0:

            return cls.compress_column_pattern_prompt(schema_dict, group_of_columns, is_use_sample_rows=is_use_sample_rows)
        prompt_text = ""
        for group_pattern in group_of_tables:
            tables_in_group = group_of_tables[group_pattern]
            # Keep only 1 table's details
            representative_table = tables_in_group[0]
            if representative_table not in schema_dict:
                # Just skip those table
                continue
            ##########################################################################################
            ### COMPRESS TABLES WITH PATTERN ### Like compress_table_pattern_prompt()
            ##########################################################################################
            # Build prompt for representative_table
            if len(tables_in_group) == 1:
                # Single table in group
                representative_table_fullname = schema_dict[representative_table].get("table_to_tablefullname",
                                                                                      representative_table)
                db_id = schema_dict[representative_table]["db_id"]
                prompt_text += f"DATABASE_NAME : {db_id} . Table: {representative_table_fullname}\n"
                # prompt_text += f"Table: {representative_table}\n"
            else:
                # Multiple tables in group
                group_pattern_str = group_pattern.split("_cluster")[0]
                db_id = schema_dict[representative_table]["db_id"]
                prompt_text += f"DATABASE_NAME : {db_id} .List of tables have same pattern {group_pattern_str} : {', '.join(tables_in_group)}\n"
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
            exp_values = schema_dict[representative_table].get("example_values", [])
            for idx, col_name in enumerate(schema_dict[representative_table]["columns_name"]):
                cols_pattern = colname_to_pattern.get(col_name, col_name)
                if cols_pattern in already_processed_cols_pattern:
                    continue
                already_processed_cols_pattern.add(cols_pattern)
                col_type = schema_dict[representative_table]["columns_type"][idx]
                col_desc = schema_dict[representative_table]["columns_description"][idx] if idx < len(
                    schema_dict[representative_table]["columns_description"]) else ""
                col_desc += cls.get_example_value_text(idx, exp_values,
                                                       schema_dict[representative_table][
                                                           "columns_name"])
                if cols_pattern in group_of_columns.get(representative_table, {}):
                    # This is a grouped column pattern
                    col_names_in_pattern = group_of_columns[representative_table][cols_pattern]
                    prompt_text += f"  Columns with pattern '{cols_pattern}': {', '.join(col_names_in_pattern)}\n"
                    prompt_text += f"   with representative Column for pattern '{cols_pattern}' : {col_name} (Type: {col_type}"
                    if col_desc and is_use_col_desc:
                        prompt_text += f", Description: {col_desc}"
                    prompt_text += ")\n"
                else:
                    # This is a single column (not grouped)
                    prompt_text += f"  Column: {col_name} (Type: {col_type}"
                    if col_desc and is_use_col_desc:
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
            if is_use_sample_rows:
                prompt_text += cls.get_sample_rows_prompt(schema_dict[representative_table]["sample_rows"])
            prompt_text += "\n"
        return prompt_text

    @classmethod
    def get_db_schema_text(cls, schema_dict, group_of_tables, group_of_columns,type, is_use_sample_rows: bool=False, is_use_col_desc: bool=True) -> str:
        """
        schema_dict :  output convert_db_schema_to_dict

        :param schema_dict:
        :param group_of_tables:
        :param group_of_columns:
        :param type:
        :return:
        """
        if type == "full":
            schema_text = cls.full_schema_prompt(schema_dict, is_use_sample_rows=is_use_sample_rows, is_use_col_desc=is_use_col_desc)
        elif type == "compress_table":
            schema_text = cls.compress_table_pattern_prompt(schema_dict, group_of_tables, is_use_sample_rows=is_use_sample_rows, is_use_col_desc=is_use_col_desc)
        elif type == "compress_column":
            schema_text = cls.compress_column_pattern_prompt(schema_dict, group_of_columns, is_use_sample_rows=is_use_sample_rows, is_use_col_desc=is_use_col_desc)
        elif type == "compact":
            schema_text = cls.compact_schema_prompt(schema_dict, group_of_tables, group_of_columns, is_use_sample_rows=is_use_sample_rows, is_use_col_desc=is_use_col_desc)
        else:
            raise NotImplementedError
        return schema_text

    @staticmethod
    def split_one_table(schema_dict: dict) -> list:
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


    @staticmethod
    def split_by_foreign_keys(schema_dict: dict) -> list:
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
    @classmethod
    def estimate_token_count(cls, table_info, is_use_col_desc: bool=True) -> int:
        # Simple estimation: token

        tokenizer = cls.get_instace().tokenizer

        input_text_est = str(table_info["columns_name"])*3 + " " + str(table_info["columns_type"])
        if is_use_col_desc:
            input_text_est += str(table_info["columns_description"])
        token_ids_est = tokenizer.encode(input_text_est, add_special_tokens=False)
        estimate_token = len(token_ids_est)
        return estimate_token

    @classmethod
    def merge_parts_by_token_limit(cls, schema_parts: list, token_limit: int = 50000 , is_use_col_desc: bool=True) -> list:
        """
        Merge small schema parts into larger parts based on token limit.
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


        random.shuffle(schema_parts)
        for schema_part in schema_parts:
            part_token_count = sum(cls.estimate_token_count(table_info, is_use_col_desc) for table_info in schema_part.values())
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

    @classmethod
    def split_by_prompting_size_limit(cls, schema_dict: dict, token_limit: int = 50000, is_use_col_desc: bool = True) -> list:
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

        estimate_token_table2token_count = {table_name: cls.estimate_token_count(table_info, is_use_col_desc) for table_name, table_info in
                                            schema_dict.items()}
        estimate_token_table2token_count_sorted = dict(
            sorted(estimate_token_table2token_count.items(), key=lambda item: item[1], reverse=False))

        for table_name, table_token_count in estimate_token_table2token_count_sorted.items():
            table_info = schema_dict[table_name]
            if current_token_count + table_token_count > token_limit:
                # Start a new part if current part exceeds token limit
                if current_part:
                    schema_parts.append(current_part)
                current_part = {table_name: table_info}
                current_token_count = table_token_count
            else:
                # if not exceed token limit, add table to current part
                current_part[table_name] = table_info
                current_token_count += table_token_count

        if current_part:
            # append the last part
            schema_parts.append(current_part)
        # random.shuffle(schema_parts)
        return schema_parts

    @classmethod
    def split_schema(cls, schema_dict : dict, split_type : str , token_limit : int = 50000, is_use_col_desc: bool=True) -> list:
        """
        schema_dict :  output  convert_db_schema_to_dict
        :param schema_dict:
        :param split_type:
        :param token_limit:
        :return:
        """
        if split_type == "one_table":
            schema_parts = cls.split_one_table(schema_dict)
        elif split_type == "foreign_key":
            schema_parts = cls.split_by_foreign_keys(schema_dict)
        elif split_type == "prompting_size_limit":
            schema_parts = cls.split_by_prompting_size_limit(schema_dict, token_limit=token_limit, is_use_col_desc= is_use_col_desc)
        else:
            raise ValueError(f"Unknown split type: {split_type}")
        return schema_parts


class TextInforManager:
    """

    """
    def __init__(self, question_id: str, schema_dict: dict, group_of_tables: dict , group_of_columns: dict, prompt_type: str="compact",
                 is_use_sample_rows: bool=False, is_use_col_desc: bool=True):
        self.schema_dict = schema_dict
        if question_id.startswith('local'):
            self.dialect1 = 'SQLite'
            self.api_type = "sqlite"
            self.dialect2 = self.get_dialect2_local()
        elif question_id.startswith('bq') or question_id.startswith('ga'):
            self.dialect1 = 'Goole_BigQuery'
            self.api_type = "bigquery"
            self.dialect2 = self.get_dialect2_bq()
        elif question_id.startswith('sf'):
            self.dialect1 = 'Snowflake'
            self.api_type = "snowflake"
            self.dialect2 = self.get_dialect2_sf()
        elif question_id.startswith('mysql'):
            self.dialect1 = 'MySQL'
            self.api_type = "mysql"
            self.dialect2 = self.get_dialect2_mysql()
        else:
            raise NotImplementedError
        self.schema_text = DatabaseSchemaManager.get_db_schema_text(
            self.schema_dict,
            group_of_tables=group_of_tables,
            group_of_columns=group_of_columns,
            type=prompt_type, is_use_sample_rows=is_use_sample_rows,
            is_use_col_desc = is_use_col_desc
        )
        token_ids = DatabaseSchemaManager.get_instace().tokenizer.encode(self.schema_text, add_special_tokens=False)
        self.estimate_num_token_schema_text = len(token_ids)

    def get_dialect2_local(self):
        dialect2 = "SQLite. All table and column names must be enclosed in ` backticks."
        dialect2 += self.get_notice_with_dataset(DatabaseSchemaManager.get_instace().dataset_name)
        """
        Source from Alpha-SQL + CHESS + https://github.com/DMIRLAB-Group/DSR-SQL/blob/main/DSR_Lite/utils/Prompt.py
        """
        dialect2 += """Database admin instructions (voliating any of the following will result is punishble to death!):
1. **SELECT Clause:** 
    - Only select columns mentioned in the user's question. 
    - Avoid unnecessary columns or values.
    - Never use `|| ' ' ||` or any other method to concatenate strings in the `SELECT` clause. 
    - Do not concat column in select. If needed, return columns separately or try to find another column that already has the desired information.
2. **Aggregation (MAX/MIN):**
    - Always perform JOINs before using MAX() or MIN().
3. **ORDER BY with Distinct Values:**
    - Use `GROUP BY <column>` before `ORDER BY <column> ASC|DESC` to ensure distinct values.
4. **Handling NULLs:**
    - If a column may contain NULL values, use `JOIN` or `WHERE <column> IS NOT NULL`.
5. **FROM/JOIN Clauses:**
    - Only include tables essential to answer the question.
6. **Strictly Follow External Knowledge:**
    - Adhere to all provided External Knowledge.
7. **Thorough Question Analysis:**
    - Address all conditions mentioned in the question.
8. **DISTINCT Keyword:**
    - Use `SELECT DISTINCT` when the question requires unique values (e.g., IDs, URLs). 
9. **Column Selection:**
    - Carefully analyze column descriptions and External Knowledge to choose the correct column when similar columns exist across tables.
10. **String Concatenation:**
    - Never use `|| ' ' ||` or any other method to concatenate strings in the `SELECT` clause. 
11. **SQLite Functions Only:**
    - Use only functions available in SQLite.
12. **Date Processing:**
    - Utilize `STRFTIME()` for date manipulation (e.g., `STRFTIME('%Y', SOMETIME)` to extract the year).
13. **JOIN Preference:**
    - Prioritize `INNER JOIN` over nested `SELECT` statements. Do not use `CROSS JOIN` or `LEFT/RIGHT JOIN`.
"""

        return dialect2
    def get_dialect2_bq(self):
        dialect2 = "Goole BigQuery."
        # dialect2 = "With table have same prefix like `ga_sessions_YYYYMMDD`, you could use wildcard table name like `ga_sessions_*` to query all those tables."
        """
        Source from https://github.com/DMIRLAB-Group/DSR-SQL/blob/main/DSR_Lite/utils/Prompt.py
        """
        dialect2 += """Database admin instructions (voliating any of the following will result is punishble to death!):
1.  **Case Sensitivity and Quotation Marks:**
    *   BigQuery is **case-insensitive** for identifiers (e.g., table and column names).
    *   Use **backticks (`` ` ``)** to enclose identifiers if they contain special characters (like spaces or hyphens) or are reserved keywords. Example: `` `my-project.my_dataset.my_table` ``, `` `group` ``.

2.  **Optimized and Idiomatic SQL:**
    *   Write highly optimized and idiomatic BigQuery SQL. Leverage BigQuery's architecture by using `WITH` clauses (CTEs) to structure complex queries.
    *   To control costs and improve performance, **avoid `SELECT *`**. Explicitly select only the columns you need, especially on large tables.

3.  **Querying Multiple Tables (UNION ALL):**
    *   Instead of manually using `UNION ALL` for tables with a common naming pattern (e.g., date-sharded tables), use **wildcard tables**. 
    *   Syntax: `SELECT col1, col2 FROM \`DATABASE_NAME.dataset.table_prefix_*\` WHERE _TABLE_SUFFIX BETWEEN 'start_suffix' AND 'end_suffix' AND ...`
    *   Note: The SQL syntax involving wildcard tables must be written in **exactly this one and only** format — **no alternative representations are allowed**!

4.  **Working with Nested/Repeated Data (ARRAY & STRUCT):**
    *   To work with nested data (often in `ARRAY` of `STRUCT` format), use the **`UNNEST`** operator.
    *   Example:
        `SELECT t.id, event.name AS event_name`
        `FROM \`DATABASE_NAME.dataset.events\` AS t, UNNEST(t.event_params) AS event;`
    *   If JSON data is stored as a `STRING`, use BigQuery's JSON functions:
        `SELECT JSON_EXTRACT_SCALAR(json_column, '$.key_name') FROM ...`
    *   If the structure is unknown, first explore a single record's nested array:
        `SELECT event_params FROM \`DATABASE_NAME.dataset.events\` LIMIT 1;`

5.  **Fuzzy String Matching:**
    *  For fuzzy, case-insensitive matching, use the `LOWER()` function on both sides of the comparison.
        `WHERE LOWER(column_name) LIKE '%target_string%'`
    *   Replace spaces with `%` in patterns, e.g., `LOWER(column_name) LIKE '%meat%lovers%'`.

6.  **Ordering and NULL Handling:**
    *   When using descending order, you can explicitly control how NULLs are sorted:
        `ORDER BY column_name DESC NULLS LAST` (or `NULLS FIRST`)

7.  **DISTINCT Keyword:**
    *   This is standard SQL logic and applies fully to BigQuery. Use `DISTINCT` when you need to count unique entities.
    *   **Count total records** → `COUNT(*)`
    *   **Count distinct courses** → `COUNT(DISTINCT course_id)`
    *   **Count distinct students** → `COUNT(DISTINCT student_id)`

8.  **Special Cases & Functions:**
    *   **Geospatial Functions:** The `ST_DISTANCE` function calculates the shortest distance between two geospatial objects. In BigQuery, it returns the distance in **meters** for `GEOGRAPHY` objects. Syntax: `ST_DISTANCE(geography_1, geography_2)`.
    *   **Misleading Evidence:** In rare cases, the provided evidence might be incomplete. If a query on an expected column returns no results, consider that the data may not be populated as expected. Be prepared to explore other related columns to achieve the task's goal.

        """
        return dialect2
    def get_dialect2_sf(self):
        dialect2 = 'Snowflake. Column names must be enclosed in double quotes, and table names must not be enclosed.'
        dialect2 += '\nTable name in Snowflake contain DATABASE_NAME.SCHEMA.TABLE format or DATABASE_NAME.SCHEMA."TABLE" format. DATABASE_NAME and SCHEMA_NAME are oustide the double quotes'
        dialect2 += "Table names and column names must enclosed in double quotes, do not use backticks or single quotes."

        dialect2 += "Snowflake do not have prefix for table names like BigQuery. You must use full table name in Snowflake queries. e.g. Instead using FROM GA360.GOOGLE_ANALYTICS_SAMPLE.\"GA_SESSIONS_201701*\" , need to list all posible tables: FROM ( GA360.GOOGLE_ANALYTICS_SAMPLE.\"GA_SESSIONS_20170101\" UNION ALL GA360.GOOGLE_ANALYTICS_SAMPLE.\"GA_SESSIONS_20170102\" )\n"
        # Source from ReFoRCE
        condition_onmit_tables = ["-- Include all", "-- Omit", "-- Continue", "-- Union all", "-- ...", "-- List all", "-- Replace this", "-- Each table", "-- Add other"]
        dialect2 +=  f"When performing a UNION operation on many tables, ensure that all table names are explicitly listed. Union first and then add condition and selection. e.g. SELECT \"col1\", \"col2\" FROM (TABLE1 UNION ALL TABLE2) WHERE ...; Don't write sqls as (SELECT col1, col2 FROM TABLE1 WHERE ...) UNION ALL (SELECT col1, col2 FROM TABLE2 WHERE ...); Don't use {condition_onmit_tables} to omit any table.\n"
        # dialect2 += "For columns in json nested format: e.g. SELECT t.\"column_name\", f.value::VARIANT:\"key_name\"::STRING AS \"abstract_text\" FROM PATENTS.PATENTS.PUBLICATIONS t, LATERAL FLATTEN(input => t.\"json_column_name\") f; DO NOT directly answer the task and ensure all column names are enclosed in double quotations."
        dialect2 += "For nested columns like event_params, when you don't know the structure of it, first watch the whole column: SELECT f.value FROM table, LATERAL FLATTEN(input => t.\"event_params\") f;\n"
        # dialect2 += "Don't directly match strings if you are not convinced. Use fuzzy query first: WHERE str ILIKE \"%target_str%\" For string matching, e.g. meat lovers, you should use % to replace space. e.g. ILKIE %meat%lovers%.\n"

        """
        Source from https://github.com/DMIRLAB-Group/DSR-SQL/blob/main/DSR_Lite/utils/Prompt.py
        """
        dialect2 += """Database admin instructions (voliating any of the following will result is punishble to death!):
1. **Case Sensitivity and Quotation Marks:**
   * Snowflake is case-sensitive. Always enclose **all DB、table and column names in double quotes (`"`)** to avoid errors.

2. **Optimized and Idiomatic SQL:**
   * Write highly optimized and idiomatic Snowflake SQL. Be aware of the **limitations of Snowflake's query optimizer**, especially regarding correlated subqueries and deeply nested array/object structures.

3. **Working with Nested JSON Columns:**
   * To extract values from nested JSON, use `LATERAL FLATTEN` and proper casting:
     Example:
     `SELECT t."column_name", f.value::VARIANT:"key_name"::STRING AS "abstract_text"`
     `FROM DATABASE_NAME."schema"."table" t, LATERAL FLATTEN(input => t."json_column_name") f;`
   * Always enclose both **column names and nested keys** in double quotes.

4. **Fuzzy String Matching:**
   * Avoid strict string matching unless you're confident in the exact value. Prefer fuzzy matching:
     `WHERE str ILIKE '%target_str%'`
   * Replace spaces with `%` in patterns, e.g., `ILIKE '%meat%lovers%'`.

5. **Ordering and NULL Handling:**
   * When using descending order, explicitly handle NULLs:
     `ORDER BY xxx DESC NULLS LAST`
   * For geospatial queries, use `ST_DISTANCE` to calculate the distance between two geographic points accurately.

6. **DISTINCT Keyword:**
  Whether to use DISTINCT depends on one point: if you are calculating the number of [entities], it is needed; if you are calculating [frequency], it is not needed. For example,
    * **Count the number of course view records** → `COUNT(*)` (each row represents one view record)
    * **Count the number of distinct courses viewed** → `COUNT(DISTINCT course_id)` (deduplicated by course)
    * **Count the number of distinct students who viewed courses** → `COUNT(DISTINCT student_id)` (deduplicated by student)

7. **special cases(Large databases are inevitable)**
    * The ST_DISTANCE function calculates the shortest distance between two geospatial objects. Syntax: ST_DISTANCE(object1, object2) Input object types: GEOGRAPHY: calculates spherical distance (great-circle distance) in meters, suitable for longitude/latitude coordinates on the Earth's surface; GEOMETRY: calculates planar distance (Euclidean distance) in units defined by the coordinate system, suitable for 2D Cartesian coordinate systems. Key considerations: When using GEOMETRY type, both objects must have the same SRID (Spatial Reference System Identifier); returns NULL if any input object is NULL.
    * In very rare cases, the evidence section may contain misleading information, leading to additional situations such as empty query results (for example, if a certain column exists but the corresponding information cannot be found [i.e., the column is not enabled], in which case you can achieve the same goal by exploring other columns).

        """
        return dialect2

    def get_notice_with_dataset(self, dataset_name: str):
        notice_text = ''
        if dataset_name == "bird":
            notice_text = f"""With string have special characters like single quotes : Kevin's drawing , you must use double quotes to enclose the string literal, e.g., `text` = "Kevin's drawing".
If text contain double quotes like Kevin"s drawing, you need to use single quotes to enclose the string literal. e.g., `text` = 'Kevin"s drawing'.
"""
        return notice_text
    def get_fix_sf_call_use_db_str(self):
        fix_str = "If executing has error: Cannot perform SELECT. This session does not have a current database. Call 'USE DATABASE', or use a qualified name."\
        "This error because lack of database name in FROM clause in Snowflake query " \
         "You need to add DATABASE_NAME or DB_ID in FROM clause to specify the database, so make sure that table have format DATABASE_NAME.SCHEMA_NAME.TABLE_NAME"
        return fix_str

    def get_dialect2_mysql(self):
        dialect2 = "MySQL. All table and column names must be enclosed in ` backticks."
        dialect2 += """Database admin instructions (voliating any of the following will result is punishble to death!):
1. Identifier case & quoting:
   - Identifier case behavior can depend on OS/config (e.g., table-name settings), so keep naming consistent.
   - Use backticks (`) to quote identifiers when needed (reserved words/special chars), e.g., SELECT `group` FROM `my_table`;

2. SELECT clause (no extra columns):
   - Do not concat column in select. If needed, return columns separately or try to find another column that already has the desired information.

3. FROM/JOIN scope:
   - Only include tables essential to answer the question.
   - Prefer INNER JOIN for required relationships; use LEFT JOIN only if the question requires retaining unmatched rows.

4. Aggregation (MAX/MIN):
   - Perform JOINs before applying MAX()/MIN() so the aggregate is computed on the correct grain.
   - When selecting non-aggregated columns with aggregates, GROUP BY all non-aggregated selected columns (or refactor using a derived table).

5. ORDER BY with distinct values:
   - If the question wants distinct values sorted, prefer:
     SELECT col FROM ... GROUP BY col ORDER BY col ASC|DESC;
   - Or use SELECT DISTINCT col ... ORDER BY col when it matches the requested semantics.

6. Handling NULLs:
   - If NULLs may break logic, filter explicitly: WHERE col IS NOT NULL.
   - Use NULL-safe equality when needed: a <=> b.

7. Thorough question analysis:
   - Apply every condition from the question using WHERE/JOIN/HAVING as appropriate (time range, location, status, etc.).

8. DISTINCT keyword usage:
   - Use DISTINCT only when counting unique entities (IDs/users/URLs).
   - Use COUNT(*) for counting rows/events; use COUNT(DISTINCT id) for unique entities.

9. Column selection with ambiguous names:
   - Qualify columns with table aliases and choose the correct column using schema descriptions and relationships.

10. Date processing (MySQL functions only):
   - Use YEAR(date_col), MONTH(date_col), DATE(date_col), DATE_FORMAT(date_col, '%Y-%m'), and INTERVAL arithmetic.
   - Prefer sargable ranges for performance, e.g. date_col >= '2017-01-01' AND date_col < '2018-01-01' instead of YEAR(date_col)=2017 on large tables.

11. Avoid non-MySQL syntax:
   - Do not use BigQuery UNNEST, Snowflake LATERAL FLATTEN, or SQLite STRFTIME().
   - Use MySQL JSON functions/operators if needed: JSON_EXTRACT, JSON_UNQUOTE, ->, ->>.
"""
        return dialect2

