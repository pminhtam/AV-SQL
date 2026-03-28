"""


Group columns by patterns and meaning in Spider2 dataset.

How to group :
- Similar prefix/suffix
- Sme type
- Meaning have small levenshtein distance

EX:
- table_fullname": "bigquery-public-data.idc_v14.tcga_biospecimen_rel9
    -Columns : max_percent_monocyte_infiltration, max_percent_necrosis,max_percent_neutrophil_infiltration ,max_percent_normal_cells,max_percent_stromal_cells,
- table_fullname": : bigquery-public-data.census_bureau_acs.cbsa_2015_5yr
    - Columns : rent_40_to_50_percent ; rent_35_to_40_percent ;rent_20_to_25_percent
    - income_20000_24999 ;income_25000_29999 ;income_40000_44999 ;income_100000_124999

Input :
Result of `step1_load_schema_infor.py`
File `tables_preprocessed.json` have structure like Spider/BIRD dataset


"""

import argparse
import glob
import json
import os
import os.path as osp
import re
from collections import defaultdict
from typing import List, Dict, Tuple
import nltk
import time

def get_table2group_mapping(group_of_tables):
    """
    Get mapping from table names to their respective groups.

    Args:
        group_of_tables (dict): dich mapping groups to list of table names.

    Returns:
        dict: Dictionary mapping table names to their groups.
    """
    table2group = {}
    for group_name in group_of_tables:
        for table_name in group_of_tables[group_name]:
            table2group[table_name] = group_name
    return table2group

def convert_db_schema_to_dict(db_schema_list):
    """
    Convert database schema list to a dictionary for easier access.

    Args:
        db_schema_list (list): List of database schemas.

    Returns:
        dict: Dictionary mapping database IDs to their schemas.
    """
    db_schema_dict = {}
    for db_schema in db_schema_list:
        db_id = db_schema['db_id']
        db_schema_dict[db_id] = {}
        for idx, column_name_original in enumerate(db_schema['nested_column_names_original']):
            idx_table , column_name = column_name_original
            # table_name = db_schema['table_names'][idx_table]  # Chứa cả thông tin về db_id nữa
            table_name = db_schema['table_names_original'][idx_table]   # Chỉ chứa tên table thôi. Dùng tên table cho dễ xử lý
            column_type = db_schema['nested_column_types'][idx]
            column_description = db_schema['nested_column_descriptions'][idx][1] if len(db_schema['nested_column_descriptions']) > idx else ""
            if table_name not in db_schema_dict[db_id]:
                db_schema_dict[db_id][table_name] = {
                    "nested_columns_name": [],
                    "nested_columns_type": [],
                    "nested_columns_description": [],
                }
            db_schema_dict[db_id][table_name]["nested_columns_name"].append(column_name)
            db_schema_dict[db_id][table_name]["nested_columns_type"].append(column_type)
            db_schema_dict[db_id][table_name]["nested_columns_description"].append(column_description)

    return db_schema_dict



# ============================================================================
# STEP 1: Pattern Detection
# perplexity.ai : Claude Sonnet 4.5 Thinking
# ============================================================================

def detect_name_pattern(name: str) -> str:
    """
    Automatically detect the pattern of a name by abstracting variable parts.

    CORRECTED - Now handles mixed NUM and VAR patterns!

    Examples:
        income_20000_24999 ;income_25000_29999 ;income_40000_44999 ;income_100000_124999 - > income_{NUM}_{NUM}
    """
    pattern = name
    # Step 1: Replace all digit sequences with {NUM}
    pattern = re.sub(r'\d+', '{NUM}', pattern)
    # Just using pattern with numbers replaced
    return pattern


def group_names_by_pattern(names: List[str]) -> Dict[str, List[str]]:
    """
    Group a list of names based on their structural patterns.
    Works with any naming convention without requiring predefined patterns.

    Returns:
        Dictionary mapping each detected pattern to list of matching names
    """
    groups = defaultdict(list)

    for name in names:
        try:
            pattern = detect_name_pattern(name)
        except:
            import pdb; pdb.set_trace()
        groups[pattern].append(name)
    for pattern in list(groups.keys()):
        # Just keep groups with at least 10 columns to reduce noise
        if len(groups[pattern]) < 10:
            del groups[pattern]
    return dict(groups)

def check_groups_by_type_and_description(
    pattern_groups: Dict[str, List[str]],
    nested_column_names_original: List[str],
    nested_column_types: List[str],
    column_descriptions: List[str]
) -> Dict[str, List[str]]:
    """
    Further validate and refine groups by checking column types and descriptions.

    Args:
        pattern_groups (Dict[str, List[str]]): Groups of column names by detected patterns.
        nested_column_names_original (List[str]): List of column names
        nested_column_types (List[str]): List of column types corresponding to the original column names.
        column_descriptions (List[str]): List of column descriptions corresponding to the original column names.

    Returns:
        Refined groups after checking types and descriptions.
    """
    refined_groups = {}

    for pattern, names in pattern_groups.items():
        type_set = set()
        description_list = list()
        for name in names:
            index = nested_column_names_original.index(name)
            col_type = nested_column_types[index] if index < len(nested_column_types) else None
            col_desc = column_descriptions[index] if index < len(column_descriptions) else None
            if col_type:
                type_set.add(col_type)
            # if col_desc:
            #     description_set.add(col_desc)
            description_list.append(col_desc)

        # If all columns in the group have the same type and similar descriptions, keep the group
        # if len(type_set) == 1 and len(description_set) <= 3:  # Allowing some variation in descriptions
        edit_distances = [ nltk.edit_distance(description_list[0], desc) for desc in description_list[1:] if desc is not None ]
        avg_edit_distance = sum(edit_distances) / len(edit_distances)   #  levenshtein distance  descriptions
        len_column_names = [len(col_name) for col_name in names]
        threshold_avg_len_column_name =  0.7 * sum(len_column_names) / len(len_column_names)
        if len(type_set) == 1 and avg_edit_distance < threshold_avg_len_column_name:  # Allowing some variation in descriptions
            refined_groups[pattern] = names
            print(
                f"Pattern: {pattern} - Types: {type_set} - Avg Edit Distance: {avg_edit_distance} - Thres Avg Len Column Name: {threshold_avg_len_column_name}")
            # import pdb; pdb.set_trace()
    return refined_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--table_file_path', default='spider2_schema_processing/preprocessed_data_compress/spider2-lite/tables_preprocessed.json', type=str, help='the table file from step 1')
    args = parser.parse_args()
    start_time = time.time()
    output_dir = os.path.dirname(args.table_file_path)
    db_schema_list = json.load(open(args.table_file_path, 'r'))
    db_schema_dict = convert_db_schema_to_dict(db_schema_list)
    max_group = 0
    num_group_list = {}
    db_schema_group_columns = {}
    for db_id in db_schema_dict:
        for table in db_schema_dict[db_id]:
            """
            Group columns by patterns and meaning
            """
            nested_column_names = db_schema_dict[db_id][table]['nested_columns_name']
            # nested
            nested_column_types = db_schema_dict[db_id][table]['nested_columns_type']
            column_descriptions = db_schema_dict[db_id][table]['nested_columns_description']
            # Step 1 : First detect and group column  name with same pattern
            pattern_groups = group_names_by_pattern(nested_column_names)
            # Step 2 : Check column  types and descriptions whether they are the same or not
            # print(len(pattern_groups))
            refined_groups = check_groups_by_type_and_description(
                pattern_groups,
                nested_column_names,
                nested_column_types,
                column_descriptions
            )
            """
            refined_groups : format { pattern1 : [ col_name1, col_name2, ...], pattern2: [ col_name3, col_name4, ...] }
            
            """
            # print(f"Database: {db_id} - Table {table} - Original Groups: {len(pattern_groups)} - Refined Groups: {len(refined_groups)}")
            # if len(refined_groups) > 10:
            #     import pdb; pdb.set_trace()
            max_group = max(max_group, len(refined_groups))
            if len(refined_groups) in num_group_list:
                num_group_list[len(refined_groups)] += 1
            else:
                num_group_list[len(refined_groups)] = 1

            if db_id not in db_schema_group_columns:
                db_schema_group_columns[db_id] = {}
            db_schema_group_columns[db_id][table] = refined_groups
    print("Number of groups distribution: ", num_group_list)
    print(f"Max groups in any table: {max_group}")

    for item in db_schema_list:
        db_id = item['db_id']
        item['group_of_columns'] = db_schema_group_columns.get(db_id, {})
    # with open(osp.join(output_dir, 'tables_preprocessed_step2_group_columns.json'), 'w') as f:
    with open(osp.join(output_dir, 'tables_preprocessed_step2_group_columns_with_example_values.json'), 'w') as f:
        json.dump(db_schema_list, f, indent=4)
    print(f"Completed grouping columns by patterns. Time taken: {time.time() - start_time:.5f} seconds.")

"""
python database/spider2/schema_processing/step2_group_column_by_pattern.py --table_file_path spider2_schema_processing/preprocessed_data_compress/spider2-lite/tables_preprocessed.json
python database/spider2/schema_processing/step2_group_column_by_pattern.py --table_file_path spider2_schema_processing/preprocessed_data_compress/spider2-lite/tables_preprocessed_with_example_values.json
Completed grouping columns by patterns. Time taken: 522.68095 seconds.
Number of groups distribution:  {0: 6651, 1: 43, 2: 793, 13: 2, 5: 2, 16: 1, 9: 1, 4: 514, 3: 89, 6: 1}
Max groups in any table: 16
Completed grouping columns by patterns. Time taken: 503.52454 seconds.

"""

"""
python database/spider2/schema_processing/step2_group_column_by_pattern.py --table_file_path spider2_schema_processing/preprocessed_data_compress/spider2-snow/tables_preprocessed.json
python database/spider2/schema_processing/step2_group_column_by_pattern.py --table_file_path spider2_schema_processing/preprocessed_data_compress/spider2-snow/tables_preprocessed_with_example_values.json
Number of groups distribution:  {0: 6415, 4: 1302, 3: 89, 1: 43, 5: 1, 2: 5, 6: 1, 13: 2, 9: 1, 16: 1}
Max groups in any table: 16
Completed grouping columns by patterns. Time taken: 21.58908 seconds
"""