"""

Load json schema information from a file and process it to extract relevant details.
Group all information and save it into a single json file.

Need add some information like :
- DDL statements
- table_fullname


Source : https://github.com/xlang-ai/Spider2/blob/main/spider2-lite/baselines/utils/utils.py

This script process:
- Load the json schema information from files
- Detect table name patterns and group tables accordingly based on patterns
- Re-cluster tables within each pattern group based on column similarity
- Save the processed schema information into a single json file

Output :
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
import matplotlib.pyplot as plt






def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def db_stats(db_stats_list):
    total_table_count = sum(item['db_stats']["No. of tables"] for item in db_stats_list)
    total_column_count = sum(item['db_stats']["No. of columns"] for item in db_stats_list)
    total_nested_column_count = sum(item['db_stats']["No. of nested columns"] for item in db_stats_list)
    total_avg_column_per_table = sum(item['db_stats']["Avg. No. of columns per table"] for item in db_stats_list)

    num_dbs = len(db_stats_list)

    avg_table_count_per_db = total_table_count / num_dbs if num_dbs > 0 else 0
    avg_total_column_count_per_db = total_column_count / num_dbs if num_dbs > 0 else 0
    avg_total_nested_column_count_per_db = total_nested_column_count / num_dbs if num_dbs > 0 else 0
    avg_avg_column_per_table_per_db = total_avg_column_per_table / num_dbs if num_dbs > 0 else 0

    print(f"No. of db: {num_dbs}")
    print(f"Average No. of tables across all database: {avg_table_count_per_db:.2f}")
    print(f"Average No. of columns across all Database: {avg_total_column_count_per_db:.2f}")
    print(f"Average No. of nested columns across all Database: {avg_total_nested_column_count_per_db:.2f}")
    print(f"Average Avg. No. of columns per table across all Databases: {avg_avg_column_per_table_per_db:.2f}")


def db_stats_bar_chart(db_stats_list):

    db_ids = [item['db_id'] for item in db_stats_list]
    table_counts = [item['db_stats']["No. of tables"] for item in db_stats_list]
    total_columns = [item['db_stats']["No. of columns"] for item in db_stats_list]
    avg_columns = [item['db_stats']["Avg. No. of columns per table"] for item in db_stats_list]

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    # Plot for table count
    ax[0].bar(db_ids, table_counts, color='blue')
    ax[0].set_title("No. of tables")
    ax[0].set_ylabel("#")
    ax[0].tick_params(axis='x', rotation=45)
    for label in ax[0].get_xticklabels():
        label.set_ha('right')

    # Plot for total column count
    ax[1].bar(db_ids, total_columns, color='green')
    ax[1].set_title("No. of columns")
    ax[1].set_ylabel("#")
    ax[1].tick_params(axis='x', rotation=45)
    for label in ax[1].get_xticklabels():
        label.set_ha('right')

    # Plot for average columns per table
    ax[2].bar(db_ids, avg_columns, color='orange')
    ax[2].set_title("Avg. No. of columns per table")
    ax[2].set_ylabel("#")
    ax[2].tick_params(axis='x', rotation=45)
    for label in ax[2].get_xticklabels():
        label.set_ha('right')

    plt.tight_layout()
    # plt.show()
    # plt.savefig(osp.join(proj_dir, 'db_statistic.png'))






def detect_name_pattern_old(name: str) -> str:
    """
    Automatically detect the pattern of a name by abstracting variable parts.

    Pattern Detection Rules:
    - Sequences of digits -> {NUM}
    - Variable text parts after common prefix -> {VAR}
    - Preserves structural information (underscores, prefixes)

    Examples:
        'ga_sessions_20170303' -> 'ga_sessions_{NUM}'
        'census_tracts_california' -> 'census_tracts_{VAR}'
        'user_activity_2024_jan' -> 'user_activity_{NUM}_jan' -> bi sai ne
    """
    pattern = name

    # Replace all digit sequences with {NUM}
    pattern = re.sub(r'\d+', '{NUM}', pattern)

    # Handle text-based variable parts
    if '{NUM}' not in pattern:
        parts = pattern.split('_')

        if len(parts) >= 3:
            # Keep first 2 parts as stable prefix, rest as variable
            prefix = '_'.join(parts[:2])
            pattern = f"{prefix}_{{VAR}}"
        elif len(parts) == 2:
            # Two-part name: prefix + variable
            pattern = f"{parts[0]}_{{VAR}}"

    return pattern

# ============================================================================
# STEP 1: Pattern Detection
# perplexity.ai : Claude Sonnet 4.5 Thinking
# ============================================================================

def detect_name_pattern(name: str) -> str:
    """
    Automatically detect the pattern of a name by abstracting variable parts.

    CORRECTED - Now handles mixed NUM and VAR patterns!

    Examples:
        'ga_sessions_20170303' -> 'ga_sessions_{NUM}'
        'census_tracts_california' -> 'census_tracts_{VAR}'
        'user_activity_2024_jan' -> 'user_activity_{NUM}_{VAR}' ✓
        "symptom_search_sub_region_1_daily" -> "symptom_search_{NUM}_{VAR}_cluster_0" wrong . fixed
        (chưa chuẩn vì phần stable prefix chỉ giữ 2 phần đầu, đáng lẽ phải đến symptom_search_sub_region_)
        "symptom_search_country_daily" -> "symptom_search_{VAR}_cluster_0" wrong
        "irs_990_ez_2016" -> irs_{NUM}_{NUM}_{VAR}_cluster_2 : wrong vì sai pattern
    """
    pattern = name
    num_stable_parts = 2  # Number of stable prefix parts to keep
    # Step 1: Replace all digit sequences with {NUM}
    pattern = re.sub(r'\d+', '{NUM}', pattern)

    # Step 2: Handle variable text parts (even if {NUM} exists!)
    parts = pattern.split('_')
    num_stable_parts = max(num_stable_parts, parts.index('{NUM}') if '{NUM}' in parts else 0)
    if len(parts) >= 3:
        # Keep first num_stable_parts ( = 2 ) parts as stable prefix
        prefix = '_'.join(parts[:num_stable_parts])
        suffix_parts = parts[num_stable_parts:]

        # Process suffix: keep {NUM}, collapse text to {VAR}
        suffix_tokens = []
        # has_text_vars = False

        for part in suffix_parts:
            if part == '{NUM}':
                suffix_tokens.append(part)
            elif re.match(r'^[a-z]+$', part, re.IGNORECASE):
                # Found text variable - will add single {VAR} later
                # has_text_vars = True
                suffix_tokens.append('{VAR}')   # fixed "irs_990_ez_2016" -> irs_{NUM}_{NUM}_{VAR}_cluster_2 : wrong vì sai pattern
            else:
                # Keep mixed patterns (e.g., 'v{NUM}')
                suffix_tokens.append(part)

        # Add {VAR} placeholder if we found any text variables
        # "irs_990_ez_2016" -> irs_{NUM}_{NUM}_{VAR}_cluster_2 : wrong vì sai pattern
        # if has_text_vars:
        #     suffix_tokens.append('{VAR}')

        pattern = '_'.join([prefix] + suffix_tokens)

    elif len(parts) == 2:
        # Two-part names: keep numbers, replace text
        if parts[1] != '{NUM}' and re.match(r'^[a-z]+$', parts[1], re.IGNORECASE):
            pattern = f"{parts[0]}_{{VAR}}"

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
        pattern = detect_name_pattern(name)
        groups[pattern].append(name)
    # groups_new = defaultdict(list)
    # for pattern in groups:
    #     # Xoa cac pattern chi co 1 table
    #     if len(groups[pattern]) > 1:
    #         groups_new[pattern] = groups[pattern]
    # groups = groups_new
    return dict(groups)

# ============================================================================
# STEP 2: Column-Based Re-clustering (IMPROVED VERSION)
# perplexity.ai : Claude Sonnet 4.5 Thinking
# ============================================================================

# def regroup_by_column(table_name_to_group_all, all_schema_dict):
#     table_name_to_group_all_new = {}
#     for db_id in table_name_to_group_all:
#         table_name_to_group = table_name_to_group_all[db_id]
#         table_name_to_group_new = {}
#         for pattern in table_name_to_group:
#             tables_in_group = table_name_to_group[pattern]
#             if len(tables_in_group) <= 1:  # chi co 1 table trong group
#                 table_name_to_group_new[tables_in_group[0]] = [tables_in_group[0]]
#                 continue
#             # Check column names in the grouped tables
#             tables_to_remove = set()
#             reference_table = tables_in_group[0]
#             reference_columns = set(all_schema_dict[db_id][reference_table])
#             for other_table in tables_in_group[1:]:
#                 other_columns = set(all_schema_dict[db_id][other_table])
#                 if reference_columns == other_columns:  # identical columns
#                     # Nếu table có cùng column thì để yên cho nó nằm trong nhóm
#                     continue
#                 else:
#                     # table khác nhau về column
#                     print(f"Different columns in db {db_id} for tables {reference_table} and {other_table}")
#                     # table_name_to_group[other_table] = [other_table]  # Không gộp nữa
#                     tables_to_remove.add(other_table)
#                     table_name_to_group_new[other_table] = [other_table]
#             # Keep only tables with identical columns
#             # Copy lại những table cùng column với table tham chiếu
#             tables_to_keep = [t for t in tables_in_group if t not in tables_to_remove]
#             if len(tables_to_keep) > 0:
#                 table_name_to_group_new[reference_table] = tables_to_keep
#             for t in tables_to_remove:
#                 table_name_to_group_new[t] = [t]
#         table_name_to_group_all_new[db_id] = table_name_to_group_new
#     return table_name_to_group_all_new


def get_column_signature(columns: List[str]) -> Tuple[str, ...]:
    """Create a hashable signature from columns (sorted for consistency)."""
    return tuple(sorted(columns))


def cluster_tables_by_columns(
        pattern_groups: Dict[str, List[str]],
        schema_dict: Dict[str, List[str]],
        verbose: bool = False
) -> Dict[str, List[str]]:
    """
    Re-cluster tables within each pattern group based on column similarity.

    KEY IMPROVEMENT: Uses clustering instead of reference comparison.
    This properly handles multiple distinct column sets within one pattern.

    Args:
        pattern_groups: Dict of pattern -> list of table names
        schema_dict: Dict of table_name -> list of column names
        verbose: If True, print split information

    Returns:
        Dict mapping group_key -> list of table names
    """
    final_groups = {}

    for pattern, tables in pattern_groups.items():
        if len(tables) == 1:
            # Single table - no comparison needed
            table_name = tables[0]
            final_groups[table_name] = [table_name]
            continue

        # Cluster by column similarity
        column_clusters = defaultdict(list)

        for table in tables:
            columns = schema_dict.get(table, [])
            col_signature = get_column_signature(columns)
            column_clusters[col_signature].append(table)

        # Log splits if verbose
        if verbose and len(column_clusters) > 1:
            print(f"\n[SPLIT] Pattern '{pattern}' has {len(column_clusters)} different column sets:")
            for i, (sig, tbls) in enumerate(column_clusters.items()):
                print(f"  Cluster {i}: {len(tbls)} tables - {tbls[0]} (and {len(tbls) - 1} others)")

        # Create final groups
        cluster_id = 0
        for col_signature, clustered_tables in column_clusters.items():
            if len(clustered_tables) == 1:
                # Standalone table with unique columns
                table_name = clustered_tables[0]
                final_groups[table_name] = [table_name]
            else:
                # Multiple tables with identical columns
                group_key = f"{pattern}_cluster_{cluster_id}"
                final_groups[group_key] = clustered_tables
                cluster_id += 1

    return final_groups


def walk_metadata_compress(dev,proj_dir , verbose=True):
    # import pdb; pdb.set_trace()
    dev_data = read_jsonl(osp.join(proj_dir, f'{dev}.jsonl'))   # question data file .jsonl

    # required_db_ids = set([item['db'] for item in dev_data])  # spider2-lite
    # required_db_ids = set([item['db_id'] for item in dev_data])
    # Líst all db_id need to answer all questions in dataset
    required_db_ids = set([item.get("db_id", item.get('db',"")) for item in dev_data])

    # currently supporting only bigquery, sqlite and snowflake
    db_base_paths = [osp.join(proj_dir,"resource/databases/bigquery/"), osp.join(proj_dir,"resource/databases/sqlite/"),
                     osp.join(proj_dir,"resource/databases/snowflake/"), osp.join(proj_dir,"resource/databases/")]
    json_glob_path = "**/*.json"

    db_stats_list = []
    all_schema_dict = {}    # save information of table and column names for all databases
    for base_path in db_base_paths:
        for db_path in glob.glob(os.path.join(base_path, "*"), recursive=False):

            db_id = os.path.basename(os.path.normpath(db_path))
            if db_id not in required_db_ids:
                continue
            all_schema_dict[db_id] = {}
            all_schema_dict[db_id]["table_names"] = []
            for json_file in glob.glob(os.path.join(db_path, json_glob_path), recursive=True):
                with open(json_file, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    table_name = data.get("table_name", "")
                    columns = data.get("column_names", [])
                    # nested_column_names = data.get("nested_column_names", [])
                    nested_column_names = data.get("nested_column_names", columns)
                    # all_schema_dict[db_id][table_name] = columns
                    all_schema_dict[db_id][table_name] = nested_column_names
                    all_schema_dict[db_id]["table_names"].append(table_name)
    # import pdb; pdb.set_trace()
    """
    Compress case : db_id có bảng trùng nhau : fec, ga360, ga4 
    Step 1 : First detect and group table name with same pattern 
    Step 2 : Check column names in the grouped tables, whether they are the same or not
    Step 3 : If they are the same, keep one table and remove other tables in
    """

    group_of_tables = {}
    # Gộp nhóm tên bảng cho từng db_id
    # Step 1 : First detect and group table name with same pattern
    for db_id in all_schema_dict:
        table_names = all_schema_dict[db_id]["table_names"]
        pattern_groups = group_names_by_pattern(table_names)
        # table_name_to_group_all[db_id] = table_name_to_group
        if verbose:
            print(f"Patterns detected: {len(pattern_groups)}")

        # import pdb; pdb.set_trace()
        # Lọc trong group phải có column giống nhau mới giữ lại
        # Step 2 : Check column names in the grouped tables, whether they are the same or not
        # Step 2: Re-cluster by column similarity
        schema_info = all_schema_dict[db_id]
        final_groups = cluster_tables_by_columns(
            pattern_groups, schema_info, verbose
        )
        group_of_tables[db_id] = final_groups
        print(f"After re-clustering, {len(final_groups)} groups remain for db {db_id}")
        print("final_groups:", final_groups)
        # import pdb; pdb.set_trace()


    # import pdb; pdb.set_trace()

    # Het step compress

    #### ##############################################


    for base_path in db_base_paths:
        for db_path in glob.glob(os.path.join(base_path, "*"), recursive=False):

            db_id = os.path.basename(os.path.normpath(db_path))
            if db_id not in required_db_ids:
                continue

            table_count = 0
            total_column_count = 0
            total_nested_column_count = 0

            table_file_path = []
            table_names_original = []
            table_fullnames = []
            column_names_original = []
            nested_column_names_original = []
            column_types = []
            nested_column_types = []
            descriptions = []
            nested_descriptions = []
            sample_rows = {}
            table_to_projDataset = {}

            for json_file in glob.glob(os.path.join(db_path, json_glob_path), recursive=True):
                with open(json_file, 'r', encoding='utf-8') as file:
                    try:
                        data = json.load(file)
                        # import pdb; pdb.set_trace()
                        table_count += 1
                        table_name = data.get("table_name", "")
                        table_fullname = data.get("table_fullname", "")
                        table_names_original.append(table_name)
                        table_fullnames.append(table_fullname)
                        table_file_path.append(json_file)

                        if 'bigquery' in base_path:  # bikeshare_trips: bigquery-public-data.austin_bikeshare
                            table_to_projDataset[table_name] = osp.basename(osp.dirname(json_file))
                        elif 'snowflake' in base_path:  # HISTORY_DAY: GLOBAL_WEATHER__CLIMATE_DATA_FOR_BI.STANDARD_TILE
                            table_to_projDataset[table_name] = osp.basename(
                                osp.dirname(osp.dirname(json_file))) + '.' + osp.basename(osp.dirname(json_file))
                        elif 'sqlite' in base_path:  # local
                            table_to_projDataset[table_name] = None  # we will not read it.
                        else:
                            # raise ValueError(f"Unknown database type: {base_path}")
                            if "spider2-snow" in base_path:
                                ## Wrong. DO NOT USE THIS PATH ANYMORE
                                table_to_projDataset[table_name] = None
                                # import pdb; pdb.set_trace()
                            else:
                                raise ValueError(f"Unknown database type: {base_path}")

                        columns = data.get("column_names", [])
                        # nested_columns_names bao gồm cả column và sub-column
                        # Vì thế phải lọc column ra để lấy sub-column nh nhất thôi
                        # nested_column_names_ori = data.get("nested_column_names", [])
                        nested_column_names_ori = data.get("nested_column_names", columns)  # không có nested thì lấy column luôn
                        column_types_in_table = data.get("column_types", [])
                        # nested_column_types_ori = data.get("nested_column_types", [])
                        nested_column_types_ori = data.get("nested_column_types", column_types_in_table)
                        # descriptions ứng với mỗi nested_column_names .
                        # Lọc description của column và sub-column riêng ra
                        descriptions_in_table = data.get("description", [])
                        # print(descriptions_in_table)
                        # if len(nested_column_names_ori) > len(columns):
                        #     """
                        #     '../Spider2/spider2-lite/resource/databases/
                        #     bigquery/open_targets_platform_2/bigquery-public-data.open_targets_platform/interactionEvidence.json'
                        #     """
                            # import pdb; pdb.set_trace()
                        # if descriptions_in_table[0] is not None:
                        #     import pdb; pdb.set_trace()
                        descriptions_in_table = [ "" if desc is None else desc for desc in descriptions_in_table]
                        """
                        description : description của từng sub-column 
                        Vì thế gặp trường hợp len(columns) <= len(descriptions_in_table)
                        1 column có thể có nhiều description ( cho sub-column ) 
                        
                        -> Cần có 2 loại :  column_* , nested_column_*
                        
                        """

                        # Tách description cho column và nested_column riêng ra
                        columns_descriptions_in_table = []   # chỉ lưu description của column không có '.'
                        nested_columns_descriptions_in_table = []    # lưu description của nested_column có '.' lớn nhất
                        nested_column_names_in_table = []
                        nested_column_types_in_table = []
                        num_dot_in_next_column = 0
                        # subcolumn : trong tên có ký tự '.'
                        # còn có subsub column nữa, nên phải đếm số '.' để phân biệt
                        # subsubcolumn có 2 dấu chấm '.'
                        # chỉ lưu những column cấp cuối cùng ( không có nested column bên trong )
                        for idx, nested_column_name in enumerate(nested_column_names_ori):
                            num_dot_in_this_column = nested_column_name.count('.')
                            num_dot_in_next_column = nested_column_names_ori[idx + 1].count('.') if idx + 1 < len(
                                nested_column_names_ori) else 0
                            if num_dot_in_this_column == 0:
                                # Đây là column thường
                                columns_descriptions_in_table.append(descriptions_in_table[idx])

                            # Column không có nested column bên trong cững lưu vào nested_column_*
                            if num_dot_in_this_column == 0 and num_dot_in_next_column < 1:
                                # Đây là column thường không có nested column bên trong
                                nested_columns_descriptions_in_table.append(descriptions_in_table[idx])
                                nested_column_names_in_table.append(nested_column_name)
                                nested_column_types_in_table.append(nested_column_types_ori[idx])
                            else:
                                # Đây là nested column
                                # Kiểm tra xem đã là cấp cuối cùng chưa
                                if num_dot_in_next_column <= num_dot_in_this_column:
                                    # Đây là cấp cuối cùng của nested column
                                    nested_columns_descriptions_in_table.append(descriptions_in_table[idx])
                                    nested_column_names_in_table.append(nested_column_name)
                                    nested_column_types_in_table.append(nested_column_types_ori[idx])
                        if len(columns) > len(columns_descriptions_in_table):
                            """
                            db_id = census_bureau_acs_1, 
                                bigquery-public-data.geo_census_tracts/us_census_tracts_national.json 
                                15 column nhưng chỉ có 14 description: dư _PARTITIONTIME
                                bigquery-public-data.geo_us_boundaries/railways.json
                                5 column nhưng chỉ có 4 description : dư _PARTITIONTIME
                            """
                            print(f"Warning: {db_id} In {json_file}, len(columns) = {len(columns)} > len(columns_descriptions) = {len(columns_descriptions_in_table)}. Padding empty descriptions.")
                            columns_descriptions_in_table.extend([""] * (len(columns) - len(columns_descriptions_in_table)))
                        # if len(nested_column_names_in_table) < len(columns):
                        #     import pdb; pdb.set_trace()
                        """
                        Các list column name, description, type phải tương ứng với nhau
                        """
                        assert len(columns_descriptions_in_table) == len(columns)
                        assert len(column_types_in_table) == len(columns)
                        assert len(nested_columns_descriptions_in_table) == len(nested_column_names_in_table)
                        assert len(nested_column_names_in_table) == len(nested_column_types_in_table)
                        # if len(columns_descriptions) != len(columns):
                        #     import pdb; pdb.set_trace()
                        # if len(nested_column_types_in_table) == 0 :
                        #     import pdb; pdb.set_trace()
                        """
                        Error with file :
                        - sqlite/city_legislation/cities_currencies.json
                        - sqlite/city_legislation/cities_countries.json
                        - sqlite/IPL/player_match.json
                        Hình như data sqlite không có nested column nên thiếu trường nested_column_*
                        """

                        if len(nested_column_types_in_table) == 0 and len(nested_column_names_in_table) == 0:
                            nested_column_types_in_table = column_types_in_table
                            nested_column_names_in_table = columns
                            nested_columns_descriptions_in_table = columns_descriptions_in_table
                        assert len(nested_column_types_in_table) > 0
                        assert len(nested_column_names_in_table) > 0
                        assert len(nested_columns_descriptions_in_table) > 0
                        # if len(columns) != len(descriptions_in_table):
                        #     print(
                        #         f"Warning: Column count and description count mismatch in {json_file} with len {len(columns)} vs {len(descriptions_in_table)}")
                            # raise ValueError(f"Column count and type count mismatch in {json_file} with len {len(columns)} vs {len(descriptions_in_table)}")
                        total_column_count += len(columns)
                        total_nested_column_count += len(nested_column_names_in_table)

                        for col_index, col_name in enumerate(columns):
                            column_names_original.append([table_count - 1, col_name])
                        for col_index, col_name in enumerate(nested_column_names_in_table):
                            nested_column_names_original.append([table_count - 1, col_name])

                        column_types.extend(column_types_in_table)
                        nested_column_types.extend(nested_column_types_in_table)

                        for desc in columns_descriptions_in_table:
                            descriptions.append([table_count - 1, desc])
                        for desc in nested_columns_descriptions_in_table:
                            nested_descriptions.append([table_count - 1, desc])

                        if "sample_rows" in data:
                            sample_rows[table_name] = data["sample_rows"]

                    except json.JSONDecodeError:
                        print(f"Error reading {json_file}")

            avg_column_per_table = total_column_count / table_count if table_count > 0 else 0

            # if 'bigquery' in base_path:
            #     db_id = f"{project_name}.{db_id}"
            # elif 'sqlite' in base_path:
            #     db_id = db_id
            # elif 'snowflake' in base_path:
            #     db_id = f"{project_name}.{db_id}"
            # else:
            #     raise ValueError(f"Unknown database type: {base_path}")
            # print()

            db_type = "sqlite"
            if 'bigquery' in db_path:
                db_type = "bigquery"
            elif 'snowflake' in db_path:
                db_type = "snowflake"
            elif 'spider2-snow' in db_path:
                db_type = "snowflake"
            # \T\O\D\O-DONE 18122025 add example_values
            example_values = []
            """
            Vì spider2 có sẵn sample_rows rồi, chỉ cần lấy value từ sample rows thôi
            spider2-snow : nested_column_names_original == column_names_original
            -> lấy nested_column_names_original luôn nhé 
            """
            # import pdb; pdb.set_trace()
            for table_idx, col_name in nested_column_names_original:
                table_name = table_names_original[table_idx]
                sample_rows_this_table = sample_rows.get(table_name, [])
                example_value_for_column = []
                # if col_name == "customDimensions.index":
                #     import pdb;
                #     pdb.set_trace()
                for sample_row in sample_rows_this_table:
                    if '.' not in col_name:
                        # non-nested column
                        value = sample_row.get(col_name, None)
                        if value is not None:
                            example_value_for_column.append(value)
                    else:
                        # nested column
                        parts = col_name.split('.')
                        value = sample_row[parts[0]]    # value cấp cao nhất

                        # print(columns, "   : ", value)
                        if type(value) is str:
                            """
                            Trường hợp example row là string -> phải chuyển về json object đã
                            """
                            try:
                                value = json.loads(value)
                            except:
                                try:
                                    value = json.loads(value.replace("'",'"'))
                                except:
                                    # import pdb;
                                    # pdb.set_trace()
                                    value = None
                                    continue

                        for part in parts[1:]:
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                value = None
                                # import pdb;
                                # pdb.set_trace()
                                break
                        if value is not None:
                            example_value_for_column.append(value)
                example_value_for_column = list(set(example_value_for_column))  # unique values only
                example_values.append(example_value_for_column)
            assert len(example_values) == len(nested_column_names_original)
            ##### DONE # 18122025
            db_stats_list.append({
                "db_id": db_id,
                "db_path": db_path,
                "db_type": db_type,
                "db_stats": {
                    "No. of tables": table_count,
                    "No. of columns": total_column_count,
                    "No. of nested columns": total_nested_column_count,
                    "Avg. No. of columns per table": avg_column_per_table,
                    "num_of_group": len(group_of_tables[db_id])
                },
                "table_file_path": table_file_path,
                "table_names_original": table_names_original,
                "table_names": table_fullnames,
                "column_names_original": column_names_original,
                "nested_column_names_original": nested_column_names_original,
                "column_types": column_types,
                "nested_column_types": nested_column_types,
                "column_descriptions": descriptions,
                "nested_column_descriptions": nested_descriptions,
                "sample_rows": sample_rows,
                "primary_keys": [],
                "foreign_keys": [],
                "table_to_projDataset": table_to_projDataset,
                "group_of_tables": group_of_tables[db_id],
                "example_values": example_values        # 18122025
            })

    db_stats(db_stats_list)
    # db_stats_bar_chart(db_stats_list)
    return db_stats_list




def process_table_json(args, proj_dir, output_dir):
    os.makedirs(osp.join(output_dir, f'preprocessed_data_compress/{args.dev}'), exist_ok=True)
    db_stats_list = walk_metadata_compress(args.dev, proj_dir)

    for item in db_stats_list:
        if 'table_names_original' in item:
            # import pdb; pdb.set_trace()
            # item['table_names'] = item['table_names_original']
            # item['column_names'] = item['column_names_original']
            item['column_names'] = item['column_descriptions']
            item['nested_column_names'] = item['nested_column_descriptions']

    # with open(osp.join(output_dir, f"preprocessed_data_compress/{args.dev}/tables_preprocessed.json"), "w", encoding="utf-8") as json_file:
    #     json.dump(db_stats_list, json_file, indent=4, ensure_ascii=False)
    with open(osp.join(output_dir, f"preprocessed_data_compress/{args.dev}/tables_preprocessed_with_example_values.json"), "w", encoding="utf-8") as json_file:
        json.dump(db_stats_list, json_file, indent=4, ensure_ascii=False)

    # with open("tables_toy.json", "w", encoding="utf-8") as json_file:
    #     json.dump([db_stats_list[5]], json_file, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', default='spider2_dev', type=str, help='the name of dev file')
    args = parser.parse_args()

    proj_dir = f"../Spider2/{args.dev}/"
    output_dir = "spider2_schema_processing/"
    process_table_json(args, proj_dir, output_dir)


"""
python database/spider2/schema_processing/step1_load_schema_infor.py --dev spider2-lite
# snow 
python database/spider2/schema_processing/step1_load_schema_infor.py --dev spider2-snow

"""


"""


spider-snow 
No. of db: 152
Average No. of tables across all database: 51.71
Average No. of columns across all Database: 3435.97                                                                                   Average No. of nested columns across all Database: 3435.97
Average Avg. No. of columns per table across all Databases: 35.15 
"""