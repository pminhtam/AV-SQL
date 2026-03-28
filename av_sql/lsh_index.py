"""

Cite code from Alpha-SQL/alphasql/database/lsh_index.py

Code to hashing values stored in database columns for similarity search using LSH.
Code :
- Create LSH index
- Query LSH index
    - Filtered result by semantic similarity.


"""
import json
import os
import sys
import pickle
import random
import argparse
import redis
from datasketch import MinHash, MinHashLSH
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from nltk.util import ngrams
from tqdm import tqdm
from typing import Tuple, Any
import shutil
import os.path as osp
import numpy as np
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

proj_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
print("proj_dir:" ,proj_dir)
sys.path = [osp.join(proj_dir)] + sys.path   #


from av_sql.sql_exec_env import SqlExecEnv
from av_sql.database_schema_manager import DatabaseSchemaManager
from av_sql.cassandra_manager import CassandraKV
from cassandra.cluster import Cluster

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class LSHIndex:
    """
    A class for creating and querying a LSH index for a database schema.
    Copy code from Alpha-SQL/alphasql/database/lsh_index.py

    Attributes:
        QUERY_DISTINCT_VALUES_SQL (str): The SQL query to get the unique values for a column.
        CACHED_LSH_INDEX (Dict[str, Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, int, str]]]]): A dictionary mapping database ids to LSH indexes.
    """
    QUERY_DISTINCT_VALUES_SQL = "SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL"
    QUERY_FIRST_5_DISTINCT_VALUES_SQL = "SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT 5"
    QUERY_COUNT_DISTINCT_VALUES_SQL = "SELECT COUNT(DISTINCT {column_name}) FROM {table_name} WHERE {column_name} IS NOT NULL"

    CACHED_LSH_INDEX: Dict[str, Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, int, str]]]] = {}
    
    @classmethod
    def get_unique_database_values(cls, db_id: str, schema_dict: Dict,sqlite_root_dir: str,  sql_env: SqlExecEnv, ignore_primary_keys: bool = True, ignore_non_text_columns: bool = True) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the unique values for each column in the database schema.
        
        Args:
            db_id (str): The database id.
            schema_dict (dict): The database schema dictionary. Contains table and column information. Get from DatabaseSchemaManager.db_schema_dict_all
            sqlite_root_dir (str): The root directory of the SQLite databases.
            sql_env (SqlExecEnv): The SQL execution environment. is SqlExecEnv instance.
            ignore_primary_keys (bool): Whether to ignore primary keys.
            ignore_non_text_columns (bool): Whether to ignore non-text columns.
        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary containing the unique values for each column.
        """
        unique_values: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        db_path = Path(sqlite_root_dir) / db_id / f"{db_id}.sqlite"   # Spider/BIRD / KaggleDBQA
        # db_path = Path(sqlite_root_dir) / f"{db_id}.sqlite" # Spider2-lite
        db_path = str(db_path)
        # import pdb; pdb.set_trace()
        db_type = schema_dict[list(schema_dict.keys())[0]].get("db_type", "sqlite")
        # import pdb; pdb.set_trace()
        if db_type == "sqlite":
            print("Using sqlite database at path:", db_path)
            if db_path not in sql_env.conns.keys():
                sql_env.start_db_sqlite(db_path)
        elif db_type == "snowflake":
            db_path = "snowflake"
            if db_path not in sql_env.conns.keys():
                sql_env.start_db_sf(db_path)
        for table_name in schema_dict:
            for idx_col , column_name in enumerate(schema_dict[table_name]["columns_name"]):
                # import pdb; pdb.set_trace()
                if ignore_primary_keys and column_name in schema_dict[table_name].get("primary_key", []):
                    # print("Ignoring primary key column:", column_name)
                    continue
                if ignore_non_text_columns and (schema_dict[table_name]["columns_type"][idx_col].lower() != "text"
                    and schema_dict[table_name]["columns_type"][idx_col].lower() != "string"     # for bigquery
                ):
                    # print("Ignoring non-text column:", column_name)
                    continue
                # Source code from CHESS/src/database_utils/db_values/preprocess.py
                #  + and code EmbeddingTaskAmbiText2SQL/extract_sql/preprocess_ambval_bird.py
                # *_code  , *_id , *code , *id , *date* ,*email* ,*number* ,*_no , *_ref , *_key ,
                if any(keyword in column_name.lower() for keyword in
                ["_id", " id", "url", "email", "time", "phone", "date", "number", "code", "zip","charternum",
                    "text", "id", "_no", "_ref",  "_key", "_point_", "tract_"
                   ]) or column_name.endswith("Id"):
                    continue
                table_full_name = schema_dict[table_name]["table_to_tablefullname"]
                if db_type == "sqlite":
                    query = cls.QUERY_DISTINCT_VALUES_SQL.format(column_name="`" +column_name+ "`", table_name="`" + table_full_name + "`")
                    query_rows, columns, _ = sql_env.exec_sql_sqlite_full(query,sqlite_path= str(db_path))
                elif db_type == "snowflake":
                    continue
                    """
                    Snowflake table name is full name with DB_ID.SCHEMA.TABLE
                    or DB_ID.SCHEMA."TABLE"
                    Column name also MUST inside "" double quotes
                    MUST user double quotes
                    
                    Error when name like `DB_ID.SCHEMA.TABLE-NAME`
                    DB_ID do not put in side any quotes
                    Error when column name inside ` backtick 
                    ALso error when table name inside ' single quote
                    """
                    query = cls.QUERY_COUNT_DISTINCT_VALUES_SQL.format(column_name='"' + column_name + '"',
                                                                 table_name=db_id + "." + table_full_name)
                    query_rows, columns, _ = sql_env.exec_sql_sf_full(query, ex_id=db_path)
                    if len(query_rows) == 0:
                        continue
                    # print(f"Column {column_name} have number of distinct value {query_rows}")
                    num_distinct_val = query_rows[0][0]
                    # import pdb;pdb.set_trace()
                    query_rows = []
                    columns = []
                    # Just index column with number of distinct values < 5000
                    # Avoid indexing column with too many distinct values
                    if num_distinct_val < 5000:
                        query = cls.QUERY_DISTINCT_VALUES_SQL.format(column_name='"' + column_name + '"',
                                                                             table_name=db_id + "." + table_full_name)
                        query_rows, columns, _ = sql_env.exec_sql_sf_full(query, ex_id=db_path)
                        is_all_number = True
                        for sample_row in random.sample(query_rows, k=min(10,len(query_rows))):
                            # Check is all row is number
                            #
                            is_all_number &= is_number(sample_row[0])
                        if is_all_number:
                            # Not hashing is all row is number
                            # print("all item is number : ", query_rows[:5])
                            query_rows = []
                            columns = []
                        # print(f"Column {column_name} have number of distinct value {len(query_rows)}")

                    # if num_distinct_val > 1000:
                    # query = cls.QUERY_FIRST_5_DISTINCT_VALUES_SQL.format(column_name='"' +column_name+ '"', table_name= db_id + "." + table_full_name)
                    # query_rows, columns, _ = sql_env.exec_sql_sf_full(query, ex_id=db_path)

                        # print("Sample column : ", query_rows[:5])


                elif db_type == "bigquery":
                    continue
                    # import pdb; pdb.set_trace()
                    query = cls.QUERY_DISTINCT_VALUES_SQL.format(column_name="`" +column_name+ "`", table_name="`" + table_full_name + "`")
                    query_rows, columns, _ = sql_env.exec_sql_bq_full(query)
                else:
                    query_rows = []
                    columns = []
                # import pdb; pdb.set_trace()
                if len(query_rows) > 0: #
                    assert len(columns) == len(query_rows[0]), f"Columns length {len(columns)} != query_rows length {len(query_rows[0])}"
                    unique_values[table_name][column_name] = [str(row[0]) for row in query_rows]
                # import pdb; pdb.set_trace()
        if "snowflake" in sql_env.conns:
            """
            Don't know why keep connection open to snowflake cause problem when it wait too long.
            So close it after use.
            """
            sql_env.conns["snowflake"].close()
            del sql_env.conns["snowflake"]
        return unique_values
        
    @classmethod
    def create_minhash(cls, string: str, signature_size: int = 128, n_gram: int = 3) -> MinHash:
        """
        Create a MinHash for a string.
        
        Args:
            string (str): The string to create a MinHash for.
            signature_size (int): The size of the signature, defaults to 128.
            n_gram (int): The size of the n-gram, defaults to 5.
        Returns:
            MinHash: A MinHash for the string.
        """
        minhash = MinHash(num_perm=signature_size)
        for d in ngrams(string, n_gram):
            minhash.update("".join(d).encode('utf8'))
        return minhash
    
    @classmethod
    def create_lsh_index(cls, output_lsh_dir: str,db_id: str,sqlite_root_dir: str,schema_dict: Dict, sql_env: SqlExecEnv , threshold: float = 0.5, signature_size: int = 128, n_gram: int = 3) -> None:
        """
        Create a LSH index for the database schema.
        Store lsh index and minhashes into pkl files.
        Args:
            output_lsh_dir (str): The output directory to store the LSH index.
            db_id (str): The database id.
            sqlite_root_dir (str): The root directory of the SQLite databases.
            schema_dict (dict): The database schema dictionary. Contains table and column information. Get from DatabaseSchemaManager.db_schema_dict_all
            sql_env (SqlExecEnv): The SQL execution environment. is SqlExecEnv instance.
            threshold (float): The threshold for the LSH index, defaults to 0.5.
            signature_size (int): The size of the signature, defaults to 128.
            n_gram (int): The size of the n-gram, defaults to 3.
        """
        unique_values = cls.get_unique_database_values(db_id , schema_dict,sqlite_root_dir=sqlite_root_dir, sql_env=sql_env)
        # print(unique_values)
        lsh_index = MinHashLSH(threshold=threshold, num_perm=signature_size)
        minhashes = {}    # Why need this? Is it redundant ?
        # Because when lsh_index.query, we only get the keys, not the actual MinHash objects or metadata.
        # minhashes store the mapping from keys to MinHash objects and their metadata.
        total_unique_values_count = sum(len(column_values) for table_values in unique_values.values() for column_values in table_values.values())
        if total_unique_values_count > 50000:
            print(f"Warning: Database {db_id} has a large number of unique values ({total_unique_values_count}). This may take a long time and consume a lot of memory.")
            return
        pbar = tqdm(total=total_unique_values_count, desc=f"Creating LSH index for database: {db_id}")
        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():
                for value_idx, value in enumerate(column_values):
                    minhash = cls.create_minhash(value, signature_size, n_gram)
                    minhash_key = f"{table_name}_{column_name}_{value_idx}"
                    minhashes[minhash_key] = (minhash, table_name, column_name, value)
                    lsh_index.insert(minhash_key, minhash)
                    pbar.update(1)
        pbar.close()
        
        lsh_index_dir_path = Path(output_lsh_dir) / "lsh_index"
        if lsh_index_dir_path.exists():
            shutil.rmtree(lsh_index_dir_path)
        lsh_index_dir_path.mkdir(parents=True)
        
        lsh_index_path = lsh_index_dir_path / f"lsh_index.pkl"
        minhashes_path = lsh_index_dir_path / f"minhashes.pkl"
        with open(lsh_index_path, "wb") as f:
            pickle.dump(lsh_index, f)
        with open(minhashes_path, "wb") as f:
            pickle.dump(minhashes, f)
        del lsh_index , unique_values, minhashes    # To free memory

    @classmethod
    def delete_lsh_index_redis(cls, host_name: str,port: int,dataset_name: str, db_id: str) -> None:
        """
        Delete a LSH index from redis.
        Args:
        """

        minhashes_redis_conn = redis.Redis(host=host_name, port=port)
        """
        Dùng StrictRedis hay Redis đều được.
        Do you need backwards compatibility? Use Redis. Don't care? Use StrictRedis.
        """
        # minhashes_redis_conn.delete(f"minhashes_{dataset_name}_{db_id}")
        # minhashes_redis_conn.delete(f"minhashes_{db_id}")
        # print("Deleted minhashes from redis for db_id:", db_id)
        print(f"Number of keys befor flush: {minhashes_redis_conn.dbsize()}")

        # Remove all data from redis
        print("Flushing all Redis databases...")
        minhashes_redis_conn.flushall(asynchronous=True)  # or r.flushall() for synchronous flushing

        print("All Redis data removed.")
        # Optional: Verify that the data has been removed
        print(f"Number of keys after flush: {minhashes_redis_conn.dbsize()}")

    @classmethod
    def create_lsh_index_redis(cls, host_name: str,port: int,dataset_name: str, db_id: str, sqlite_root_dir: str, schema_dict: Dict,
                         sql_env: SqlExecEnv, threshold: float = 0.5, signature_size: int = 128,
                         n_gram: int = 3) -> None:
        """
        Create a LSH index for the database schema.
        Store lsh index and minhashes into redis.
        Why use redis ?
            - Because data too large -> out of memory (5mil value cost 50GB RAM) if store in memory like def create_lsh_index

        Args:
            threshold (float): The threshold for the LSH index, defaults to 0.5.
            signature_size (int): The size of the signature, defaults to 128.
            n_gram (int): The size of the n-gram, defaults to 3.
        """

        # print(unique_values)
        minhashes_redis_conn = redis.Redis(host=host_name, port=port)

        # Check if existing LSH index in redis
        # check_existing_lsh = minhashes_redis_conn.hgetall(f"minhashes_{dataset_name}_{db_id}")
        count_hash = minhashes_redis_conn.hlen(f"minhashes_{dataset_name}_{db_id}")
        print(f"Existing LSH index in redis for db_id {db_id} has {count_hash} entries.")
        # return
        if count_hash > 10 :
            print(f"LSH index already exists in redis for db_id {db_id}, skipping creation.")
            return
        # import pdb; pdb.set_trace()
        unique_values = cls.get_unique_database_values(db_id, schema_dict, sqlite_root_dir=sqlite_root_dir,
                                                       sql_env=sql_env)
        lsh_index = MinHashLSH(threshold=threshold, num_perm=signature_size,
                               storage_config={
                                   "type": "redis",
                                   "basename": f"lsh_{dataset_name}_{db_id}".encode('utf8'),
                                   "redis": {"host": host_name, "port": port},
                               },
                               )
        # minhashes = {}  # Why need this? Is it redundant ?
        # Because when lsh_index.query, we only get the keys, not the actual MinHash objects or metadata.
        # minhashes store the mapping from keys to MinHash objects and their metadata.
        total_unique_values_count = sum(
            len(column_values) for table_values in unique_values.values() for column_values in table_values.values())
        pbar = tqdm(total=total_unique_values_count, desc=f"Creating LSH index for database: {db_id}")
        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():
                for value_idx, value in enumerate(column_values):
                    minhash = cls.create_minhash(value, signature_size, n_gram)
                    minhash_key = f"{table_name}_{column_name}_{value_idx}"
                    # minhashes[minhash_key] = (minhash, table_name, column_name, value)
                    """
                    hset(key , field, value)
                    key= f"minhashes_{db_id}" :  to separate different  LSH indexes for different db_id 
                    to query use : value = hget(key, field)
                    Thực chất cái hset store a parent key with format f"minhashes_{dataset_name}_{db_id}" , and all fields inside is subkey
                    So the parent key f"minhashes_{dataset_name}_{db_id}" have same levels with lsh_index with basename f"lsh_{dataset_name}_{db_id}"
                    """
                    minhashes_redis_conn.hset(f"minhashes_{dataset_name}_{db_id}", minhash_key, pickle.dumps((minhash, table_name, column_name, value)))
                    lsh_index.insert(minhash_key, minhash)
                    pbar.update(1)
        pbar.close()
        minhashes_redis_conn.connection_pool.disconnect()

    @classmethod
    def create_lsh_index_cassandra(cls, host_name: str, port: int, dataset_name: str, db_id: str, sqlite_root_dir: str,
                               schema_dict: Dict,
                               sql_env: SqlExecEnv, threshold: float = 0.5, signature_size: int = 128,
                               n_gram: int = 3) -> None:
        """
        Create a LSH index for the database schema.
        Store lsh index and minhashes into cassandra.
        Cassandra : tablename chỉ gồm _ , không có - hay ký tự đặc biệt khác.

        Insert data chậm vãi, chậm gấp 5 lần so với redis. Redis tầm 120it/s, cassandra tầm 25it/s với cấu hình local.
        Args:
            threshold (float): The threshold for the LSH index, defaults to 0.5.
            signature_size (int): The size of the signature, defaults to 128.
            n_gram (int): The size of the n-gram, defaults to 3.
        """
        values_data_store_path = os.path.join(f"unique_values_data_store_{dataset_name}", f"{db_id}.json")
        os.makedirs(os.path.dirname(values_data_store_path), exist_ok=True)
        """
        Store all unique values into json file for debug and future use.
        
        """
        # is_db_existing = CassandraKV.check_existence(contact_points=[host_name], port=port, keyspace=f"minhashes_{dataset_name}_{db_id}".lower()[:45], table=f"minhashes_{dataset_name}_{db_id}")
        # import pdb; pdb.set_trace()
        # if is_db_existing:
        #     print(f"LSH index already exists in cassandra for db_id {db_id}, skipping creation.")
        #     return
        # lsh_cluster = Cluster([host_name], port=port)
        # lsh_session = lsh_cluster.connect()
        # https://github.com/ekzhu/datasketch/blob/7b4ebacafe39c93b28058f7da9e4881cedac1c46/datasketch/storage.py#L279
        #  (got 56 characters for "lsh_spider2_snow_amazon_vendor_analytics__sample_dataset"
        STORAGE_CONFIG_CASSANDRA = {
            "basename": f"lsh_{dataset_name}_{db_id}".encode('utf8'),
            "type": "cassandra",
            "cassandra": {
                "seeds": [host_name],
                # "session": lsh_session,    # Co the dung session da tao san neu muon. Nhưng ko được shutdown session . , không nên
                "keyspace": f"lsh_{dataset_name}_{db_id}".lower()[:47],  # BUG15112025 maximun 48 char , BUG10122025
                # "keyspace": "lsh",
                # Why can not use keyspace name like f"lsh_{dataset_name}_{db_id}" ?
                # Because all UPPERCASE letters in keyspace name will be converted to lowercase in Cassandra.
                "replication": {"class": "SimpleStrategy", "replication_factor": "1"},
                "drop_keyspace": False, # delete existing keyspace if exists
                "drop_tables": True,   # delete existing tables if exists
            },
        }
        lsh_index = MinHashLSH(threshold=threshold, num_perm=signature_size,
                               storage_config=STORAGE_CONFIG_CASSANDRA)
        print(f"minhashes_{dataset_name}_{db_id}")
        # minhash_kv_cassandra = CassandraKV(contact_points=[host_name], port=port, keyspace=f"minhashes_{dataset_name}_{db_id}", table=f"minhashes_{dataset_name}_{db_id}")
        minhash_kv_cassandra = CassandraKV(contact_points=[host_name], port=port, keyspace=f"minhashes_{dataset_name}_{db_id}".lower()[:45], table=f"minhashes_{dataset_name}_{db_id}")
        # import pdb; pdb.set_trace()

        unique_values = cls.get_unique_database_values(db_id, schema_dict, sqlite_root_dir=sqlite_root_dir,
                                                       sql_env=sql_env)
        #  Store all unique values into json file for debug and future use.
        with open(values_data_store_path, "w", encoding="utf-8") as f:
            json.dump(unique_values, f, ensure_ascii=False, indent=4)
            print(f"Stored unique values into {values_data_store_path}")

        print("Len unique_values : ", sum(len(column_values) for table_values in unique_values.values() for column_values in table_values.values()))
        # return
        # unique_values = {}
        # minhashes = {}  # Why need this? Is it redundant ?
        # Because when lsh_index.query, we only get the keys, not the actual MinHash objects or metadata.
        # minhashes store the mapping from keys to MinHash objects and their metadata.
        total_unique_values_count = sum(
            len(column_values) for table_values in unique_values.values() for column_values in table_values.values())
        print(db_id, " have total unique values ", total_unique_values_count)
        # return
        pbar = tqdm(total=total_unique_values_count, desc=f"Creating LSH index for database: {db_id}")
        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():
                for value_idx, value in enumerate(column_values):
                    minhash = cls.create_minhash(value, signature_size, n_gram)
                    minhash_key = f"{table_name}_{column_name}_{value_idx}"

                    minhash_kv_cassandra.put(minhash_key,
                                              pickle.dumps((minhash, table_name, column_name, value)))
                    lsh_index.insert(minhash_key, minhash)
                    pbar.update(1)
        pbar.close()
        minhash_kv_cassandra.close()
        del unique_values

    @classmethod
    def query_lsh_index(cls, lsh_dir: str, db_id:str, query: str, top_k: int = 10, signature_size: int = 128, n_gram: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Query the LSH index for the database schema.
        
        Args:
            database_schema (DatabaseSchema): The database schema to query.
            query (str): The query to search for.
            top_k (int): The number of results to return, defaults to 10.
            signature_size (int): The size of the signature, defaults to 128.
            n_gram (int): The size of the n-gram, defaults to 3.
        Returns:
            A list of tuples containing the score and the metadata.
        """
        lsh_index_dir_path = Path(lsh_dir) / "lsh_index"
        lsh_index_path = lsh_index_dir_path / f"lsh_index.pkl"
        minhashes_path = lsh_index_dir_path / f"minhashes.pkl"
        if db_id not in cls.CACHED_LSH_INDEX:
            with open(lsh_index_path, "rb") as f:
                lsh_index = pickle.load(f)
            with open(minhashes_path, "rb") as f:
                minhashes = pickle.load(f)
            cls.CACHED_LSH_INDEX[db_id] = (lsh_index, minhashes)
        lsh_index, minhashes = cls.CACHED_LSH_INDEX[db_id]
        
        query_minhash = cls.create_minhash(query, signature_size, n_gram)
        results = lsh_index.query(query_minhash)
        similar_items = [(result_key, minhashes[result_key][0].jaccard(query_minhash)) for result_key in results]
        # import pdb; pdb.set_trace()
        """
        jaccard : =1.0 khi text ngắn như "a" vs "ag" vs "ab"
        
        query_minhash_0 = cls.create_minhash("â", signature_size, n_gram)
        query_minhash_1 = cls.create_minhash("ab", signature_size, n_gram)
        query_minhash_0.jaccard(query_minhash_1) # = 1
        Nguyên nhân: vì text quá ngắn, ngram=3 nên chỉ có 1 ngram là "â" vs "ab". 
        Do đó, jaccard = 1. Fix bằng cách giảm ngram xuống 2 hoặc 1.
        """
        # similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[:top_k]
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        total_item = 0
        similar_items_final = []
        for similar_item in similar_items:
            if total_item >= top_k:  # > topk nhưng lsh_score=1.0 thì vẫn lấy
                if similar_item[1] == 1.0:
                    similar_items_final.append(similar_item)
                    total_item += 1
            else:
                similar_items_final.append(similar_item)
                total_item += 1
        similar_items = similar_items_final
        print("len(similar_items)  : ", len(similar_items))
        # import pdb; pdb.set_trace()
        return [
            {
                "query": query,
                "lsh_score": score,
                "table_name": minhashes[result_key][1],
                "column_name": minhashes[result_key][2],
                "value": minhashes[result_key][3]
            }
            for result_key, score in similar_items
        ]

    @classmethod
    def query_lsh_index_redis(cls,host_name: str,port: int,dataset_name: str, db_id: str, query: str, top_k: int = 10, signature_size: int = 128,
                        n_gram: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Query the LSH index for the database schema.

        Args:
            database_schema (DatabaseSchema): The database schema to query.
            query (str): The query to search for.
            top_k (int): The number of results to return, defaults to 10.
            signature_size (int): The size of the signature, defaults to 128.
            n_gram (int): The size of the n-gram, defaults to 3.
        Returns:
            A list of tuples containing the score and the metadata.
        """
        lsh_index = MinHashLSH(threshold=0.5, num_perm=signature_size,
                               storage_config={
                                   "type": "redis",
                                   "basename": f"lsh_{dataset_name}_{db_id}".encode('utf8'),
                                   "redis": {"host": host_name, "port": port},
                               },
                               )
        minhashes_redis_conn = redis.Redis(host=host_name, port=port)

        query_minhash = cls.create_minhash(query, signature_size, n_gram)
        results = lsh_index.query(query_minhash)
        similar_items = [(result_key, pickle.loads(minhashes_redis_conn.hget(f"minhashes_{dataset_name}_{db_id}",result_key))[0].jaccard(query_minhash)) for result_key in results]
        # similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[:top_k]
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        total_item = 0
        similar_items_final = []
        for similar_item in similar_items:
            if total_item >= top_k:     # > topk nhưng lsh_score=1.0 thì vẫn lấy
                if similar_item[1] == 1.0:
                    similar_items_final.append(similar_item)
                    total_item += 1
            else:
                similar_items_final.append(similar_item)
                total_item += 1
        similar_items = similar_items_final
        print("len(similar_items)  : ",len(similar_items))
        # import pdb; pdb.set_trace()
        return [
            {
                "query": query,
                "lsh_score": score,
                "table_name": pickle.loads(minhashes_redis_conn.hget(f"minhashes_{dataset_name}_{db_id}",result_key))[1],
                "column_name": pickle.loads(minhashes_redis_conn.hget(f"minhashes_{dataset_name}_{db_id}",result_key))[2],
                "value": pickle.loads(minhashes_redis_conn.hget(f"minhashes_{dataset_name}_{db_id}",result_key))[3]
            }
            for result_key, score in similar_items
        ]

    @classmethod
    def query_lsh_index_cassandra(cls,host_name: str,port: int,dataset_name: str, db_id: str, query: str, top_k: int = 10, signature_size: int = 128,
                        n_gram: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Query the LSH index for the database schema.

        Args:
            database_schema (DatabaseSchema): The database schema to query.
            query (str): The query to search for.
            top_k (int): The number of results to return, defaults to 10.
            signature_size (int): The size of the signature, defaults to 128.
            n_gram (int): The size of the n-gram, defaults to 3.
        Returns:
            A list of tuples containing the score and the metadata.
        """
        # lsh_cluster = Cluster([host_name], port=port)
        # lsh_session = lsh_cluster.connect()
        """
        BUG14122025 : không dùng session ngoài cho lsh nữa, để lib datasketch nó tự tạo sesion .
        session của lsh là dùng chung cho tất cả các instance storage cassandra.
        """
        STORAGE_CONFIG_CASSANDRA = {
            "basename": f"lsh_{dataset_name}_{db_id}".encode('utf8'),
            "type": "cassandra",
            "cassandra": {
                "seeds": [host_name],
                # "session": lsh_session,  # Co the dung session da tao san neu muon, không nên
                "keyspace": f"lsh_{dataset_name}_{db_id}".lower()[:47],
                "replication": {"class": "SimpleStrategy", "replication_factor": "1"},
                "drop_keyspace": False,
                "drop_tables": False,
            },
        }
        lsh_index = MinHashLSH(threshold=0.5, num_perm=signature_size,
                               storage_config=STORAGE_CONFIG_CASSANDRA
                               )
        minhash_kv_cassandra = CassandraKV(contact_points=[host_name], port=port, keyspace=f"minhashes_{dataset_name}_{db_id}".lower()[:45], table=f"minhashes_{dataset_name}_{db_id}")

        query_minhash = cls.create_minhash(query, signature_size, n_gram)
        results = lsh_index.query(query_minhash)
        similar_items = [(result_key, pickle.loads(minhash_kv_cassandra.get(result_key))[0].jaccard(query_minhash)) for result_key in results]
        # similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[:top_k]
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        total_item = 0
        similar_items_final = []
        for similar_item in similar_items:
            if total_item >= top_k:  # > topk nhưng lsh_score=1.0 thì vẫn lấy
                if similar_item[1] == 1.0:
                    similar_items_final.append(similar_item)
                    total_item += 1
            else:
                similar_items_final.append(similar_item)
                total_item += 1
        similar_items = similar_items_final
        print("len(similar_items)  : ", len(similar_items))
        result = [
            {
                "query": query,
                "lsh_score": score,
                "table_name": pickle.loads(minhash_kv_cassandra.get(result_key))[1],
                "column_name": pickle.loads(minhash_kv_cassandra.get(result_key))[2],
                "value": pickle.loads(minhash_kv_cassandra.get(result_key))[3]
            }
            for result_key, score in similar_items
        ]
        # lsh_session.shutdown()
        minhash_kv_cassandra.close()
        # lsh_cluster.shutdown()
        del minhash_kv_cassandra, query_minhash, lsh_index
        # import pdb; pdb.set_trace()
        return result

class embedding_function:
    """

    """
    def __init__(self, embedding_model_name, device: str = 'cpu'):
        self.embedding_model_name =  embedding_model_name
        self.sentence_transformer = SentenceTransformer(self.embedding_model_name, device=device)
        self.ef = self.sentence_transformer.encode

    def embed_documents(self, texts):
        return self.ef(texts)
    def __del__(self):
        self.sentence_transformer.to("cpu")
        del self.sentence_transformer, self.ef, self.embedding_model_name
    def embed_query(self, query):
        return self.ef([query])[0].tolist()
class ValueManager:
    """
    Manager value:
    - find similar value in database using LSH
    - Semantic embedding value
    - Filter by semantic ebedding vector
    Code from alphasql/runner/preprocessor.py

    There are two storage type :
    - .pkl file to store dict
    - cassandra
    Config
    storage_type : "pkl"" | "cassandra"
        lsh_dir : "" if pkl
        hostname  : if cassandra
        port   : if cassandra
    embedding_model:
        embedding_model_type : "local" or "openapi" ...
        embedding_model_name : ""

    Sample config in .yaml file :
    #value_manager:  # config for ValueManager object
    #  storage:
    #    storge_type: "cassandra" # "pkl" or "cassandra"
    #    lsh_dir: "" # for pkl storage type
    #    hostname: "localhost"
    #    port: 9042
    #  embedding_model:
    #    embedding_model_type: "local"
    #    embedding_model_name: 'all-MiniLM-L6-v2'
    #  lsh_top_k: 5  # optional, default 5
    #  lsh_signature_size: 128 # optional, default 128
    #  lsh_n_gram: 3 # optional, default 3
    #  edit_similarity_threshold: 0.3
    #  embedding_similarity_threshold: 0.3

    """
    _instance = None  # Class variable to hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # If no instance exists, create a new one
            cls._instance = super(ValueManager, cls).__new__(cls)
            # Optional: Add any initialization logic here if __init__ is not used
            # For example, if you need to set attributes immediately after creation
        return cls._instance  # Return the existing or newly created instance

    def __init__(self, config):
        if not hasattr(self, '_initialized'): # Prevent re-initialization
            self.config = config
            self.storage_type = self.config['value_manager']['storage']['storage_type']
            if self.storage_type == "cassandra":
                self.hostname = self.config['value_manager']['storage']['hostname']
                self.port = self.config['value_manager']['storage']['port']
            elif self.storage_type == "pkl":
                self.lsh_dir = self.config['value_manager']['storage']['lsh_dir']
            else:
                raise ValueError(f"Invalid storage type: {self.storage_type}")
            self.dataset_name = self.config["dataset_name"]


            self.embedding_model_type = self.config['value_manager']['embedding_model']['embedding_model_type']
            self.embedding_model_name = self.config['value_manager']['embedding_model']['embedding_model_name']
            self.device = self.config['value_manager']['embedding_model'].get("device", "cuda")
            # if self.embedding_model_type == "local":
            #     self.EMBEDDING_MODEL_CALLABLE = embedding_function(embedding_model_name=self.embedding_model_name)
            # else:
            #     raise ValueError(f"Invalid embedding model type: {self.embedding_model_type}")

            self.lsh_top_k = self.config['value_manager'].get("lsh_top_k", 5)
            self.lsh_signature_size = self.config['value_manager'].get("lsh_signature_size", 128)
            self.lsh_n_gram = self.config['value_manager'].get("lsh_n_gram", 3)

            self.edit_similarity_threshold = self.config['value_manager'].get("edit_similarity_threshold", 0.3)
            self.embedding_similarity_threshold = self.config['value_manager'].get("embedding_similarity_threshold", 0.5)
        else:
            print("ValueManager Instance already initialized")
    @staticmethod
    def get_instance():
        return ValueManager._instance

    def filter_candidate_values_by_edit_similarity(self,
                                                   candidate_values: List[Dict[str, Any]],
                                                   edit_similarity_threshold: float) -> List[Dict[str, Any]]:
        """
        Filter the candidate values by edit similarity.
        Code from alphasql/runner/preprocessor.py

        Args:
            candidate_values (List[Dict[str, Any]]): The candidate values.
            edit_similarity_threshold (float): The threshold of the edit similarity.

        Returns:
            The filtered candidate values.
        """
        filtered_candidate_values = []
        for candidate_value in candidate_values:
            table_name = candidate_value["table_name"]
            column_name = candidate_value["column_name"]
            query = candidate_value["query"]
            value = candidate_value["value"]
            edit_similarity = SequenceMatcher(None, value, query).ratio()
            if edit_similarity >= edit_similarity_threshold:
                filtered_candidate_values.append({
                    "query": query,
                    "table_name": table_name,
                    "column_name": column_name,
                    "value": value,
                    "edit_similarity": edit_similarity
                })
        return filtered_candidate_values

    def filter_candidate_values_by_embedding_similarity(self,
                                                        candidate_values: List[Dict[str, Any]],
                                                        embedding_similarity_threshold: float,
                                                        EMBEDDING_MODEL_CALLABLE : embedding_function) -> List[Dict[str, Any]]:
        """
        Filter the candidate values by embedding similarity.
        Code from alphasql/runner/preprocessor.py

        Args:
            candidate_values (List[Dict[str, Any]]): The candidate values.
            embedding_similarity_threshold (float): The threshold of the embedding similarity.

        Returns:
            The filtered candidate values.
        """
        cosine_similarity = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        to_embeded_list = [candidate_value["value"] for candidate_value in candidate_values]
        to_embeded_list += [candidate_value["query"] for candidate_value in candidate_values]
        to_embeded_list = list(set(to_embeded_list))
        embeddings = EMBEDDING_MODEL_CALLABLE.embed_documents(to_embeded_list)
        # embeddings = EMBEDDING_MODEL_CALLABLE.embed(to_embeded_list)
        embeddings = {to_embeded_list[i]: embeddings[i] for i in range(len(to_embeded_list))}

        filtered_candidate_values = []
        for candidate_value in candidate_values:
            table_name = candidate_value["table_name"]
            column_name = candidate_value["column_name"]
            value = candidate_value["value"]
            query = candidate_value["query"]
            edit_similarity = candidate_value["edit_similarity"]
            value_embedding = embeddings[value]
            query_embedding = embeddings[query]
            embedding_similarity = cosine_similarity(value_embedding, query_embedding)
            print(f"Value: {value}, Query: {query}, Embedding similarity: {embedding_similarity}")
            if embedding_similarity >= embedding_similarity_threshold:
                filtered_candidate_values.append({
                    "query": query,
                    "table_name": table_name,
                    "column_name": column_name,
                    "value": value,
                    "edit_similarity": edit_similarity,
                    "embedding_similarity": embedding_similarity
                })
        return filtered_candidate_values

    def get_relevant_values(self, db_id: str , query_str: str, EMBEDDING_MODEL_CALLABLE : embedding_function,
                            logger = None) -> (List, str):
        """
        Get the relevant values for a query string:

        :param db_id: db_id
        :param query_str: value extracted from CTe agent.
        :return:
        """
        lsh_candidate_values = []
        if self.storage_type == "pkl":
            results = LSHIndex.query_lsh_index(
                lsh_dir=os.path.join(self.lsh_dir, db_id),
                db_id=db_id,
                query =query_str,
                top_k=self.lsh_top_k,
                signature_size=self.lsh_signature_size,
                n_gram=self.lsh_n_gram
            )
        elif self.storage_type == "cassandra":
            results = LSHIndex.query_lsh_index_cassandra(
                host_name=self.hostname,
                port=self.port,
                # dataset_name=self.dataset_name,
                dataset_name="spider2_snow",
                db_id=db_id,
                query=query_str,
                top_k=self.lsh_top_k,
                signature_size=self.lsh_signature_size,
                n_gram=self.lsh_n_gram
            )
        else:
            raise ValueError(f"Invalid storage type: {self.storage_type}")
        lsh_candidate_values.extend(results)
        print("lsh_candidate_values : \n", lsh_candidate_values)
        if logger:
            logger.info(f"[ValueManager] : query_str : {query_str} -  lsh_candidate_values : \n  {str(lsh_candidate_values)}")

        # Step 2: Use edit distance to filter the candidate values.
        edit_similarity_candidate_values = self.filter_candidate_values_by_edit_similarity(lsh_candidate_values,
                                                                                           self.edit_similarity_threshold)
        print("[************** Step 2: Use edit distance to filter the candidate values. : \n",
              edit_similarity_candidate_values)
        if logger:
            logger.info("[ValueManager] : ************** Step 2: Use edit distance to filter the candidate values. : \n" +
                  str(edit_similarity_candidate_values))
        # Step 3: Use embedding similarity to filter the candidate values.
        embedding_similarity_candidate_values = self.filter_candidate_values_by_embedding_similarity(
            edit_similarity_candidate_values, self.embedding_similarity_threshold,
        EMBEDDING_MODEL_CALLABLE = EMBEDDING_MODEL_CALLABLE)
        print("************** Step 3: Use embedding similarity to filter the candidate values. : \n",
              embedding_similarity_candidate_values)
        if logger:
            logger.info("[ValueManager] : ************** Step 3: Use embedding similarity to filter the candidate values. : \n" +
                  str(embedding_similarity_candidate_values))

        # Step 4: Filter the candidate values with lower than COEFFICIENT * max_similarity_score
        # COEFFICIENT = 0.0 means no filtering
        COEFFICIENT = 0.0

        if len(embedding_similarity_candidate_values) > 0:
            text_prompt = f"Relevant stored values for '{query_str}' in database:\n"
            for item in embedding_similarity_candidate_values:
                text_prompt += f"\"{item['value']}\" from table '{item['table_name']}', column '{item['column_name']}'\n"
            print(text_prompt)
        else:
            text_prompt = ""
        return embedding_similarity_candidate_values, text_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="spider")
    args = parser.parse_args()
    # dataset_name = "spider"
    # dataset_name = "bird"
    # dataset_name = "bird_train"
    # dataset_name = "spider2_snow"
    # dataset_name = "spider2_lite"
    dataset_name = args.dataset_name
    if dataset_name=="spider":
        snowflake_credential_path=""
        bigquery_credential_path=""
        sqlite_root_dir = "/mnt/disk2/tampm/data_text2sql/spider_data/database/"
        table_file_path = "av_sql_preprocessed_data/spider/tables_example_values.json"
        root_output_lsh_dir = "av_sql_preprocessed_data/spider/lsh_output/"
    elif dataset_name=="bird":
        snowflake_credential_path=""
        bigquery_credential_path=""
        sqlite_root_dir = "/mnt/disk2/tampm/data_text2sql/bird/dev_20240627/dev_databases/"
        table_file_path = "av_sql_preprocessed_data/bird/dev_tables_example_values.json"
        root_output_lsh_dir = "av_sql_preprocessed_data/bird/lsh_output/"
    # elif dataset_name=="spider2_snow":
    #     snowflake_credential_path="av_sql/configs/snowflake_credential_bk.json"
    #     bigquery_credential_path="av_sql/configs/bigquery_credential.json"
    #     sqlite_root_dir = "../Spider2/spider2-lite/resource/databases/spider2-localdb/"
    #     # table_file_path = "spider2_schema_processing/preprocessed_data_compress/spider2-snow/tables_preprocessed.json"
    #     table_file_path = "spider2_schema_processing/preprocessed_data_compress/spider2-snow/tables_preprocessed_step2_group_columns_with_example_values.json"
    #     root_output_lsh_dir = "spider2_schema_processing/preprocessed_data_compress/spider2-snow/lsh_output/"
    #     """
    #     Data too much > 5 mil records -> Out of memory -> killed by OS
    #     How to reduce memory usage: ????
    #     Solution Using redis  . NOT OK. Because redis store in memory too.
    #     Need to use Cassandra for LSH and MongoDB for minhashes_dict
    #     """
    #     # host_name = 'localhost'
    #     host_name = '100.64.241.89'
    #     # port = 6379   # redis port
    #     port = 9042 # cassandra port
    #     """
    #     Spider2.0 : 5GB in cassandra for minhashes + lsh_index
    #     """
    elif dataset_name == "kaggleDBQA":
        snowflake_credential_path=""
        bigquery_credential_path=""
        sqlite_root_dir = "/mnt/disk2/tampm/data_text2sql/kaggle_dbqa/databases/"
        table_file_path = "av_sql_preprocessed_data/kaggleDBQA/KaggleDBQA_tables_example_values.json"
        root_output_lsh_dir = "av_sql_preprocessed_data/kaggleDBQA/lsh_output/"
    sql_env = SqlExecEnv(snowflake_credential_path=snowflake_credential_path,
                         bigquery_credential_path=bigquery_credential_path,
                        sqlite_root_dir=sqlite_root_dir )
    os.makedirs(root_output_lsh_dir, exist_ok=True)
    database_schema_manager_instance = DatabaseSchemaManager(table_file_path=table_file_path,dataset_name=dataset_name)
    is_continue = False
    # LSHIndex.delete_lsh_index_redis(host_name, port, dataset_name, db_id)
    for db_id in database_schema_manager_instance.db_schema_dict_all:
        print(f"Creating LSH index for database: {db_id}")
        output_lsh_dir_db_id = os.path.join(root_output_lsh_dir, db_id)
        schema_dict = database_schema_manager_instance.db_schema_dict_all[db_id]
        # For Spider BIRD
        LSHIndex.create_lsh_index(output_lsh_dir_db_id,db_id,sqlite_root_dir=sqlite_root_dir, schema_dict=schema_dict, sql_env= sql_env,  threshold=0.5, signature_size=128, n_gram=3)

        # db_id : census_bureau_acs_1
        # db_id : fda
        ######### Spider2-snow  ###############################
        ########################################################
        # if db_id == "EBI_CHEMBL":  # break this

        # if db_id == "GEO_OPENSTREETMAP":  # break this
        #     is_continue = True
        # """
        # Những database quá lớn :
        # - FEC : 620669
        # - EBI_CHEMBL : 868223
        # """
        # # For Spider2-Snow
        # # LSHIndex.create_lsh_index_redis(host_name,port,dataset_name, db_id,sqlite_root_dir=sqlite_root_dir, schema_dict=schema_dict, sql_env= sql_env,  threshold=0.5, signature_size=128, n_gram=3)
        # if not is_continue:
        #     print("Done : ", db_id)
        #     continue
        # try:
        #     LSHIndex.create_lsh_index_cassandra(host_name,port,dataset_name, db_id,sqlite_root_dir=sqlite_root_dir, schema_dict=schema_dict, sql_env= sql_env,  threshold=0.5, signature_size=128, n_gram=3)
        # except Exception as e:
        #     """
        #     EBI_CHEMBL : bị lỗi vì nhiều value quá : 868223
        #     """
        #     print(f"Error creating LSH index for database {db_id}: {e}")



    # results = LSHIndex.query_lsh_index(lsh_dir= lsh_dir, db_id = db_id, query="example query", top_k=5, signature_size=128, n_gram=3)
    # print(results)
