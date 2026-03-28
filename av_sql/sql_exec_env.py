"""

- Executes SQL queries:  BigQuery, Snowflake, SQLite
- Get database schema and columns
- Convert schema to string format for prompts




"""

"""

Code utilities for executing SQL queries and handling database connections.

Database types:
- SQLite
- BigQuery
- Snowflake


"""

import sqlite3
import io
import csv
import os
import time
from google.cloud import bigquery
from google.oauth2 import service_account
import snowflake.connector
import json
import pandas as pd
from multiprocessing import Process, Queue
import mysql.connector

def hard_cut(str_e, length=0):
    if length:
        if len(str_e) > length:
            str_e = str_e[:int(length)]+"\n"
    return str_e


class SqlExecEnv:
    """

    """
    _instance = None  # Class variable to hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # If no instance exists, create a new one
            cls._instance = super(SqlExecEnv, cls).__new__(cls)
            # Optional: Add any initialization logic here if __init__ is not used
            # For example, if you need to set attributes immediately after creation
        return cls._instance  # Return the existing or newly created instance

    def __init__(self, snowflake_credential_path = "", bigquery_credential_path = "", sqlite_root_dir :str="", mysql_env = None):
        # __init__ will be called every time an instance is "created"
        # but the actual object creation is controlled by __new__
        if not hasattr(self, '_initialized'): # Prevent re-initialization
            self.conns = {}
            self.snowflake_credential_path = snowflake_credential_path
            self.bigquery_credential_path = bigquery_credential_path
            self.sqlite_root_dir = sqlite_root_dir
            if type(mysql_env) == dict:
                mysql_host = mysql_env.get("host", "localhost")
                mysql_user = mysql_env.get("user", "root")
                mysql_password = mysql_env.get("password", "123456")
            else:
                mysql_host = "localhost"
                mysql_user = "root"
                mysql_password = "123456"
            self.mysql_host = mysql_host
            self.mysql_user = mysql_user
            self.mysql_password = mysql_password
            self._initialized = True
    @staticmethod
    def get_instance():
        if SqlExecEnv._instance is None:
            raise ValueError("Instance not created yet. Please create an instance first.")
        return SqlExecEnv._instance

    def get_rows(self, cursor, max_len):
        rows = []
        current_len = 0
        for row in cursor:
            row_str = str(row)
            rows.append(row)
            if current_len + len(row_str) > max_len:
                break
            current_len += len(row_str)
        return rows

    def get_csv(self, columns, rows):
        output = io.StringIO()
        writer = csv.writer(output, delimiter=';')
        writer.writerow(columns)
        writer.writerow(["-----"] * len(columns))
        writer.writerows(rows)
        csv_content = output.getvalue()
        output.close()
        return csv_content

    def start_db_sqlite(self, sqlite_path):
        if sqlite_path not in self.conns:
            uri = f"file:{sqlite_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self.conns[sqlite_path] = conn
            # print(f"sqlite_path: {sqlite_path}, (self.conns): {self.conns.keys()}")

    def start_db_mysql(self, database):
        if database not in self.conns:
            # print(f"Start MySQL connection to database: {database}")
            conn = mysql.connector.connect(
                host=self.mysql_host,
                user=self.mysql_user,
                password=self.mysql_password,
                database=database,
            )
            self.conns[database] = conn

    def start_db_sf(self, ex_id):
        if ex_id not in self.conns.keys():
            snowflake_credential = json.load(open(self.snowflake_credential_path))
            self.conns[ex_id] = snowflake.connector.connect(**snowflake_credential)
    def close_db_sf(self, ex_id):
        # BUG29112025 : close conns snowflake when complete one question
        # Because if conns too long will have error :  ##ERROR##390114 (08001): Authentication token has expired.
        if ex_id in self.conns.keys():
            self.conns[ex_id].close()
            del self.conns[ex_id]

    def close_db(self):
        # print("Close DB")
        for key, conn in list(self.conns.items()):
            try:
                if conn:
                    conn.close()
                    # print(f"Connection {key} closed.")
                    del self.conns[key]
            except Exception as e:
                print(f"When closing DB for {key}: {e}")

    def exec_sql_sqlite_full(self, sql_query,  max_len=30000, sqlite_path=None):
        """
        Execute SQL query on SQLite database and return full results.

        :param sql_query:
        :param max_len:
        :param sqlite_path:
        :return:
        """

        cursor = self.conns[sqlite_path].cursor()
        rows = []
        columns = []
        msg = ""
        try:
            cursor.execute(sql_query)
            column_info = cursor.description
            rows = self.get_rows(cursor, max_len)
            columns = [desc[0] for desc in column_info]
        except Exception as e:
            # BUG12112025: return "##ERROR##", số lượng return phải cố định
            # vì  hàm exec_sql_sqlite mặc định lấy giá trị rows, columns = self.exec_sql_sqlite_full(...)
            return rows, columns , "##ERROR##" + str(e)
        finally:
            try:
                cursor.close()
            except Exception as e:
                print("Failed to close cursor:", e)
        return rows, columns, msg

    def exec_sql_sqlite(self, sql_query, save_path=None, max_len=30000, sqlite_path=None):
        """
        Convert SQL query results from SQLite database to CSV format.
        :param sql_query:
        :param save_path:
        :param max_len:
        :param sqlite_path:
        :return:
            a string of CSV content
        """
        rows, columns, msg = self.exec_sql_sqlite_full(sql_query, max_len, sqlite_path)
        if not rows:
            if len(msg) > 0: # msg = "##ERROR##" + str(e)
                # Khi này rows = None, columns = None
                return msg  # BUG12112025 : khi câu SQL lỗi, trả về lỗi
            # Còn không có lỗi, chỉ không có data tức trường hợp rows = []
            return "No data found for the specified query.\n"
        else:
            csv_content = self.get_csv(columns, rows)
            if save_path:
                with open(save_path, 'w', newline='') as f:
                    f.write(csv_content)
                return 0
            else:
                return hard_cut(csv_content, max_len)
    def exec_sql_mysql_full(self, sql_query,  max_len=30000, db_id=None):
        """
        Execute SQL query on SQLite database and return full results.

        :param sql_query:
        :param max_len:
        :param db_id:
        :return:
        """

        conn = mysql.connector.connect(
            host=self.mysql_host,
            user=self.mysql_user,
            password=self.mysql_password,
            database=db_id,
        )
        cursor = conn.cursor()
        rows = []
        columns = []
        msg = ""
        try:
            cursor.execute(sql_query)
            column_info = cursor.description
            all_result = cursor.fetchall()
            rows = self.get_rows(all_result, max_len)
            columns = [desc[0] for desc in column_info]
        except Exception as e:
            # BUG12112025: return "##ERROR##", số lượng return phải cố định
            # vì  hàm exec_sql_sqlite mặc định lấy giá trị rows, columns = self.exec_sql_sqlite_full(...)
            return rows, columns , "##ERROR##" + str(e)
        finally:
            try:
                cursor.close()
                conn.close()
                del cursor
                del conn
            except Exception as e:
                print("Failed to close cursor:", e)
        return rows, columns, msg

    def exec_sql_mysql(self, sql_query, save_path=None, max_len=30000, db_id=None):
        """
        Convert SQL query results from SQLite database to CSV format.
        :param sql_query:
        :param save_path:
        :param max_len:
        :param db_id:
        :return:
            a string of CSV content
        """
        rows, columns, msg = self.exec_sql_mysql_full(sql_query, max_len, db_id)
        if not rows:
            if len(msg) > 0: # msg = "##ERROR##" + str(e)
                # Khi này rows = None, columns = None
                return msg  # BUG12112025 : khi câu SQL lỗi, trả về lỗi
            # Còn không có lỗi, chỉ không có data tức trường hợp rows = []
            return "No data found for the specified query.\n"
        else:
            csv_content = self.get_csv(columns, rows)
            if save_path:
                with open(save_path, 'w', newline='') as f:
                    f.write(csv_content)
                return 0
            else:
                return hard_cut(csv_content, max_len)
    def exec_sql_sf_full(self, sql_query, ex_id):
        """
        Execute SQL query on Snowflake database and return FULL results.
        :param sql_query:
        :param ex_id:
        :return:
        """
        rows = []
        columns = []
        msg = ""
        with self.conns[ex_id].cursor() as cursor:
            try:
                cursor.execute(sql_query)
                column_info = cursor.description
                rows = []
                for row in cursor:
                    rows.append(row)
                columns = [desc[0] for desc in column_info]
            except Exception as e:
                return rows, columns , "##ERROR##" + str(e)
        return rows, columns, msg
    def exec_sql_sf(self, sql_query, save_path, max_len, ex_id):
        """

        :param sql_query:
        :param save_path:
        :param max_len:
        :param ex_id:
        :return:
            a string of CSV content
        """
        with self.conns[ex_id].cursor() as cursor:
            try:
                cursor.execute(sql_query)
                column_info = cursor.description
                rows = self.get_rows(cursor, max_len)
                columns = [desc[0] for desc in column_info]
            except Exception as e:
                return "##ERROR##" + str(e)
        # Convert to CSV format
        if not rows:
            return "No data found for the specified query.\n"
        else:
            csv_content = self.get_csv(columns, rows)
            if save_path:
                with open(save_path, 'w', newline='') as f:
                    f.write(csv_content)
                return 0
            else:
                return hard_cut(csv_content, max_len)
    def exec_sql_bq_full(self, sql_query):
        """
        Execute SQL query on BigQuery database and return FULL results.
        :param sql_query:
        :return:

        """
        bigquery_credential = service_account.Credentials.from_service_account_file(self.bigquery_credential_path)
        client = bigquery.Client(credentials=bigquery_credential, project=bigquery_credential.project_id)
        query_job = client.query(sql_query)
        try:
            result_iterator = query_job.result()
        except Exception as e:
            return  [], [], "##ERROR##" + str(e)
        rows = []
        # import pdb; pdb.set_trace()
        columns = [field.name for field in result_iterator.schema]
        for row_dict in result_iterator:
            row = [row_dict[col] for col in columns]
            rows.append(row)
        return rows, columns, ""
    def exec_sql_bq(self, sql_query, save_path, max_len):
        """

        :param sql_query:
        :param save_path:
        :param max_len:
        :return:
            a string of CSV content
        """
        bigquery_credential = service_account.Credentials.from_service_account_file(self.bigquery_credential_path)
        client = bigquery.Client(credentials=bigquery_credential, project=bigquery_credential.project_id)
        query_job = client.query(sql_query)
        try:
            result_iterator = query_job.result()
        except Exception as e:
            return "##ERROR##" + str(e)
        rows = []
        current_len = 0
        for row in result_iterator:
            if current_len > max_len:
                break
            current_len += len(str(dict(row)))
            rows.append(dict(row))
        # Convert to CSV format.
        # Because rows is a list of dict, we can use pandas to convert to CSV
        df = pd.DataFrame(rows)
        # Check if the result is empty
        if df.empty:
            return "No data found for the specified query.\n"
        else:
            # Save or print the results based on the is_save flag
            if save_path:
                df.to_csv(f"{save_path}", index=False)
                return 0
            else:
                return hard_cut(df.to_csv(index=False), max_len)

    def execute_sql_api(self, sql_query, ex_id, save_path=None, api="sqlite", max_len=30000, db_id=None,
                        timeout=300):
        """
        Chạy câu SQL ứng với từng loại database

        what is ex_id ? ex_id is only for snowflake connection id

        """
        start_time = time.time()
        result = ""
        if api == "bigquery":
            result = self.exec_sql_bq(sql_query, save_path, max_len)
        elif api == "snowflake":
            if ex_id not in self.conns.keys():
                self.start_db_sf(ex_id)
            result = self.exec_sql_sf(sql_query, save_path, max_len, ex_id)
        elif api == "sqlite":
            sqlite_path = os.path.join(self.sqlite_root_dir, f"{db_id}.sqlite") # Spdier2.0 localdb
            if not os.path.exists(sqlite_path): # Spider1, BIRD
                sqlite_path = os.path.join(self.sqlite_root_dir, f"{db_id}/{db_id}.sqlite")
            # print(f"sqlite_path: {sqlite_path}")
            if sqlite_path not in self.conns.keys():
                self.start_db_sqlite(sqlite_path)
            result = self.execute_sqlite_with_timeout(sql_query, save_path, max_len, sqlite_path, timeout=timeout)
            # result = self.exec_sql_sqlite(sql_query, save_path, max_len, sqlite_path)
        elif api == "mysql":
            # print(f"Execute SQL in MySQL database: {db_id} query : {sql_query}")
            # if db_id not in self.conns.keys():
            #     self.start_db_mysql(db_id)
            result = self.exec_sql_mysql(sql_query, save_path, max_len, db_id)
            # print(f"mysql db_id: {db_id} have result : {str(result)}")
        running_time = time.time() - start_time
        if "##ERROR##" in str(result):
            return {"status": "error", "error_msg": str(result), "running_time": running_time}
        else:
            return {"status": "success", "msg": str(result), "running_time": running_time}

    def execute_sqlite_with_timeout(self, sql_query, save_path, max_len, sqlite_path, timeout=300):
        def target(q):
            result = self.exec_sql_sqlite(sql_query, save_path, max_len, sqlite_path)
            q.put(str(result))

        q = Queue()
        p = Process(target=target, args=(q,))
        p.start()

        p.join(timeout)
        if p.is_alive():
            try:
                p.terminate()
                p.join(timeout=2)
                if p.is_alive():
                    print("Terminate failed, forcing kill.")
                    p.kill()
                    p.join()
            except Exception as e:
                print(f"Error when stopping process: {e}")
            print(f"##ERROR## {sql_query} Timed out")
            return {"status": "error", "error_msg": f"##ERROR## {sql_query} Timed out\n"}
        else:
            if not q.empty():
                result = q.get()
                return result
            else:
                import pdb; pdb.set_trace()
                raise RuntimeError("Process p dead")

if __name__ == "__main__":
    snowflake_credential_path = "../Spider2/spider2-lite/evaluation_suite/snowflake_credential.json"
    bigquery_credential_path = "../Spider2/spider2-lite/evaluation_suite/bigquery_credential.json"
    sqlite_folder_path = "../Spider2/spider2-lite/resource/databases/spider2-localdb/"

    sql_env = SqlExecEnv(snowflake_credential_path, bigquery_credential_path)
    # api_type = "sqlite"
    # api_type = "bigquery"
    api_type = "snowflake"
    if api_type == "sqlite":
        instance_id = "local004"
        sql_query = """
        WITH CustomerData AS (
        SELECT
            customer_unique_id,
            COUNT(DISTINCT orders.order_id) AS order_count,
            SUM(payment_value) AS total_payment,
            JULIANDAY(MIN(order_purchase_timestamp)) AS first_order_day,
            JULIANDAY(MAX(order_purchase_timestamp)) AS last_order_day
        FROM customers
            JOIN orders USING (customer_id)
            JOIN order_payments USING (order_id)
        GROUP BY customer_unique_id
    )
    SELECT
        customer_unique_id,
        order_count AS PF,
        ROUND(total_payment / order_count, 2) AS AOV,
        CASE
            WHEN (last_order_day - first_order_day) < 7 THEN
                1
            ELSE
                (last_order_day - first_order_day) / 7
            END AS ACL
    FROM CustomerData
    ORDER BY AOV DESC
    LIMIT 3"""
        sqlite_path = os.path.join(sqlite_folder_path, "E_commerce.sqlite")
        # sql_env.start_db_sqlite(sqlite_path)
    elif api_type == "bigquery":
        instance_id = "bq003"
        sql_query = """
        WITH cte1 AS (
    SELECT
        CONCAT(EXTRACT(YEAR FROM (PARSE_DATE('%Y%m%d', date))), '0',
            EXTRACT(MONTH FROM (PARSE_DATE('%Y%m%d', date)))) AS month,
        SUM(totals.pageviews) / COUNT(DISTINCT fullVisitorId) AS avg_pageviews_non_purchase
    FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_2017*`,
        UNNEST (hits) AS hits,
        UNNEST (hits.product) AS product
    WHERE
        _table_suffix BETWEEN '0401' AND '0731'
        AND totals.transactions IS NULL
        AND product.productRevenue IS NULL
    GROUP BY month
),
cte2 AS (
    SELECT
        CONCAT(EXTRACT(YEAR FROM (PARSE_DATE('%Y%m%d', date))), '0',
            EXTRACT(MONTH FROM (PARSE_DATE('%Y%m%d', date)))) AS month,
        SUM(totals.pageviews) / COUNT(DISTINCT fullVisitorId) AS avg_pageviews_purchase
    FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_2017*`,
        UNNEST (hits) AS hits,
        UNNEST (hits.product) AS product
    WHERE
        _table_suffix BETWEEN '0401' AND '0731'
        AND totals.transactions >= 1
        AND product.productRevenue IS NOT NULL
    GROUP BY month
)
SELECT
    month, avg_pageviews_purchase, avg_pageviews_non_purchase
FROM cte1 INNER JOIN cte2
USING(month)
ORDER BY month;"""
        sqlite_path = ""
    elif api_type == "snowflake":
        instance_id = "sf011"
        sql_query = """
         WITH TractPop AS (
    SELECT
        CG."BlockGroupID",
        FCV."CensusValue",
        CG."StateCountyTractID",
        CG."BlockGroupPolygon"
    FROM
        CENSUS_GALAXY__ZIP_CODE_TO_BLOCK_GROUP_SAMPLE.PUBLIC."Dim_CensusGeography" CG
    JOIN
        CENSUS_GALAXY__ZIP_CODE_TO_BLOCK_GROUP_SAMPLE.PUBLIC."Fact_CensusValues_ACS2021" FCV
        ON CG."BlockGroupID" = FCV."BlockGroupID"
    WHERE
        CG."StateAbbrev" = 'NY'
        AND FCV."MetricID" = 'B01003_001E'
),

TractGroup AS (
    SELECT
        CG."StateCountyTractID",
        SUM(FCV."CensusValue") AS "TotalTractPop"
    FROM
        CENSUS_GALAXY__ZIP_CODE_TO_BLOCK_GROUP_SAMPLE.PUBLIC."Dim_CensusGeography" CG
    JOIN
        CENSUS_GALAXY__ZIP_CODE_TO_BLOCK_GROUP_SAMPLE.PUBLIC."Fact_CensusValues_ACS2021" FCV
        ON CG."BlockGroupID" = FCV."BlockGroupID"
    WHERE
        CG."StateAbbrev" = 'NY'
        AND FCV."MetricID" = 'B01003_001E'
    GROUP BY
        CG."StateCountyTractID"
)

SELECT
    TP."BlockGroupID",
    TP."CensusValue",
    TP."StateCountyTractID",
    TG."TotalTractPop",
    CASE WHEN TG."TotalTractPop" <> 0 THEN TP."CensusValue" / TG."TotalTractPop" ELSE 0 END AS "BlockGroupRatio"
FROM
    TractPop TP
JOIN
    TractGroup TG
    ON TP."StateCountyTractID" = TG."StateCountyTractID";"""
        sqlite_path = ""
    result = sql_env.execute_sql_api(sql_query, ex_id=None, api=api_type,
                                     sqlite_path=sqlite_path)
    print(result)
    sql_env.close_db()




"""
python baselines/sql_exec_env.py

"""
