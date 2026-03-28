"""


Utility functions for extracting Common Table Expressions (CTEs). The information extract:
- Table names
- Column names



"""

import json
import re
from sqlglot import parse_one, exp
from sqlglot.optimizer.scope import build_scope, find_all_in_scope
from sqlglot.optimizer.qualify import qualify


def replace_single_quotes_in_select(sql: str) -> str:
    """
    Replace all single quotes (') with backticks (`) only in the SELECT list,
    i.e., from the first occurrence of SELECT to the first FROM that follows it.
    The rest of the SQL remains unchanged.
    """
    # (?is): i = case-insensitive, s = dot matches newline
    # Capture three parts: prefix up to SELECT, the SELECT..FROM chunk, and the rest
    m = re.search(r'(?is)^(.*?\bselect\b)(.*?)(\bfrom\b.*)$', sql.strip(), re.DOTALL)
    if not m:
        return sql  # no SELECT/FROM structure found; return original

    pre_select = m.group(1)   # includes 'SELECT'
    select_chunk = m.group(2) # between SELECT and FROM (the select list and optional comments)
    post_from = m.group(3)    # 'FROM' and everything after

    # Replace all single quotes with backticks only within the SELECT chunk
    select_chunk_fixed = select_chunk.replace("'", "`")

    return f"{pre_select}{select_chunk_fixed}{post_from}"
def add_limit_if_not_present(sql_query: str, limit_value: int,dialect: str):
    """
    Adds a LIMIT clause to a SQL query if it doesn't already have one.

    Args:
        sql_query (str): The SQL query string.
        limit_value (int): The value to use for the LIMIT clause.

    Returns:
        str: The modified SQL query string with a LIMIT clause (if added).
    """
    try:
        expression_tree = parse_one(sql_query, dialect=dialect)
    except:
        return sql_query
        # import pdb; pdb.set_trace()
    # Check if a LIMIT clause already exists
    if not expression_tree.find(exp.Limit):
        # If no LIMIT clause, add one
        expression_tree = expression_tree.limit(limit_value)

    return expression_tree.sql(dialect=dialect)

def replace_qoutes_in_string(sql: str) -> str:
    """
    To avoid error when parsing sql with sql like
    "WITH relevant_cards AS (\n  SELECT `id`, `name`, `borderColor` \n  FROM `cards` \n  WHERE `name` = 'Ancestor\\'s Chosen'\n)"
    There are 3 single quotes ` in the sql string.
    We need to replace the middle single quotes in the string literal with double quotes ".
    :param sql:
    :return:
    """
    # m = re.search(r'\'(.*?)\'(.*?)\'', sql.strip(), re.DOTALL)
    m = re.search(r"'(.*?)'(.*?)'", sql.strip(), re.DOTALL)  # 3 single quotes
    if not m:
        m = re.search(r'"(.*?)"(.*?)"', sql.strip(), re.DOTALL) # 3 double quotes
    if not m:
        return sql
    error_str = m.group(0)
    fixed_str = error_str[0] + error_str[1:-1].replace("'", '"') + error_str[-1]
    sql_fixed = sql.replace(error_str, fixed_str)
    return sql_fixed

def extract_cte_info(cte_sql: str,dialect: str= "mysql"):
    """
    Extracts table names and column names from the given CTE SQL string.
    Contains information :
    - CTE SQL queries: Lưu tên virtual table và sql tương ứng
        - cte table names
            - sql corresponding to each cte table
    - Table columns used in the CTE SQL :
        - Table names
            - Column names within each table
    Args:
        cte_sql (str): The CTE SQL string.
        dialect : sqlglot support multiple dialects, default is mysql.
                From tht we can parse different sql syntaxes : bigquery, snowflake,
    Returns:
        dict: A dictionary containing lists of table names and column names.
    """

    cte_sql_dict = dict()   #
    table_column_used_dict = dict() #
    value_dict = dict() #
    try:
        ast = parse_one(cte_sql, dialect=dialect)
    except Exception as err1:
        try:
            ast = parse_one(cte_sql + " \n SELECT * FROM this_is_not_table ", dialect=dialect)
        except Exception as err2:

            cte_sql_fixed = replace_qoutes_in_string(cte_sql)
            try:
                ast = parse_one(cte_sql_fixed, dialect=dialect)
            except:
                try:
                    ast = parse_one(cte_sql_fixed + " \n SELECT * FROM this_is_not_table ", dialect=dialect)
                except:

                    return cte_sql_dict, table_column_used_dict, value_dict, str(err2)
    table_alias_dict = {} #
    for table in ast.find_all(exp.Table):
        if table.name == 'this_is_not_table':
            continue
        table_column_used_dict[table.name] = []
        table_column_used_dict[table.name.lower()] = [] #
        if table.alias == "":
            continue
        table_alias_dict[table.alias] = table.name
        table_alias_dict[table.alias.lower()] = table.name

    for cte in ast.find_all(exp.CTE):

        cte_name = cte.alias
        try:
            cte_sql_query = cte.this.sql(dialect=dialect)   # bước này làm cho lower và upper bị loạn
            cte_sql_query_limit = add_limit_if_not_present(cte_sql_query, 5, dialect=dialect)
        except:
            continue
        cte_sql_dict[cte_name] = cte_sql_query_limit

        try:
            # ast_cte = parse_one(cte_sql_query_new, dialect=dialect)
            ast_cte = parse_one(cte_sql_query, dialect=dialect)
            qualify(ast_cte)
            cte_root = build_scope(ast_cte)

            # import pdb; pdb.set_trace()
            for column in find_all_in_scope(cte_root.expression, exp.Column):                   #  dictionary : {table_name: [column_name]}
                table_name = column.table
                table_true_name = table_name
                if table_name in table_alias_dict:  #
                    table_true_name = table_alias_dict[table_name]

                if table_true_name == "":
                    continue    #
                if table_true_name not in table_column_used_dict:
                    raise ValueError(f"Table name {table_true_name} not found in table_column_used_dict")
                table_column_used_dict[table_true_name].append(column.name)
            for value in ast_cte.find_all(exp.Literal):  #  value
                value_name = value.name
                value_parent = value.parent.this
                if value_parent:
                    if hasattr(value_parent, "table"):
                        if value_parent.table in table_alias_dict:
                            value_dict[value_name] = {"table": table_alias_dict[value_parent.table],
                                                      "column": value_parent.name}
                        else:
                            value_dict[value_name] = {"table": value_parent.table, "column": value_parent.name}
        except Exception as e:
            # print(e)
            pass

    for table_name in table_column_used_dict:   #

        table_column_used_dict[table_name] = list(set(table_column_used_dict[table_name]))

    cte_tables = set(cte_sql_dict.keys())
    cte_sql_dict_copy = cte_sql_dict.copy()
    for cte_name, cte_sql in cte_sql_dict_copy.items():
        for other_cte in cte_tables:
            if other_cte == cte_name:

                continue
            pattern = r'\b' + re.escape(other_cte) + r'\b'
            if re.search(pattern, cte_sql, flags=re.IGNORECASE):
                print(f"Removing CTE {cte_name} because it uses CTE table {other_cte}")
                del cte_sql_dict[cte_name]
                break

    return cte_sql_dict, table_column_used_dict, value_dict, ""


if __name__ == "__main__":
    # Test extract CTE info
    cte_example = """
    WITH top_customers AS (
        SELECT customer_id, SUM(amount) AS total_spent
        FROM orders
        WHERE order_date >= '2023-01-01'
        GROUP BY customer_id
        HAVING total_spent > 1000
    ),
    recent_orders AS (
        SELECT order_id, customer_id, order_date
        FROM orders
        WHERE order_date >= '2023-06-01'
    )
    """

    # Manually extracted info for testing
    cte_info = {
        "table_names": ["orders"],
        "column_names": ["customer_id", "amount", "order_date", "order_id"]
    }

    cte_sql_dict, table_column_used_dict, value_dict, parser_err_str = extract_cte_info(cte_example)
    print(cte_sql_dict)

"""

python av_sql/extract_cte_utils.py

"""