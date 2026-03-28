"""

This Agent do:
- Read part of database schema
- Ask LLM do generate CTE and seleted column JSON form question and part of database schema
- Check valid JSON output format
- Check valid CTE output
    - Exec SQl in CTE . Get result




"""
import glob
import json
import os
from typing import List, Dict, Any, Tuple
import re
import shutil

from .chat import ChatLLM
from .sql_exec_env import SqlExecEnv
from .database_schema_manager import TextInforManager
from .utils import initialize_logger, extract_all_blocks, get_in_context_examples
from .extract_cte_utils import extract_cte_info
from .lsh_index import ValueManager, is_number , embedding_function

class CTEAgent:
    """

    Agent responsible for generating CTEs (Common Table Expressions) based on provided schema and context.

    """
    def __init__(self, config: Dict[str, Any],
                 question_id: str, question_text: str,
                 db_id: str,
                 idx_part: int,
                 schema_dict: dict,
                 group_of_tables: dict, group_of_columns: dict,
                 cte_agents_list: list,
                 ext_knowledge_str: str = "",

                 # Configurations parameters
                 question_log_dir: str="",
                 # Other components
                 ):
        """Initialize the tools needed for CTE generation"""
        # Data must be provided
        self.config = config
        self.question_id = question_id
        self.question_text = question_text
        self.db_id = db_id
        self.idx_part = idx_part
        self.schema_dict = schema_dict
        self.group_of_tables = group_of_tables
        self.group_of_columns = group_of_columns
        self.cte_agents_list = cte_agents_list  # List of CTE agents to get CTEs . Use there results to get filtered schema


        self.ext_knowledge_str = ext_knowledge_str


        # Configurations : config for LLM and template
        self.model_name = config['cte_agent']['model_name']             # MUST have
        self.temperature = config['cte_agent'].get('temperature', 0)
        self.template_path = config['cte_agent']['template_path']       # MUST have
        self.json_type = config['cte_agent'].get('json_type', "single_table_chess") #

        self.prompt_type = config['cte_agent'].get('prompt_type', 'compact')
        self.stream = config['cte_agent'].get('stream', False)  # client.chat.completions.create : stream
        self.max_fix_attempt = config['cte_agent'].get('max_fix_attempt', 5)
        self.run_type = config['cte_agent'].get('run_type', "independent")   # independent : not use previous CTE, sequential : Add previous CTE to context
        self.max_tokens = config['cte_agent'].get('max_tokens', 2048)
        self.is_use_col_desc = self.config['cte_agent'].get('is_use_col_desc', True)
        self.question_log_dir = question_log_dir

        # Create needed components to execute the agent's task
        self.template = open(self.template_path, 'r').read() if self.template_path is not None else ""
        self.log_path = os.path.join(self.question_log_dir, "log.txt")
        self.previous_cte_str = ""
        self.example_block_text = ""
        self.init_components()
        self.fix_time = 0

        # Output attributes need to be set after cte generation
        self.message = []
        self.message_with_reasoning = []
        self.all_json_attempts = []
        self.all_cte_attempts = []

        self.final_cte = {}
        self.final_json = {}
        self.filtered_schema_dict = {} #
        self.filtered_schema_text_infor = None # TextInforManager
        self.relevant_values_text = ""  # contain relevant stored values extracted from ValueManager

    def init_components(self):
        """
        Init other objects needed for CTE generation

        :return:
        """
        self.logger = initialize_logger(self.log_path, logger_name=self.question_id)
        self.llm = ChatLLM(model_name=self.model_name, temperature=self.temperature, max_fix_attempt=self.max_fix_attempt,
                           max_tokens= self.max_tokens,stream= self.stream, logger = self.logger)
        self.sql_env = SqlExecEnv.get_instance()  # Use to execute and validate generated CTEs
        self.schema_text_infor = TextInforManager(question_id=self.question_id,
                                                  schema_dict=self.schema_dict,
                                                  group_of_tables=self.group_of_tables,
                                                  group_of_columns=self.group_of_columns,
                                                  prompt_type=self.prompt_type,
                                                  is_use_sample_rows=self.config['cte_agent'].get("is_use_sample_rows", False),
                                                  is_use_col_desc = self.is_use_col_desc)
        if self.run_type == "sequential":
            self.previous_cte_str = self.get_previous_cte_str()
        num_incontext_shot = self.config['cte_agent'].get('num_incontext_shot', 0)
        incontext_shot_folder_path = self.config['cte_agent'].get('incontext_shot_folder_path', "not_default_path_yet")

        incontext_shot_folder_path = os.path.join(incontext_shot_folder_path, self.schema_text_infor.api_type)
        self.example_block_text = get_in_context_examples(num_incontext_shot, incontext_shot_folder_path)
        pass
    def __del__(self):
        self.logger.info(f"[CTEAgent] Deleting CTEAgent for question_id: {self.question_id} part {self.idx_part}")
        del self.llm
        del self.schema_text_infor
        del self.message
        del self.message_with_reasoning
        del self.final_json
        del self.final_cte
        del self.filtered_schema_dict
        del self.filtered_schema_text_infor
        del self.relevant_values_text

    def get_previous_cte_str(self) -> str:
        """
        Get previous CTEs from other CTE agents in the cte_agents_list
        Use in case config .yaml:
        - run_type == 'sequential'
        :return:
        """
        previous_cte_str = ""
        num_exec_cte = 0
        existing_cte_name = []
        for cte_agent in self.cte_agents_list:
            # import pdb; pdb.set_trace()
            final_json = cte_agent.final_json
            final_cte = cte_agent.final_cte
            if final_json['json_answer'] == 'Y':
                if 'cte_exec_result_dict' in final_cte and len(final_cte['cte_exec_result_dict']) > 0:
                    previous_cte_str += f"The CTE is generated from previous step with other part of database schema could help:\n"
                    for cte_name in final_cte['cte_exec_result_dict']:
                        if cte_name.lower() in existing_cte_name:
                            continue
                        existing_cte_name.append(cte_name.lower())
                        cte_sql_query_this_name = final_cte['cte_exec_result_dict'][cte_name]['cte_sql']
                        cte_exec_result = final_cte['cte_exec_result_dict'][cte_name]['msg'][:200]
                        previous_cte_str += f"```cte\nWITH {cte_name} AS \n ({cte_sql_query_this_name}\n)\n```\n"
                        previous_cte_str += f"The execution result of SQL query in this CTE is:\n"
                        # previous_cte_str += f"-- CTE Name: {cte_name}\n"
                        previous_cte_str += f"\n{cte_exec_result}\n"
                    num_exec_cte += 1
            if len(previous_cte_str) > 10000:  # limit the length of previous_cte_str to avoid too long prompt
                break
        assert (num_exec_cte > 0) == (len(previous_cte_str) > 0)    # have executed CTE  thì previous_cte_str phải có nội dung
        if len(previous_cte_str) > 0:
            previous_cte_str = "\n###\n" + previous_cte_str
            previous_cte_str += "Symbol ; separate value in a row of CTE execution result.\n Do not use table and column in previous CTEs to answer the question. "
            previous_cte_str += "The previous CTEs are only for your reference the CTE syntax and more understanding of the database and question.\n###\n"
        return previous_cte_str
    def generate_cte(self):
        """
        Generate CTE based on the provided context and schema
        This is the main function of the CTEAgent.

        """
        history_path = os.path.join(self.question_log_dir, f"cte_message_part_{self.idx_part}.json")
        history_path_with_reasoning = os.path.join(self.question_log_dir, f"cte_message_part_{self.idx_part}_with_reasoning.json")

        self.logger.info(f"[CTEAgent] Start generating CTE generate_cte() for question_id: {self.question_id} part {self.idx_part}")
        self.logger.info(f"[CTEAgent] question_id {self.question_id} part {self.idx_part} have estimate token {self.schema_text_infor.estimate_num_token_schema_text}")

        request_kwargs = {
            "SCHEMA_STR": self.schema_text_infor.schema_text + self.previous_cte_str,
            "EXAMPLES_BLOCK": self.example_block_text,
            "EXTERNAL_KNOWLEDGE_STR": self.ext_knowledge_str,
            "QUESTION_TEXT": self.question_text,
            "DIALECT1": self.schema_text_infor.dialect1,
            "DIALECT2": self.schema_text_infor.dialect2,
        }
        # message = [{"role": "system", "content": self.template.format(**request_kwargs)}]
        self.message = [{"role": "user", "content": self.template.format(**request_kwargs)}]
        self.message_with_reasoning = self.message.copy()
        valid_response = False

        while self.fix_time < self.max_fix_attempt and  (not valid_response):
            self.fix_time += 1
            self.logger.info(f"[CTEAgent] {self.question_id} part {self.idx_part} Attempt {self.fix_time}: Generating CTE...")

            message_input_llm = []
            if len(self.message) > 6: # 3 turn
                message_input_llm = [self.message[0]] + self.message[-4:]
            else:
                message_input_llm = self.message
            # response = self.llm.get_model_response_format(messages=self.message, code_format_list=['cte','json'])
            response = self.llm.get_model_response_format(messages_inp=message_input_llm, code_format_list=['cte','json'])
            del message_input_llm
            self.response = response
            # response is answer with right format

            valid_response, fix_response_str = self.check_valid_response(response)
            self.logger.info(f"[CTEAgent] {self.question_id}  part {self.idx_part} Attempt {self.fix_time}: valid_response={valid_response} fix_response_str={fix_response_str}")
            if valid_response:
                """
                If CTE is valid, check consistent between JSON and CTE
                """
                valid_response_consis, list_lacking_col = self.check_consistent_json_cte()
                valid_response = valid_response_consis
                if not valid_response_consis:
                    self.logger.info(f"[CTEAgent] {self.question_id}  part {self.idx_part} Attempt {self.fix_time}: Inconsistent JSON and CTE. : {str(list_lacking_col)}")
                    fix_response_str += "\nThe JSON and CTE are inconsistent. The columns used in CTE must be included in JSON selected columns. The lacking columns are: " + ", ".join(list_lacking_col) + ". Please revise the JSON or CTE to make them consistent."
            self.message.append({"role": "assistant", "content": response})
            self.message.append({"role": "user", "content": fix_response_str})

            self.message_with_reasoning.append({"role": "assistant", "content": response, "reasoning": self.llm.reasoning})
            self.message_with_reasoning.append({"role": "user", "content": fix_response_str})

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.message, f, ensure_ascii=False, indent=4)
        with open(history_path_with_reasoning, 'w', encoding='utf-8') as f:
            json.dump(self.message_with_reasoning, f, ensure_ascii=False, indent=4)

        self.finalize(valid_response)

    @classmethod
    def history_is_right_prompt(cls,schema_text_infor: TextInforManager, message_content: str):
        """

        Using when load previous history

        :param schema_text_infor:
        :param message_content:
        :return:
        """
        # Check whether this log file match with current schema_text
        # is_right_prompt = schema_text_infor.schema_text.split("\n  Column:")[0] in message_content
        is_right_prompt = True
        for table_name in schema_text_infor.schema_dict:
            # all table name must in message content prompt
            # is_right_prompt = is_right_prompt and (f"Table: {table_name}" in message_content)
            # If use f"Table: {table_name}" will not cover with group_of_tables
            is_right_prompt = is_right_prompt and (f"{table_name}" in message_content)
        return is_right_prompt

    def load_history(self, history_previous_run_folder: str):
        """

        Load previous message history from file
        Most code similar with function generate_cte()
        :param history_previous_run_folder:
        :return:
        """
        self.logger.info(f"[CTEAgent] Load previous CTE message for question_id: {self.question_id} part {self.idx_part}")
        history_path = os.path.join(self.question_log_dir, f"cte_message_part_{self.idx_part}.json")
        history_path_with_reasoning = os.path.join(self.question_log_dir, f"cte_message_part_{self.idx_part}_with_reasoning.json")
        element_log_dir = self.question_log_dir.split("/")
        idx_run = int(element_log_dir[-1])
        question_id = element_log_dir[-2]
        assert question_id == self.question_id
        # Copy previous log file
        # because previous log file contain log about generate cte and sql
        if not os.path.exists(os.path.join(self.question_log_dir,"log_previous_run.txt")):
            shutil.copyfile(os.path.join(history_previous_run_folder, self.question_id, str(idx_run), "log.txt"), os.path.join(self.question_log_dir,"log_previous_run.txt"))

        cte_json_previous_run_path_list = glob.glob(f"{history_previous_run_folder}/{self.question_id}/{idx_run}/cte_message_part_*.json")
        # import pdb; pdb.set_trace()
        assert len(cte_json_previous_run_path_list) > 0, "Not found any json file previous cte message history."
        exist_log_file = False
        history_previous_run_path = ""
        history_previous_run_path_with_reasoning = ""
        for cte_previous_run_json_path in cte_json_previous_run_path_list:
            with open(cte_previous_run_json_path, 'r', encoding='utf-8') as f:
                message = json.load(f)
            # import pdb; pdb.set_trace()
            # Check whether this log file match with current schema_text
            # if self.schema_text_infor.schema_text.split("\n  Column:")[0] in message[0]['content']:
            if CTEAgent.history_is_right_prompt(self.schema_text_infor, message[0]['content']):
                exist_log_file = True

                if cte_previous_run_json_path.endswith("_with_reasoning.json"):
                    history_previous_run_path_with_reasoning = cte_previous_run_json_path
                else:
                    history_previous_run_path = cte_previous_run_json_path
        assert exist_log_file, f"Total {cte_json_previous_run_path_list} part but Not found previous cte message history for this part."
        if os.path.exists(history_previous_run_path):
            with open(history_previous_run_path, 'r', encoding='utf-8') as f:
                self.message = json.load(f)
        self.logger.info(f"[CTEAgent] Loaded previous CTE message history for question_id: {self.question_id} part {self.idx_part}")
        valid_response = False
        for one_msg in self.message:
            if one_msg['role'] == 'assistant':
                valid_response, fix_response_str = self.check_valid_response(one_msg['content'])
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.message, f, ensure_ascii=False, indent=4)
        shutil.copyfile(history_previous_run_path_with_reasoning, history_path_with_reasoning)
        self.finalize(valid_response)
        self.logger.info(f"[CTEAgent] Finish process previous CTE message history for question_id: {self.question_id} part {self.idx_part}")
    def finalize(self, valid_response):
        """

        Finalize the process after ask LLM
        Func do :
        - choose the final json and cte among all attempts
        - re ask LLM to confirm the decision
        - filter schema_dict based on final_json columns

        :return:
        self.final_json
        self.final_cte

         self.filtered_schema_dict
         self.filtered_schema_text_infor
        """
        if not valid_response:
            self.logger.warning(f"[CTEAgent]  {self.question_id} : Failed to generate valid CTE after {self.fix_time} attempts.")
            # print(f"[CTEAgent] Failed to generate valid CTE after {self.fix_time} attempts.")
            self.merge_all_attempts()
        else:
            self.logger.warning(f"[CTEAgent]  {self.question_id} : Success to generate valid CTE after {self.fix_time} attempts.")
            self.final_json = self.all_json_attempts[-1]
            self.final_cte = self.all_cte_attempts[-1]

        self.filter_schema_dict()

    def filter_schema_dict(self):

        final_json_columns_dict = self.final_json['json_columns_dict']
        final_cte_table_column_used_dict = self.final_cte['table_column_used_dict']
        """
        final_json_columns_dict and final_cte_table_column_used_dict 
        {table_name : [column_name1, column_name2, ...]}
        
        """

        filtered_schema_dict = {}
        """
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
        # Filter schema_dict based on final_json table columns
        for table_name, table_info in self.schema_dict.items():
            filtered_columns_name = []
            filtered_columns_type = []
            filtered_example_values = []
            filtered_columns_description = []
            filtered_sample_rows = [{} for _ in range(len(table_info['sample_rows']))]

            #  filter columns using JSON
            for table_full_name_out, columns_out in final_json_columns_dict.items():
                if table_info['table_to_tablefullname'] == table_full_name_out \
                    or table_info['table_to_tablefullname'].lower() == table_full_name_out.lower() \
                        or table_info['table_to_tablefullname'].split('.')[-1].lower() == table_full_name_out.split('.')[-1].lower():

                    for selected_column in columns_out:
                        if type(selected_column) == str:
                            for idx, col_name in enumerate(table_info['columns_name']):
                                if col_name.lower() == selected_column.lower():
                                    filtered_columns_name.append(col_name)
                                    filtered_columns_type.append(table_info['columns_type'][idx])
                                    filtered_example_values.append(table_info['example_values'][idx])
                                    filtered_columns_description.append(table_info['columns_description'][idx])

                                    for idx_row in range(len(table_info['sample_rows'])):

                                        if col_name in table_info['sample_rows'][idx_row]:
                                            filtered_sample_rows[idx_row][col_name] = table_info['sample_rows'][idx_row][col_name]
                                        elif col_name.lower() in table_info['sample_rows'][idx_row]:
                                            filtered_sample_rows[idx_row][col_name.lower()] = table_info['sample_rows'][idx_row][col_name.lower()]
                                        else:
                                            print(f"Error when filter sample rows for question {self.question_id} table {table_name} column {col_name}:")


            for table_full_name_out, columns_out in final_cte_table_column_used_dict.items():
                if table_info['table_to_tablefullname'] == table_full_name_out \
                    or table_info['table_to_tablefullname'].lower() == table_full_name_out.lower() \
                        or table_info['table_to_tablefullname'].split('.')[-1].lower() == table_full_name_out.split('.')[-1].lower():

                    for selected_column in columns_out:
                        for idx, col_name in enumerate(table_info['columns_name']):
                            if col_name.lower() == selected_column.lower():
                                if col_name in filtered_columns_name:
                                    self.logger.info(f"[CTEAgent] {self.question_id} part {self.idx_part} column {col_name} in table {table_name} already added from JSON filtering.")
                                    continue
                                filtered_columns_name.append(col_name)
                                filtered_columns_type.append(table_info['columns_type'][idx])
                                filtered_example_values.append(table_info['example_values'][idx])
                                filtered_columns_description.append(table_info['columns_description'][idx])

                                for idx_row in range(len(table_info['sample_rows'])):
                                    # Filter sample rows : just keep the selected columns value
                                    if col_name in table_info['sample_rows'][idx_row]:
                                        filtered_sample_rows[idx_row][col_name] = table_info['sample_rows'][idx_row][
                                            col_name]
                                    elif col_name.lower() in table_info['sample_rows'][idx_row]:
                                        filtered_sample_rows[idx_row][col_name.lower()] = \
                                        table_info['sample_rows'][idx_row][col_name.lower()]
                                    else:
                                        print(
                                            f"Error when filter sample rows for question {self.question_id} table {table_name} column {col_name}:")
            if len(filtered_columns_name) > 0:

                filtered_column_table_info = {
                    'db_id': table_info["db_id"],
                    'db_type': table_info["db_type"],
                    'table_to_tablefullname': table_info['table_to_tablefullname'],
                    'columns_name': filtered_columns_name,
                    'columns_type': filtered_columns_type,
                    'example_values': filtered_example_values,
                    'sample_rows': filtered_sample_rows,
                    'columns_description': filtered_columns_description,
                    'primary_key': table_info['primary_key'],
                    'foreign_keys': table_info['foreign_keys'],
                }
                filtered_schema_dict[table_name] = filtered_column_table_info

        self.filtered_schema_dict = filtered_schema_dict
        self.filtered_schema_text_infor = TextInforManager(question_id=self.question_id,
                                                  schema_dict=self.filtered_schema_dict,
                                                  group_of_tables=self.group_of_tables,
                                                  group_of_columns=self.group_of_columns,
                                                  prompt_type=self.prompt_type,
                                                is_use_sample_rows=self.config['sql_agent'].get("is_use_sample_rows",False),
                                                is_use_col_desc = self.config['sql_agent'].get('is_use_col_desc', True)
                                                           )

    def merge_all_attempts(self):
        """

        Merge all JSON and CTE from all attempts
        Check column used in CTE with column in json
        Func do :
        - Combine all tables and columns from all attempts in JSON part
        - Choose the last executable CTE among all attempts in CTE part
        :return:
        """
        self.final_json = self.merge_all_attempts_json()
        self.final_cte = self.merge_all_attempts_cte()
        pass
    def merge_all_attempts_json(self):
        """

        Merge all JSON from all attempts
        Get final json result.
        Func do :
        - Combine all tables and columns from all attempts
        :return:
        """
        final_json_columns_dict = {}
        final_json_think = ""
        ## Merge all json columns dict from all attempts
        for json_output in self.all_json_attempts:
            json_columns_dict = json_output['json_columns_dict']
            for table_full_name, columns in json_columns_dict.items():
                if table_full_name not in final_json_columns_dict:
                    final_json_columns_dict[table_full_name] = []
                # Merge columns in all attempts
                final_json_columns_dict[table_full_name].extend(columns)
            valid_json = json_output['valid_json']
            json_answer = json_output['json_answer']
            json_think = json_output['json_think']
            if valid_json and json_answer == "Y":
                final_json_think = json_think

        # Remove duplicate columns in each table
        for table_full_name in final_json_columns_dict:
            # Remove duplicate columns
            try:
                final_json_columns_dict[table_full_name] = list(set(final_json_columns_dict[table_full_name]))
            except:
                continue
        final_json = {
            "json_answer": "Y" if len(final_json_columns_dict) > 0 else "N",
            "json_think": final_json_think,
            "json_columns_dict" : final_json_columns_dict
        }
        return final_json

    def merge_all_attempts_cte(self):
        """

        Merge all CTE from all attempts
        Func do :
        - Choose the last executable CTE among all attempts
        :return:
        """
        final_cte = {
                "cte_sql_query": "",
                "cte_sql_dict": {}, # {cte_name : sql query }
                "table_column_used_dict": {}, # {table_name : [column_name ,..] }
                "cte_exec_result_dict": {}  # {cte_name : {'msg': result , 'cte_sql': sql query }}
            }
        cte_sql_dict = {}
        final_cte_exec_result_dict = {}
        for i in range(len(self.all_cte_attempts), 0, -1):
            cte_output = self.all_cte_attempts[i-1]
            valid_cte = cte_output['valid_cte']
            cte_sql_query = cte_output['cte_sql_query']
            cte_sql_dict = cte_output['cte_sql_dict']
            table_column_used_dict = cte_output['table_column_used_dict']
            cte_exec_result_dict = cte_output['cte_exec_result_dict']
            if valid_cte:

                final_cte = {
                    "cte_sql_query": cte_sql_query,
                    "cte_sql_dict": cte_sql_dict,
                    "table_column_used_dict": table_column_used_dict,
                    "cte_exec_result_dict": cte_exec_result_dict
                }
                return final_cte
            else:

                for cte_name in cte_exec_result_dict:
                    if cte_name not in final_cte_exec_result_dict:
                        final_cte_exec_result_dict[cte_name] = cte_exec_result_dict[cte_name]
        final_cte_sql_query = "WITH "
        for cte_name in final_cte_exec_result_dict:
            final_cte_sql_query += f" {cte_name} AS ( {final_cte_exec_result_dict[cte_name]['cte_sql']} ),\n"
            cte_sql_dict[cte_name] = final_cte_exec_result_dict[cte_name]['cte_sql']
        final_cte = {
            "cte_sql_query": final_cte_sql_query,
            "cte_sql_dict": cte_sql_dict,  # {cte_name : sql query }
            "table_column_used_dict": {},  # {table_name : [column_name ,..] }
            "cte_exec_result_dict": final_cte_exec_result_dict  # {cte_name : {'msg': result , 'cte_sql': sql query }}
        }
        return final_cte

    def check_valid_response(self, response: str):
        """
        Check if the response from the model is valid.

        Model response 2 main blocks :
        - JSON block
        - CTE block
        So we need to extract both blocks from the response.
        Use 2 blocks to check the consistency and validity of the response.

        Some conditions to check:
        - CTE should be executable in the sql_env
        - All columns used in CTE should be in the json columns list
        - If json answer is N, then no CTE should be provided


        - Case1: If json answer N --> do not have CTE .
            - Not have CTE. Correct. Remove the table. Done
            - If still have CTE, --> ask model agent to revise. Ask
        - Case2: If json answer Y --> have CTE:
            - Not hava CTe --> ask model to revise. Ask
            - Have CTE -
                - Validate the CTE code by executing it in the sql_env
                    - Can execute --> Add to needed table. Done.
                        - Check column used in CTE with column in json
                    - Cannot execute --> ask model to revise. Ask
        :param response:
        :return:
        """

        # Validate the generated CTE
        # import pdb; pdb.set_trace()
        cte_blocks = extract_all_blocks(response, 'cte')
        json_blocks = extract_all_blocks(response, 'json')
        valid_json, fix_json_str, json_info = self.check_valid_json(json_blocks)    #
        # Next : parse json_info to get json_answer, json_think, json_columns_dict
        if self.json_type == "single_table_chess":
            json_answer, json_think, json_columns_dict = self.parse_json_single_table(json_info)
        elif self.json_type == "multi_table_macsql":
            json_answer, json_think, json_columns_dict = self.parse_json_multi_table(json_info)
        else:
            raise NotImplementedError(f"json_type {self.json_type} not implemented yet.")
        self.all_json_attempts.append({"valid_json": valid_json,
                                       "fix_json_str": fix_json_str,
                                       "json_answer": json_answer,
                                       "json_think": json_think,
                                       "json_columns_dict": json_columns_dict})

        if json_answer == "N" and len(cte_blocks) == 0:
            # Do not have CTE. Correct. because json answer is N. Not table relevant to question
            valid_cte = True
            fix_cte_str = ""
            cte_sql_query = ""
            cte_sql_dict = {}
            table_column_used_dict = {}
            cte_exec_result_dict = {}
        else:
            # Need to validate the CTE code
            valid_cte, fix_cte_str,cte_sql_query, cte_sql_dict, table_column_used_dict, cte_exec_result_dict = self.check_valid_cte(cte_blocks)

        self.all_cte_attempts.append({"valid_cte": valid_cte,
                                      "fix_cte_str": fix_cte_str,
                                      "cte_sql_query": cte_sql_query,
                                      "cte_sql_dict": cte_sql_dict,
                                      "table_column_used_dict": table_column_used_dict,
                                      "cte_exec_result_dict": cte_exec_result_dict})

        fix_response_str = ""

        if json_answer == "N":
            # Do not have CTE
            if len(cte_blocks) == 0:
                # Correct. Do not have CTE
                valid_cte = True
                fix_cte_str = ""
            else:
                # Still have CTE. Ask model to revise
                pass
        elif json_answer == "Y":
            # Have CTE
            if len(cte_blocks) == 0:
                # Not have CTE. Ask model to revise
                valid_cte = False
            elif len(cte_blocks) >= 1:
                # Have CTE. Validate the CTE code
                pass
        valid_response = valid_json and valid_cte


        if not valid_response:
            fix_response_str = ""
            if not valid_json:
                fix_response_str += fix_json_str + " "
            if not valid_cte:
                fix_response_str += fix_cte_str + " "
        return valid_response, fix_response_str

    def check_consistent_json_cte(self):
        """
        In case table relevant to question (json_answer = Y)
        Check whether the columns used in CTE match with columns list in JSON.

        - If json_answer = N : Ask LLM to sure that all columns in the provided schema are not relevant to question
        - If json_answer = Y : Sure that all columns used in CTE are in the json columns list.
        If not consistent, ask model to revise.
        :param:
        self.final_json['json_columns_dict']
        self.final_cte['table_column_used_dict']
        :return:
        """

        final_json_columns_dict = self.all_json_attempts[-1]['json_columns_dict']
        final_cte_table_column_used_dict = self.all_cte_attempts[-1]['table_column_used_dict']
        is_consist = True
        list_lacking_col = []
        for table_full_name_out_cte, columns_out_cte in final_cte_table_column_used_dict.items():
            # is_consist_table = False
            for table_full_name_out_json, columns_out_json in final_json_columns_dict.items():
                if table_full_name_out_cte == table_full_name_out_json \
                    or table_full_name_out_cte.lower() == table_full_name_out_json.lower() \
                        or table_full_name_out_cte.split('.')[-1].lower() == table_full_name_out_json.split('.')[-1].lower():
                    # Check columns used in CTE are in json columns list
                    # is_consist_table = True
                    for col_used in columns_out_cte:
                        is_consist_column = False
                        for col_json in columns_out_json:
                            if str(col_used).lower() == str(col_json).lower() \
                                or col_used == col_json:
                                is_consist_column = True
                        if not is_consist_column:
                            list_lacking_col.append(f"Table: {table_full_name_out_cte} - Column: {col_used} ;")
                        # if not is_consist_column:
                        #     import pdb; pdb.set_trace()
                        is_consist = is_consist and is_consist_column

        if not is_consist:

            self.logger.warning(f"[CTEAgent]  {self.question_id} : Inconsistent between JSON columns and CTE columns used.")
        return is_consist, list_lacking_col
    def check_valid_json(self, json_blocks: list):
        """
        Check the number of json blocks in the response.
        It should be exactly 1 json block.
        :param json_blocks:
        :return:
        """
        if len(json_blocks) == 1:
            try:
                json_info = json.loads(json_blocks[0])
                valid_json = True
                fix_json_str = ''
            except Exception as err:

                json_info = {}
                valid_json = False
                fix_json_str = "The JSON block is not valid. When parsing the JSON block, the following error occurred: " + str(err) + ". " \
                               "Please provide a valid JSON block indicating whether a CTE is needed in the ```json ...  ``` format." \
                               "Please regenerate the response."
        elif len(json_blocks) == 0:

            json_info = {}
            valid_json = False
            fix_json_str = "The response does not contain a JSON block. " \
                           "Please  include a JSON block indicating whether a CTE is needed in the ```json ...  ``` format." \
                           "Please regenerate the response."
        else:

            fix_json_str = ""
            try:
                json_info = json.loads(json_blocks[-1])
            except Exception as err:

                json_info = {}
                print("When parsing the last JSON block among multiple blocks, error:", err)
                fix_json_str += f"When parsing the last JSON block among multiple blocks, error: {err}"
            valid_json = False
            fix_json_str += "The response contains multiple JSON blocks. " \
                           "Please provide only one JSON block indicating whether a CTE is needed in the ```json ...  ``` format." \
                            "Do last JSON block as final answer? " \
                           "Please regenerate the response."
        return valid_json, fix_json_str, json_info
    def parse_json_single_table(self, json_info: dict):
        """
        Check json block for single table schema.
        Check the content inside the json block.
        ```json
        {{
            "think": "reason step by step to decide wherether this table is relevant to question. If yes, summarize what information in the table can be used to answer the question.",
            "answer": "Y or N only to answer whether this table is relevant to question or not",
            "columns": [col_name1, col_name2]
        }}
        ```

        :param json_info:
        :return:
            - json_answer : "Y" or "N"
            - json_think : str including the reasoning
            - json_columns_dict : dict {table_name: [col_name1, col_name2]}
        """

        json_answer = json_info.get("answer", "")   # Y or N . Y if table relevant to question
        json_think = json_info.get("think", "")
        json_columns_list = json_info.get("columns", [])
        json_columns_dict = {}
        if len(json_columns_list) > 0:
            table_full_name = self.schema_dict[list(self.schema_dict.keys())[0]]['table_to_tablefullname']
            json_columns_dict = {table_full_name: json_columns_list}
        return json_answer, json_think, json_columns_dict

    def parse_json_multi_table(self, json_info: dict):
        """
        Check the content inside the json block.
        ```json
        {{
            "think": "reason step by step to decide whether this table is relevant to question. If yes, summarize what information in the table can be used to answer the question.",
            table_name_2: list of column name like [col_name1, col_name2]  or "drop_all"
            table_name_1: list of column name like [col_name1, col_name2]  or "drop_all"
        }}
        ```

        :param json_info:
        :return:
            - json_answer : "Y" or "N"
            - json_think : str including the reasoning
            - json_columns_dict : dict {table_name: [col_name1, col_name2]}
        """
        if type(json_info) != dict:
            return "N", "", {}
        json_answer = "N"
        json_think = json_info.get("think", "")
        json_columns_dict = {}
        # import pdb; pdb.set_trace()
        for table_name, column_name_list in json_info.items():
            if type(column_name_list) is list and len(column_name_list) > 0:
                json_columns_dict[table_name] = column_name_list
                json_answer = "Y"
        return json_answer, json_think, json_columns_dict

    def check_valid_cte(self, cte_blocks: list):
        """

        Check whether the CTE code is valid by executing it in the sql_env.
        This function:
        - Check the number of CTE blocks in the response. It should be exactly 1 CTE block.
        - Validate the CTE code by executing it in the sql_env.
        :param cte_blocks:
        :return:
                - valid_cte : bool indicating whether the CTE is valid
                - fix_cte_str : str indicating the reason why the CTE is invalid
                - cte_sql_query : str containing the CTE SQL query
                - cte_sql_dict : dict mapping CTE names to their SQL queries {cte_name : cte_sql}
                - table_column_used_dict : dict mapping CTE names to the tables and columns used {cte_name : {table_name: [column_name1, column_name2]}}
                - cte_exec_result_dict : dict mapping CTE names to their execution results {cte_name : exec_result}
        """
        if len(cte_blocks) == 1:
            cte_sql_query = cte_blocks[0]
            valid_cte = True
            fix_cte_str = ''
            if len(cte_sql_query.strip()) == 0:
                # In case response JSON is "N" and cte is empty ```cte\n```\n\n```
                import pdb; pdb.set_trace()
                valid_cte = False
                cte_sql_dict = {}
                table_column_used_dict = {}
                cte_exec_result_dict = {}
                return valid_cte, fix_cte_str, cte_sql_query, cte_sql_dict, table_column_used_dict, cte_exec_result_dict
        elif len(cte_blocks) == 0:
            # No cte block found. Ask model to revise
            cte_sql_query = ''
            valid_cte = False
            fix_cte_str = "The response does not contain a CTE block. " \
                           "Please  include a CTE block in the ```cte ...  ``` format." \
                           "Please regenerate the response."
            cte_sql_dict = {}
            table_column_used_dict = {}
            cte_exec_result_dict = {}
            return valid_cte, fix_cte_str, cte_sql_query, cte_sql_dict, table_column_used_dict, cte_exec_result_dict
        else:
            # Multiple cte blocks found. Ask model to revise

            cte_sql_query = cte_blocks[-1]
            valid_cte = False
            fix_cte_str = "The response contains multiple CTE blocks. " \
                           "Please provide only one CTE block in the ```cte ...  ``` format." \
                           "Do last CTE block as final answer? " \
                           "Please regenerate the response."

        cte_sql_dict, table_column_used_dict, value_dict, parser_err_str = extract_cte_info(cte_sql_query, dialect = self.schema_text_infor.api_type)
        try:
            relevant_values_text_this_turn = self.process_relevant_values(value_dict, cte_sql_query)
            if len(relevant_values_text_this_turn) > 0:
                relevant_values_text_this_turn += "Use these relevant values to revise the format and spelling mistakes of value in the CTE if any or check whether the column contains such values."
        except Exception as err:
            relevant_values_text_this_turn = ""
            self.logger.error(f"[CTEAgent] {self.question_id} Error when process relevant values in CTE: {err}")
        fix_cte_str += f"{relevant_values_text_this_turn}\n"
        # Validate the CTE code by executing it in the sql_env
        cte_exec_result_dict = {}
        if len(cte_sql_dict) == 0:
            valid_cte = False
            fix_cte_str += " The final query is not CTE or the CTE query have problem so it could not be parsed correctly."\
                            f" Using sqlglot parser have error : {parser_err_str} . " \
                            "Please generate right CTE query with right syntax."
        else:
            ex_id = self.question_id  # Để lưu conn cho từng db_id - tiết kiệm time

            for cte_name, cte_sql in cte_sql_dict.items():
                cte_exec_result = self.sql_env.execute_sql_api(cte_sql, ex_id=ex_id,
                                                         api=self.schema_text_infor.api_type,
                                                         max_len=10000,
                                                         db_id=self.db_id)
                if cte_exec_result['status'] == "error":
                    valid_cte = False

                    fix_cte_str += f" The CTE '{cte_name}' could not be executed due to an error: {cte_exec_result['error_msg']}. Please revise the CTE."
                    if "Call 'USE DATABASE', or use a qualified name" in cte_exec_result['error_msg']:
                        fix_cte_str += self.schema_text_infor.get_fix_sf_call_use_db_str()
                else:
                    cte_exec_result_dict[cte_name] = {'msg': cte_exec_result.get('msg', ''),
                                                        'cte_sql': cte_sql
                                                      }
                    if "No data found for the specified query" in cte_exec_result.get('msg', ''):
                        valid_cte = False
                        fix_cte_str += f" The CTE '{cte_name}' executed successfully but returned no data. So the query of CTE '{cte_name}' is right syntax but the conditional may be wrong. Checking the reason why CTE query '{cte_name}' returns empty. Please revise the CTE to ensure it returns relevant data. {relevant_values_text_this_turn}"

        return valid_cte, fix_cte_str,cte_sql_query, cte_sql_dict, table_column_used_dict, cte_exec_result_dict

    def process_relevant_values(self, value_dict: dict, cte_sql_query: str):
        """

        Process relevant values used in the CTE.
        Use ValueManager to find relevant values in the database.

        :param value_dict:
        :return:
        """
        value_manager_instance = ValueManager.get_instance()
        relevant_values_text_this_turn = ""

        """
        Create embedding model callable in each turn to avoid error in multiprocessing mode.
        """
        if value_manager_instance.embedding_model_type == "local":
            EMBEDDING_MODEL_CALLABLE = embedding_function(embedding_model_name=value_manager_instance.embedding_model_name,
                                                          device=value_manager_instance.device)
        else:
            raise ValueError(f"Invalid embedding model type: {value_manager_instance.embedding_model_type}")

        if len(value_dict) > 0 and value_manager_instance is not None:
            for value in value_dict:

                # match = re.search(r"\b" + re.escape(value)+ r"\b", cte_sql_query, re.IGNORECASE)
                match = re.findall(r"\b" + re.escape(value)+ r"\b", cte_sql_query, re.IGNORECASE)
                """
                The \b markers ensure value is matched only as a standalone word, not when it appears inside identifiers like:
                - value_abcxyz -> not match
                - abcxyz_value -> not match
                - value$$%# -> match
                """

                if match and len(match) > 0:

                    match = list(set(match))  # remove duplicate
                    for original_value in match:
                        if is_number(original_value):
                            continue
                        _, text_prompt = value_manager_instance.get_relevant_values(db_id=self.db_id,
                                                                            query_str = original_value, EMBEDDING_MODEL_CALLABLE=EMBEDDING_MODEL_CALLABLE,
                                                                                    logger = self.logger)
                        relevant_values_text_this_turn += text_prompt
                if value not in match:
                    _, text_prompt = value_manager_instance.get_relevant_values(db_id=self.db_id,
                                                                        query_str = value, EMBEDDING_MODEL_CALLABLE=EMBEDDING_MODEL_CALLABLE,
                                                                                logger = self.logger)
                    relevant_values_text_this_turn += text_prompt


        self.relevant_values_text += relevant_values_text_this_turn
        del EMBEDDING_MODEL_CALLABLE
        return relevant_values_text_this_turn