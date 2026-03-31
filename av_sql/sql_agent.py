"""

 agent generate final sql query
- CTE agent
- database manager : database schema




"""
import time
import os
import json
import shutil
from typing import List, Dict, Any, Tuple

from .chat import ChatLLM
from .cte_agent import CTEAgent
from .sql_exec_env import SqlExecEnv
from .utils import initialize_logger, extract_all_blocks

class SQLAgent:
    """
    SQL generator
    """
    def __init__(self, config: Dict[str, Any],
                 question_id: str,
                 question_text: str,
                 db_id: str,
                 cte_agents_list: list[CTEAgent],
                 revise_sql: Any,
                 ext_knowledge_str: str = "",
                 # Configurations parameters
                 question_log_dir: str = "",
                 ):
        """Initialize the tools needed for SQL generation"""
        # Data must be provided
        self.config = config
        self.question_id = question_id
        self.question_text = question_text
        self.db_id = db_id
        self.cte_agents_list = cte_agents_list  # List of CTE agents to get CTEs . Use there results to get filtered schema
        self.revise_sql = revise_sql  # ReviseSQL class to revise SQL if needed
        self.ext_knowledge_str = ext_knowledge_str

        # Configurations : config for LLM and template
        self.model_name = config['sql_agent']['model_name']             # MUST have
        self.temperature = config['sql_agent'].get('temperature', 0)
        self.template_path = config['sql_agent']['template_path']       # MUST have
        self.prompt_type = config['sql_agent'].get('prompt_type', 'compact')
        self.stream = config['sql_agent'].get('stream', False)      # # client.chat.completions.create : stream
        self.max_fix_attempt = config['sql_agent'].get('max_fix_attempt', 5)
        self.max_tokens = config['sql_agent'].get('max_tokens', 2048)

        self.is_get_schema_text_from_cte_agent = config['sql_agent'].get('is_get_schema_text_from_cte_agent', True)
        self.is_get_cte_text_from_cte_agent = config['sql_agent'].get('is_get_cte_text_from_cte_agent', True)

        self.question_log_dir = question_log_dir

        # Create needed components to execute the agent's task
        self.log_path = os.path.join(self.question_log_dir, "log.txt")
        self.logger = initialize_logger(self.log_path, logger_name=self.question_id)
        self.llm = ChatLLM(model_name=self.model_name, temperature=self.temperature, max_fix_attempt=self.max_fix_attempt,
                           max_tokens = self.max_tokens, stream= self.stream, logger = self.logger)
        self.template = open(self.template_path, 'r').read()
        self.sql_env = SqlExecEnv.get_instance() # Use to execute and validate generated CTEs

        # Output attributes need to be set after sql generation
        self.message = []
        self.message_with_reasoning = []
        self.sql_result_msg = ""    # result  when execution is successful SQL query
        self.sql_query_final = "" # final SQL query generated
        self.sql_exec_result = None
    def __del__(self):
        self.logger.info(f"[SQLAgent] Deleting SQLAgent instance for question ID: {self.question_id}")
        # BUG13122025 : out of memory. So need to clean RAM
        del self.llm
        del self.message
        del self.message_with_reasoning

    def generate_sql(self, planning_text: str = ""):
        """Generate final SQL based on the provided context, CTEs, and schema"""
        history_path = os.path.join(self.question_log_dir, "cte_message_final.json")
        history_path_with_reasoning = os.path.join(self.question_log_dir, "cte_message_final_with_reasoning.json")
        final_sql_path = os.path.join(self.question_log_dir, f"{self.question_id}.sql")
        final_exec = os.path.join(self.question_log_dir, f"{self.question_id}_exec.json")
        self.logger.info(f"[SQLAgent] Generating final SQL for question ID: {self.question_id}")
        schema_text = self.get_schema_text()
        cte_text = self.get_cte_text()
        schema_text = schema_text
        dialect1 = self.cte_agents_list[0].schema_text_infor.dialect1
        dialect2 = self.cte_agents_list[0].schema_text_infor.dialect2
        request_kwargs = {
            "SCHEMA_STR": schema_text,
            "CTE_STR": cte_text,
            "EXTERNAL_KNOWLEDGE_STR": str(self.ext_knowledge_str) + "\n" + str(planning_text),
            "QUESTION_TEXT": self.question_text,
            "DIALECT1": dialect1,
            "DIALECT2": dialect2,
        }
        self.message = [{"role": "user", "content": self.template.format(**request_kwargs)}]
        self.message_with_reasoning = self.message.copy()
        fix_time = 0
        valid_response = False
        sql_query = ''
        sql_exec_result = None
        while fix_time < self.max_fix_attempt and  (not valid_response):
            fix_time += 1
            # response = self.llm.get_model_response_format(messages=self.message, code_format_list=['sql'])
            message_input_llm = []
            if len(self.message) > 6: # 3 turn
                message_input_llm = [self.message[0]] + self.message[-4:]
            else:
                message_input_llm = self.message
            response = self.llm.get_model_response_format(messages_inp=message_input_llm, code_format_list=['sql'])
            del message_input_llm
            self.response = response
            valid_response,sql_query, sql_exec_result, fix_sql_str = self.check_valid_response(response)
            self.message.append({"role": "assistant", "content": response})
            self.message.append({"role": "user", "content": fix_sql_str})
            self.message_with_reasoning.append({"role": "assistant", "content": response, "reasoning": self.llm.reasoning})
            self.message_with_reasoning.append({"role": "user", "content": fix_sql_str})
            self.logger.info(f"[SQLAgent]  {self.question_id} Attempt {fix_time} to generate valid SQL.   fix_response_str={fix_sql_str}")
            # print(f"[SQLAgent]  {self.question_id} Attempt {fix_time} to generate valid SQL.  fix_response_str={fix_sql_str}")
        with open(history_path, 'w') as f:
            json.dump(self.message, f, indent=4)
        with open(history_path_with_reasoning, 'w') as f:
            json.dump(self.message_with_reasoning, f, indent=4)

        self.sql_exec_result = sql_exec_result
        if sql_exec_result['status'] == 'success':
            with open(final_sql_path, 'w') as f:
                f.write(sql_query)
            self.sql_result_msg = sql_exec_result['msg']
            self.sql_query_final = sql_query
        if valid_response:  # SQL not empty and executed successfully
            with open(final_exec, 'w') as f:
                json.dump({sql_query : sql_exec_result}, f, indent=4)
            self.logger.info(f"[SQLAgent] Successfully generated valid SQL for question ID: {self.question_id}")
            print(f"[SQLAgent] Successfully generated valid SQL for question ID: {self.question_id}")
        # import pdb; pdb.set_trace()
        else:
            self.logger.info(f"[SQLAgent] Fail to generated valid SQL for question ID: {self.question_id}")
        # if valid_response and self.revise_sql is not None:
        # if sql_exec_result['status'] == 'success' and self.revise_sql is not None:
        if self.revise_sql is not None:
            self.revise_sql_func(schema_text, cte_text, dialect2, sql_query, sql_exec_result )

    def generate_multi_candidate_sql(self, planning_text: str = "", idx_run: int=0, num_response=3):
        """

        Model generate multiple candidate SQL queries.
        For each SQL query, check validity and execution result. and revise if needed

        Just generate one turn

        :param planning_text:
        :param num_response: number of response each calling times
        :return:
        """
        self.logger.info(f"[SQLAgent] multi-candidate Generating multi-candidate final SQL for question ID: {self.question_id} with idx_run={idx_run}, num_response={num_response}")
        history_path = os.path.join(self.question_log_dir, f"multi_candidate_sql_message_final_{idx_run}.json")
        schema_text = self.get_schema_text()
        cte_text = self.get_cte_text()
        schema_text = schema_text
        dialect1 = self.cte_agents_list[0].schema_text_infor.dialect1
        dialect2 = self.cte_agents_list[0].schema_text_infor.dialect2
        request_kwargs = {
            "SCHEMA_STR": schema_text,
            "CTE_STR": cte_text,
            "EXTERNAL_KNOWLEDGE_STR": str(self.ext_knowledge_str) + "\n" + str(planning_text),
            "QUESTION_TEXT": self.question_text,
            "DIALECT1": dialect1,
            "DIALECT2": dialect2,
        }
        self.message = [{"role": "user", "content": self.template.format(**request_kwargs)}]
        all_response = self.llm.get_model_response_format_multi_candidate(messages_inp=self.message, n=num_response)
        self.message.append(all_response)
        with open(history_path, 'w') as f:
            json.dump(self.message, f, indent=4)
        for choice_index in all_response:
            self.logger.info(f"[SQLAgent] multi-candidate Processing choice index {choice_index} for question ID: {self.question_id} with idx_run={idx_run}")
            response = all_response[choice_index]['content']
            sql_blocks = extract_all_blocks(response, 'sql')
            for sql_block in sql_blocks:
                name_suffix = str(time.time())
                final_sql_path = os.path.join(self.question_log_dir, f"{self.question_id}_{idx_run}_{name_suffix}.sql")
                final_exec = os.path.join(self.question_log_dir, f"{self.question_id}_exec_{idx_run}_{name_suffix}.json")
                valid_response, sql_query, sql_exec_result, fix_sql_str = self.check_valid_sql([sql_block])
                if sql_exec_result['status'] == 'success':
                    with open(final_sql_path, 'w') as f:
                        f.write(sql_query)
                if valid_response:  # SQL not empty and executed successfully
                    with open(final_exec, 'w') as f:
                        json.dump({sql_query: sql_exec_result}, f, indent=4)
                if self.revise_sql is not None:
                    self.revise_sql_func(schema_text, cte_text, dialect2, sql_query, sql_exec_result, name_suffix="_" + str(idx_run) +"_" + name_suffix)

    def load_history(self, history_previous_run_folder: str):
        """
        Load from history json file
        Most code similar with function generate_sql()
        :param history_previous_run_folder:
        :return:
        """

        self.logger.info(f"[SQLAgent] Loading previous history from {history_previous_run_folder}")
        history_path = os.path.join(self.question_log_dir, "cte_message_final.json")
        history_path_with_reasoning = os.path.join(self.question_log_dir, "cte_message_final_with_reasoning.json")
        final_sql_path = os.path.join(self.question_log_dir, f"{self.question_id}.sql")
        final_exec = os.path.join(self.question_log_dir, f"{self.question_id}_exec.json")

        element_log_dir = self.question_log_dir.split("/")
        idx_run = int(element_log_dir[-1])
        question_id = element_log_dir[-2]
        assert question_id == self.question_id
        # Copy previous log file
        # because previous log file contain log about generate cte and sql
        if not os.path.exists(os.path.join(self.question_log_dir,"log_previous_run.txt")):
            shutil.copyfile(os.path.join(history_previous_run_folder, self.question_id, str(idx_run), "log.txt"), os.path.join(self.question_log_dir,"log_previous_run.txt"))
        history_previous_run_path = os.path.join(history_previous_run_folder, self.question_id, str(idx_run), "cte_message_final.json")
        history_previous_run_path_with_reasoning = os.path.join(history_previous_run_folder, self.question_id, str(idx_run), "cte_message_final_with_reasoning.json")
        assert os.path.exists(
            history_previous_run_path), f"[PlannerAgent] Previous history file not found: {history_previous_run_path}"
        if os.path.exists(history_previous_run_path):
            with open(history_previous_run_path, 'r', encoding='utf-8') as f:
                self.message = json.load(f)
        self.logger.info(f"[SQLAgent] Loaded previous history from {history_previous_run_folder}")
        valid_response = False
        sql_query = ''
        sql_exec_result = None
        for one_msg in self.message:
            if one_msg['role'] == 'assistant':
                response = one_msg['content']
                valid_response, sql_query, sql_exec_result, fix_sql_str = self.check_valid_response(response)


        with open(history_path, 'w') as f:
            json.dump(self.message, f, indent=4)
        shutil.copyfile(history_previous_run_path_with_reasoning, history_path_with_reasoning)

        self.sql_exec_result = sql_exec_result
        if sql_exec_result['status'] == 'success':
            with open(final_sql_path, 'w') as f:
                f.write(sql_query)
            self.sql_result_msg = sql_exec_result['msg']
            self.sql_query_final = sql_query
        if valid_response:  # SQL not empty and executed successfully
            with open(final_exec, 'w') as f:
                json.dump({sql_query: sql_exec_result}, f, indent=4)
            self.logger.info(f"[SQLAgent] Successfully generated valid SQL for question ID: {self.question_id}")
            print(f"[SQLAgent] Successfully generated valid SQL for question ID: {self.question_id}")
        # import pdb; pdb.set_trace()
        else:
            self.logger.info(f"[SQLAgent] Fail to generated valid SQL for question ID: {self.question_id}")
        # if valid_response and self.revise_sql is not None:
        # if sql_exec_result['status'] == 'success' and self.revise_sql is not None:
        if self.revise_sql is not None:
            schema_text = self.get_schema_text()
            cte_text = self.get_cte_text()
            dialect2 = self.cte_agents_list[0].schema_text_infor.dialect2
            self.revise_sql_func(schema_text, cte_text, dialect2, sql_query, sql_exec_result)


    def revise_sql_func(self, schema_text, cte_text, dialect2, sql_query, sql_exec_result, name_suffix="" ):
        """
        Revise the SQL query if needed
        Just run revise  when:
        - revise in config
        x - SQL executed successfully : Do not need. Revise all SQL

        :return:
        """
        revise_sql_path = os.path.join(self.question_log_dir, f"{self.question_id}_revise{name_suffix}.sql")
        revise_sql_exec_path = os.path.join(self.question_log_dir, f"{self.question_id}_exec_revise{name_suffix}.json")
        # Revise SQL if needed
        # self.revise_sql.generate_revise_sql(schema_text=schema_text, cte_text=cte_text, dialect2=dialect2)
        history_path = os.path.join(self.question_log_dir, f"revise_sql{name_suffix}.json")
        history_path_with_reasoning = os.path.join(self.question_log_dir, f"revise_sql_with_reasoning{name_suffix}.json")
        self.logger.info(f"[ReviseSQL] Generating revise for question ID: {self.question_id}")
        # if "msg" in sql_exec_result:
        #     SQL_OUTPUT = sql_exec_result['msg']
        # else:
        #     SQL_OUTPUT = sql_exec_result['error_msg']
        SQL_OUTPUT = sql_exec_result.get("msg", sql_exec_result.get("error_msg", "") )
        if len(SQL_OUTPUT) > 3000:
            SQL_OUTPUT = SQL_OUTPUT[:3000] + "\n... (truncated)"
        request_kwargs = {
            "SCHEMA_STR": schema_text,
            "CTE_STR": cte_text,
            "EXTERNAL_KNOWLEDGE_STR": self.ext_knowledge_str,
            "QUESTION_TEXT": self.question_text,
            "DIALECT2": dialect2,
            "SQL_QUERY" : sql_query,
            "SQL_OUTPUT": SQL_OUTPUT,
        }
        message = [{"role": "user", "content": self.revise_sql.template.format(**request_kwargs)}]
        message_with_reasoning = message.copy()
        fix_time = 0
        valid_response = False
        sql_query = ''
        sql_exec_result = None
        while fix_time < self.max_fix_attempt and (not valid_response):
            fix_time += 1
            message_input_llm = []
            if len(message) > 6: # 3 turn
                # Nếu sau quá 3 turn mà k được thì chỉ lấy 2 turn cuối
                message_input_llm = [message[0]] + message[-4:]
            else:
                message_input_llm = message
            response = self.revise_sql.llm.get_model_response_format(messages_inp=message_input_llm , code_format_list=['sql'])
            valid_response, sql_query, sql_exec_result, fix_sql_str = self.check_valid_response(response)
            del message_input_llm
            message.append({"role": "assistant", "content": response})
            message.append({"role": "user", "content": fix_sql_str})
            message_with_reasoning.append({"role": "assistant", "content": response, "reasoning": self.revise_sql.llm.reasoning})
            message_with_reasoning.append({"role": "user", "content": fix_sql_str})
            self.logger.info(f"[ReviseSQL]  {self.question_id} Attempt {fix_time} to generate valid SQL. fix_response_str={fix_sql_str}")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(message, f, ensure_ascii=False, indent=4)
        with open(history_path_with_reasoning, 'w', encoding='utf-8') as f:
            json.dump(message_with_reasoning, f, ensure_ascii=False, indent=4)
        self.logger.info(f"[ReviseSQL] Successfully generated plan for question ID: {self.question_id}")
        if sql_exec_result['status'] == 'success':
            with open(revise_sql_path, 'w') as f:
                f.write(sql_query)
        if valid_response:  # SQL not empty and executed successfully
            with open(revise_sql_exec_path, 'w') as f:
                json.dump({sql_query : sql_exec_result}, f, indent=4)


    def get_schema_text(self):
        """
        Get the filtered schema context based on the CTEs generated by the CTE agents.
        :return:
        """
        if not self.is_get_schema_text_from_cte_agent:
            self.logger.info("[SQLAgent] do not get schema text from CTE agents, return empty schema text.")
            return ""
        schema_text = ''
        for cte_agent in self.cte_agents_list:

            if cte_agent.schema_text_infor.estimate_num_token_schema_text < 2000:
            # if cte_agent.schema_text_infor.estimate_num_token_schema_text < 500:
            # if cte_agent.schema_text_infor.estimate_num_token_schema_text < 200:

                self.logger.info("[SQLAgent] Schema text is short, using full schema text.")
                schema_text += cte_agent.schema_text_infor.schema_text
            else:   # Nếu dài quá thì lấy filtered_schema_text_infor : tức chỉ lấy cột được chọn thôi
                schema_text += cte_agent.filtered_schema_text_infor.schema_text
        return schema_text
    def get_cte_text(self):
        """

          :
        - What information the CTE table could provide
        - CTE query
        - CTE execution result
        :return:
        """
        if not self.is_get_cte_text_from_cte_agent:
            self.logger.info("[SQLAgent] do not get CTE text from CTE agents, return empty CTE text.")
            return ""
        ext_cte = '' #
        num_exec_cte = 0
        have_relevant_value = False
        existing_cte_name = []
        for cte_agent in self.cte_agents_list:
            self.logger.info(f"[SQLAgent] Processing CTE part {cte_agent.idx_part}")
            final_json = cte_agent.final_json
            final_cte = cte_agent.final_cte
            if final_json['json_answer'] == 'Y':
                table_names = list(final_json['json_columns_dict'].keys())
                think = final_json['json_think']
                ext_cte += f"-- The table {table_names} is selected because {think}\n"
                if 'cte_exec_result_dict' in final_cte and len(final_cte['cte_exec_result_dict']) > 0:
                    cte_sql_query = final_cte['cte_sql_query']
                    ext_cte += f"From the table {table_names}, the following CTE is defined could help to answer the question:\n"
                    ext_cte += f"```cte\n{cte_sql_query}\n```\n"
                    ext_cte += f"The execution result of each component in the CTE is:\n"
                    for cte_name in final_cte['cte_exec_result_dict']:
                        if cte_name.lower() in existing_cte_name:
                            continue
                        existing_cte_name.append(cte_name.lower())
                        cte_sql_query_this_name = final_cte['cte_exec_result_dict'][cte_name]['cte_sql']
                        cte_exec_result = final_cte['cte_exec_result_dict'][cte_name]['msg'][:500]
                        ext_cte += f"```cte\nWITH {cte_name} AS \n ({cte_sql_query_this_name}\n)\n```\n"
                        ext_cte += f"The execution result of SQL query in this CTE is:\n"
                        # ext_cte += f"-- CTE Name: {cte_name}\n"
                        ext_cte += f"\n{cte_exec_result}\n"
                    num_exec_cte += 1
            # Add result from LSH based relevant value extraction
            if cte_agent.relevant_values_text.strip() != "":
                ext_cte += cte_agent.relevant_values_text + "\n"
                have_relevant_value = True
        if have_relevant_value:
            ext_cte += """Because of potential spelling mistakes, use this relevant values information to write the final SQL query so that it aligns more accurately with the values stored in the database.
Use these relevant values to revise the format and spelling mistakes of value in the CTE if any or check whether the column contains such values.
"""
        if num_exec_cte > 0: # tránh trường hợp không có table nào được chọn
            ext_cte += "You could use above information to help you write the final SQL query to answer the question.\n Symbol ; separate value in a row of CTE execution result.\n"
            ext_cte += """This CTE (common table expression) is output from previous CTEagents' execution, when this agent just see part of the database schema.
Each CTE includes information related to the question.
You could use CTE and their results to help you get more insights about the database and write better SQL queries.
"""
        return ext_cte


    def check_valid_response(self, response: str):
        sql_blocks = extract_all_blocks(response, 'sql')
        valid_sql,sql_query,  sql_exec_result, fix_sql_str = self.check_valid_sql(sql_blocks)
        return valid_sql,sql_query, sql_exec_result, fix_sql_str
    def check_valid_sql(self, sql_blocks: list):
        if len(sql_blocks) == 1:
            sql_query = sql_blocks[0]
            valid_sql = True
            fix_sql_str = ''
        elif len(sql_blocks) == 0:
            # No cte block found. Ask model to revise
            # import pdb; pdb.set_trace()
            sql_query = ''
            valid_sql = False
            fix_sql_str = "The response does not contain a SQL block. " \
                           "Please  include a CTE block in the ```sql ...  ``` format." \
                           "Please regenerate the response."
        else:
            # Multiple cte blocks found. Ask model to revise

            sql_query = sql_blocks[-1]
            valid_sql = False
            fix_sql_str = "The response contains multiple SQL blocks. " \
                           "Please provide only one SQL block to answer the question in the ```sql ...  ``` format." \
                           "Do last SQL block as final answer? " \
                           "Please regenerate the response."
        ex_id = self.question_id

        sql_exec_result = self.sql_env.execute_sql_api(sql_query, ex_id=ex_id,
                                                       api=self.cte_agents_list[0].schema_text_infor.api_type,
                                                       max_len=10000,
                                                       db_id=self.db_id)
        if type(sql_exec_result) is dict and 'status' in sql_exec_result:
            if sql_exec_result['status'] == 'error':
                valid_sql = False
                fix_sql_str += f" The SQL execution resulted in an error: {sql_exec_result['error_msg']}. " \
                               f"Please revise the SQL query to fix the error."
            else:

                # valid_sql = True
                if "No data found for the specified query" in sql_exec_result.get('msg', ''):
                    valid_sql = False
                    fix_sql_str += f"The SQL executed successfully but returned no data. So the SQL query is right syntax but the conditional may be wrong. Checking the reason why SQL query returns empty. Please revise the SQL to ensure it returns true data."
        if len(sql_exec_result) == 0:
            valid_sql = False
        return valid_sql,sql_query, sql_exec_result, fix_sql_str



class PlannerAgent:
    """
    Planner
    Agent to plan the generation of SQL query


    """
    def __init__(self, config: Dict[str, Any],
                 question_id: str,
                 question_text: str,
                 db_id: str,
                 ext_knowledge_str: str = "",
                 # Configurations parameters
                 question_log_dir: str = "",):

        # Data must be provided
        self.config = config
        self.question_id = question_id
        self.question_text = question_text
        self.db_id = db_id

        self.ext_knowledge_str = ext_knowledge_str
        # Configurations : config for LLM and template
        self.model_name = config['planner_agent']['model_name']             # MUST have
        self.temperature = config['planner_agent'].get('temperature', 0)
        self.template_path = config['planner_agent']['template_path']       # MUST have
        self.max_tokens = config['planner_agent'].get('max_tokens', 2048)

        self.question_log_dir = question_log_dir

        # Create needed components to execute the agent's task
        self.log_path = os.path.join(self.question_log_dir, "log.txt")
        self.logger = initialize_logger(self.log_path, logger_name=self.question_id)
        self.llm = ChatLLM(model_name=self.model_name, temperature=self.temperature,
                           max_tokens = self.max_tokens, logger = self.logger)
        self.template = open(self.template_path, 'r').read()

        # Output attributes need to be set after sql generation
        self.planning_text = ""
        self.message = []
        self.message_with_reasoning = []

        pass
    def __del__(self):
        self.logger.info(f"[PlannerAgent] Deleting PlannerAgent instance for question ID: {self.question_id}")
        del self.llm
        del self.message
        del self.message_with_reasoning
    def generate_plan(self, schema_text: str, cte_text: str, dialect1: str):
        """
        Call LLM to generate plan question

        :return:
        """
        history_path = os.path.join(self.question_log_dir, "planning.json")
        history_path_with_reasoning = os.path.join(self.question_log_dir, "planning_with_reasoning.json")
        self.logger.info(f"[PlannerAgent] Generating plan for question ID: {self.question_id}")
        request_kwargs = {
            "SCHEMA_STR": schema_text,
            "CTE_STR": cte_text,
            "EXTERNAL_KNOWLEDGE_STR": self.ext_knowledge_str,
            "QUESTION_TEXT": self.question_text,
            "dialect1": dialect1,
        }
        self.message = [{"role": "user", "content": self.template.format(**request_kwargs)}]
        self.message_with_reasoning = self.message.copy()
        response = self.llm.get_model_response_txt(messages=self.message)
        self.planning_text = response
        self.message.append({"role": "assistant", "content": response})
        self.message_with_reasoning.append({"role": "assistant", "content": response, "reasoning": self.llm.reasoning})
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.message, f, ensure_ascii=False, indent=4)
        with open(history_path_with_reasoning, 'w', encoding='utf-8') as f:
            json.dump(self.message_with_reasoning, f, ensure_ascii=False, indent=4)
        self.logger.info(f"[PlannerAgent] Successfully generated plan for question ID: {self.question_id}")
    def load_history(self, history_previous_run_folder: str):
        """

        Load previous message history from file
        Most code similar with function generate_plan()
        :param history_previous_run_folder:
        :return:
        """
        self.logger.info(f"[PlannerAgent] Loading previous history from {history_previous_run_folder}")

        history_path = os.path.join(self.question_log_dir, "planning.json")
        history_path_with_reasoning = os.path.join(self.question_log_dir, "planning_with_reasoning.json")
        element_log_dir = self.question_log_dir.split("/")
        idx_run = int(element_log_dir[-1])
        question_id = element_log_dir[-2]
        assert question_id == self.question_id

        history_previous_run_path = os.path.join(history_previous_run_folder, self.question_id, str(idx_run), "planning.json")
        history_previous_run_path_with_reasoning = os.path.join(history_previous_run_folder, self.question_id, str(idx_run), "planning_with_reasoning.json")
        assert os.path.exists(history_previous_run_path), f"[PlannerAgent] Previous history file not found: {history_previous_run_path}"
        if os.path.exists(history_previous_run_path):
            with open(history_previous_run_path, 'r', encoding='utf-8') as f:
                self.message = json.load(f)
        assert self.message[0]['role'] == 'user', "[PlannerAgent] The first message must be from user."
        assert self.ext_knowledge_str in self.message[0]['content'], "[PlannerAgent] The external knowledge string must be in the first user message."
        assert self.question_text in self.message[0]['content'], "[PlannerAgent] The question text must be in the first user message."
        for one_msg in self.message:
            if one_msg['role'] == 'assistant':
                self.planning_text = one_msg['content']

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.message, f, ensure_ascii=False, indent=4)
        shutil.copyfile(history_previous_run_path_with_reasoning, history_path_with_reasoning)
        self.logger.info(f"[PlannerAgent] Loaded previous history from {history_previous_run_folder}")


class ReviseSQL:
    """
    Revisor
    Revise the SQL query based on the execution result
    """
    def __init__(self, config: Dict[str, Any],
                 question_id: str,
                 question_text: str,
                 db_id: str,
                 ext_knowledge_str: str = "",
                 # Configurations parameters
                 question_log_dir: str = "",):

        # Data must be provided
        self.config = config
        self.question_id = question_id
        self.question_text = question_text
        self.db_id = db_id

        self.ext_knowledge_str = ext_knowledge_str
        # Configurations : config for LLM and template
        self.model_name = config['revise_sql']['model_name']             # MUST have
        self.temperature = config['revise_sql'].get('temperature', 0)
        self.template_path = config['revise_sql']['template_path']       # MUST have
        self.max_tokens = config['revise_sql'].get('max_tokens', 2048)

        self.question_log_dir = question_log_dir

        # Create needed components to execute the agent's task
        self.log_path = os.path.join(self.question_log_dir, "log.txt")
        self.logger = initialize_logger(self.log_path, logger_name=self.question_id)
        self.llm = ChatLLM(model_name=self.model_name, temperature=self.temperature,
                           max_tokens = self.max_tokens, logger = self.logger)
        self.template = open(self.template_path, 'r').read()

        # Output attributes need to be set after sql generation
        self.message = []
        self.message_with_reasoning = []

    def __del__(self):
        self.logger.info(f"[ReviseSQL] Deleting PlannerAgent instance for question ID: {self.question_id}")
        del self.llm
        del self.message
        del self.message_with_reasoning
