"""


This code handle one question




"""
import glob
import os
import random
import json
from typing import List, Dict, Any, Tuple
import math
from transformers import AutoTokenizer
import shutil

from .database_schema_manager import DatabaseSchemaManager
from .cte_agent import CTEAgent
from .sql_agent import SQLAgent, PlannerAgent, ReviseSQL
from .utils import initialize_logger, get_in_context_examples
from .chat import ChatLLM
from .database_schema_manager import TextInforManager
from .sql_exec_env import SqlExecEnv

class QuestionInstance:
    def __init__(self, config: Dict[str, Any],
                 question_id: str, question_text: str,
                 ext_knowledge: str,
                 db_id: str,
                 idx_run: int,
                 database_schema_manager: DatabaseSchemaManager
                 ):
        # Data must be provided
        self.config = config
        self.question_id = question_id
        self.question_text = question_text
        self.ext_knowledge = ext_knowledge  # with spider2: path to md file

        self.db_id = db_id

        # Configurations :
        self.log_dir = config.get('log_dir', "log/test_log") #  log, file .sql
        self.question_log_dir = os.path.join(self.log_dir, self.question_id, str(idx_run))  #  log, file .sql
        self.log_path = os.path.join(self.question_log_dir, "log.txt")
        self.dataset_name = config.get('dataset_name', 'spider2')
        os.makedirs(self.question_log_dir, exist_ok=True)
        # Create needed components to execute the agent's task

        self.cte_agent_messages_dict = {}   # store messages history for each CTE agent corresponding to each part of DatabaseSchema
        self.database_schema_manager = database_schema_manager    # DatabaseSchemaManager object
        self.logger = initialize_logger(self.log_path, logger_name=self.question_id)

        # Object in run time
        self.cte_agents_list = []
        self.sql_agent = None
        self.planner_agent = None
        self.revise_sql = None

    def run(self):
        if self.config.get('is_continue_run', False):   #
            # if len(glob.glob(self.question_log_dir + "/*.sql")) > 0:
            # if len(glob.glob(self.question_log_dir + f"/{self.question_id}_exec.json")) > 0:    # SQL not empty and executed successfully
            if len(glob.glob(self.question_log_dir + "/revise_sql*.json")) > 0:    # run full pipeline with revise sql
            # if len(glob.glob(self.question_log_dir + f"/{self.question_id}*_exec_revise*.json")) > 0:    # SQL not empty and executed successfully
            # if len(glob.glob(self.question_log_dir + f"/{self.question_id}*_exec_revise*.json")) > 8:    # multi candidates
                print(f"[QuestionInstance] : Log file already exists at {self.log_path} with {len(glob.glob(self.question_log_dir + f'/{self.question_id}*_exec_revise*.json'))} sql. Skipping execution.")
                # self.logger.info(f"[QuestionInstance] : Log file already exists at {self.log_path}. Skipping execution.")
                return
        # import pdb; pdb.set_trace()
        self.logger.info(f"[QuestionInstance] : Start question instance :id={self.question_id}, db_id={self.db_id}, question='{self.question_text}")

        ext_knowledge_str = ""  # Content external cho vào prompt

        if "rephrase_agent" in self.config:
            self.logger.info(f"[QuestionInstance] {self.question_id}: Start Rephrased question")
            rephrase_agent = RephraseQuestionAgent(
                config=self.config,
                question_id=self.question_id,
                question_text=self.question_text,
                db_id=self.db_id,
                ext_knowledge=self.ext_knowledge,
                question_log_dir=self.question_log_dir,
            )
            if self.config['rephrase_agent'].get("history_previous_run_folder", "") != "":
                try:
                    rephrase_agent.load_history(self.config['rephrase_agent']["history_previous_run_folder"])
                except Exception as e:
                    self.logger.error(f"[QuestionInstance] {self.question_id} : Failed to load previous history. Error: {e}. Regenerate Rephrased question.")
                    rephrase_agent.generate_rephrase()
            else:
                rephrase_agent.generate_rephrase()
            external_knowledge_str_list = rephrase_agent.external_knowledge_str_list
            # Thêm ext knowledge vào prompt nếu nó ngắn gọn
            if len(external_knowledge_str_list) == 1 and len(external_knowledge_str_list[0]) < 300 and len(external_knowledge_str_list[0]) >0:
                ext_knowledge_str += external_knowledge_str_list[0]
            ext_knowledge_str += "\nThe question can be further explained in more detail as follows:\n" +rephrase_agent.rephrase_result
            self.logger.info(f"[QuestionInstance] {self.question_id}: Done Rephrased question")
            del rephrase_agent
        else:
            ext_knowledge_str = ""
            if self.dataset_name == "spider2" or "spider2" in self.dataset_name:
                # Read file
                external_knowledge_folder_path = self.config.get('external_knowledge_folder_path', './')
                if self.ext_knowledge:  # self.ext_knowledge not None
                    external_knowledge_path = os.path.join(external_knowledge_folder_path, self.ext_knowledge)
                    if os.path.isfile(external_knowledge_path) and external_knowledge_path.lower().endswith("md"):
                        with open(external_knowledge_path, 'r', encoding='utf-8') as f:
                            ext_knowledge_str = f.read()
            else:
                ext_knowledge_str = self.ext_knowledge
        schema_dict_dbid = self.database_schema_manager.db_schema_dict_all[self.db_id]
        group_of_tables_dbid = self.database_schema_manager.group_of_tables_dict_all[self.db_id]
        group_of_columns_dbid = self.database_schema_manager.group_of_columns_dict_all[self.db_id]

        prompt_type = self.config['cte_agent'].get('prompt_type', '')
        if  prompt_type == "compact" or prompt_type == "compress_table" :
            # just have valuable in Spider2.0
            schema_dict_dbid_filtered = {}
            for db_id_schema, infor in schema_dict_dbid.items():
                # filter each table
                schema_text_infor = TextInforManager(
                    question_id="local_test_question",
                    schema_dict={db_id_schema: infor},
                    group_of_tables=group_of_tables_dbid,
                    group_of_columns=group_of_columns_dbid,
                    prompt_type=prompt_type,
                    is_use_sample_rows=True
                )
                if len(schema_text_infor.schema_text) > 0:
                    schema_dict_dbid_filtered[db_id_schema] = infor
                # print(len(schema_text_infor.schema_text))
                del schema_text_infor
            # replace original schema_dict_dbid with schema_dict_dbid_filtered filter redundance table
            schema_dict_dbid = schema_dict_dbid_filtered
        schema_split_type = self.config.get("schema_split_type", "one_table")
        schema_parts = DatabaseSchemaManager.split_schema(schema_dict_dbid, split_type=schema_split_type,
                                                          token_limit=self.config.get('schema_part_token_limit', 30000),
                                                          is_use_col_desc=self.config['cte_agent'].get('is_use_col_desc', True))
        random.shuffle(schema_parts)   # shuffle to avoid any bias
        print(f"[QuestionInstance] {self.question_id} mode {self.config['cte_agent']['model_name']} : Start question instance. have {len(schema_parts)} part ")
        self.logger.info(f"[QuestionInstance] {self.question_id}: Start question instance. have {len(schema_parts)} part ")
        max_cte_try_to_run = 3
        cte_try = 0
        table_used = []
        while cte_try < max_cte_try_to_run and len(table_used) == 0:
            """
            Try run cte until filtered table_used > 0
            """
            cte_try +=1
            table_used = []
            if "is_use_sample_rows" not in self.config['cte_agent']:
                self.config['cte_agent']['is_use_sample_rows'] = random.choice([True, False])
            if "is_use_col_desc" not in self.config['cte_agent']:
                self.config['cte_agent']['is_use_col_desc'] = random.choice([True, False])
            for idx_part, schema_part in enumerate(schema_parts):
                cte_agent = CTEAgent(
                    config=self.config,
                    question_id=self.question_id,
                    question_text=self.question_text,
                    db_id = self.db_id,
                    idx_part = idx_part,
                    schema_dict=schema_part,
                    group_of_tables=group_of_tables_dbid,
                    group_of_columns=group_of_columns_dbid,
                    cte_agents_list=self.cte_agents_list,
                    ext_knowledge_str=ext_knowledge_str,

                    question_log_dir = self.question_log_dir,

                )
                if len(cte_agent.schema_text_infor.schema_text) > 0:

                    if self.config['cte_agent'].get("history_previous_run_folder", "") != "":
                        try:
                            cte_agent.load_history(self.config['cte_agent']["history_previous_run_folder"])
                        except Exception as e:
                            self.logger.error(f"[QuestionInstance] {self.question_id} part {idx_part} : Failed to load previous history. Error: {e}. Regenerate CTE.")
                            cte_agent.generate_cte()
                            # continue    # méo chạy lại nữa
                    else:
                        cte_agent.generate_cte()
                    self.cte_agents_list.append(cte_agent)
                    table_used.extend(list(cte_agent.filtered_schema_dict.keys()))
                else:

                    self.logger.info(f"[QuestionInstance] {self.question_id} part {idx_part} : Skip CTE generation due to no schema text ")
                    del cte_agent
        self.logger.info(f"[QuestionInstance] {self.question_id}: End CTE generation. Start SQL generation ")

        if "revise_sql" in self.config:
            self.revise_sql = ReviseSQL(
                config=self.config,
                question_id=self.question_id,
                question_text=self.question_text,
                db_id=self.db_id,
                ext_knowledge_str=ext_knowledge_str,
                question_log_dir = self.question_log_dir,
            )

        self.sql_agent = SQLAgent(
            config=self.config,
            question_id=self.question_id,
            question_text=self.question_text,
            db_id=self.db_id,

            cte_agents_list=self.cte_agents_list,
            revise_sql = self.revise_sql,
            ext_knowledge_str=ext_knowledge_str,

            question_log_dir = self.question_log_dir,
        )
        if "planner_agent" in self.config:
            # Use PlannerAgent
            self.planner_agent = PlannerAgent(
                config=self.config,
                question_id=self.question_id,
                question_text=self.question_text,
                db_id=self.db_id,
                ext_knowledge_str=ext_knowledge_str,
                question_log_dir = self.question_log_dir,
            )
            if self.config['planner_agent'].get("history_previous_run_folder", "") != "":
                try:
                    self.planner_agent.load_history(self.config['planner_agent']["history_previous_run_folder"])
                except Exception as e:
                    self.logger.error(f"[QuestionInstance] {self.question_id} : Failed to load previous history. Error: {e}. Regenerate Plan.")
                    self.planner_agent.generate_plan(self.sql_agent.get_schema_text(),
                                                 self.sql_agent.get_cte_text(),
                                                 dialect1 = self.cte_agents_list[0].schema_text_infor.dialect1)
            else:
                self.planner_agent.generate_plan(self.sql_agent.get_schema_text(),
                                             self.sql_agent.get_cte_text(),
                                             dialect1 = self.cte_agents_list[0].schema_text_infor.dialect1)
            planning_text = self.planner_agent.planning_text
        else:
            self.planner_agent = None
            planning_text = ""
        if self.config['sql_agent'].get("history_previous_run_folder", "") != "":
            try:
                self.sql_agent.load_history(self.config['sql_agent']["history_previous_run_folder"])
            except Exception as e:
                self.logger.error(f"[QuestionInstance] {self.question_id} : Failed to load previous history. Error: {e}. Regenerate SQL.")
                if "multi_candidates" in self.config['sql_agent']:
                    num_response = self.config['sql_agent']['multi_candidates'].get('num_response', 3)
                    num_runs = self.config['sql_agent']['multi_candidates'].get('num_runs', 1)
                    for idx_run in range(num_runs):
                        self.sql_agent.generate_multi_candidate_sql(planning_text, idx_run, num_response=num_response)
                else:
                    self.sql_agent.generate_sql(planning_text)
        else:
            if "multi_candidates" in self.config['sql_agent']:
                num_response = self.config['sql_agent']['multi_candidates'].get('num_response', 3)
                num_runs = self.config['sql_agent']['multi_candidates'].get('num_runs', 1)
                for idx_run in range(num_runs):
                    self.sql_agent.generate_multi_candidate_sql(planning_text, idx_run, num_response=num_response)
            else:
                self.sql_agent.generate_sql(planning_text)
        # BUG29112025
        self.logger.info(f"[QuestionInstance] {self.question_id}: End SQL generation ")
        SqlExecEnv.get_instance().close_db_sf(self.question_id) # close db if is snowflake


    def __del__(self):
        for cte_agent in self.cte_agents_list:
            del cte_agent
        del self.sql_agent
        del self.planner_agent
        print(f"[QuestionInstance] : Deleting QuestionInstance id={self.question_id}")


    def __str__(self):
        return f"QuestionInstance(id={self.question_id}, db_id={self.db_id}, question='{self.question_text}')"

class RephraseQuestionAgent:
    """
    Rewriter Agent
    Agent responsible for rephrase question based on:
    - Original question text
    - External knowledge

    """
    def __init__(self, config: Dict[str, Any],
                 question_id: str, question_text: str,
                 db_id: str,
                 ext_knowledge: str = "",
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
        """
        ext_knowledge :  
        """
        self.ext_knowledge = ext_knowledge


        # Configurations : config for LLM and template
        self.token_limit = config.get('schema_part_token_limit', 2048) # For split long ext know

        self.dataset_name = config.get('dataset_name', 'spider2')
        self.external_knowledge_folder_path = config.get('external_knowledge_folder_path', './')

        self.model_name = config['rephrase_agent']['model_name']             # MUST have
        self.temperature = config['rephrase_agent'].get('temperature', 0)
        self.template_path = config['rephrase_agent']['template_path']       # MUST have

        self.question_log_dir = question_log_dir

        # Create needed components to execute the agent's task
        self.template = open(self.template_path, 'r').read() if self.template_path is not None else ""
        self.log_path = os.path.join(self.question_log_dir, "log.txt")


        self.logger = initialize_logger(self.log_path, logger_name=self.question_id)
        self.llm = ChatLLM(model_name=self.model_name, temperature=self.temperature, logger = self.logger)

        num_incontext_shot = self.config['rephrase_agent'].get('num_incontext_shot', 0)
        incontext_shot_folder_path = self.config['rephrase_agent'].get('incontext_shot_folder_path', "not_default_path_yet")
        self.example_block_text = get_in_context_examples(num_incontext_shot, incontext_shot_folder_path)

        self.all_message = []
        self.all_message_with_reasoning  = []
        self.rephrase_result = ""
        # self.external_knowledge_str = self.get_external_knowledge()
        self.external_knowledge_str_list = self.get_external_knowledge()

    def generate_rephrase(self):
        """
        Call LLM to generate rephrase question

        :return:
        """
        history_path = os.path.join(self.question_log_dir, "rephrase_question.json")
        history_path_with_reasoning = os.path.join(self.question_log_dir, "rephrase_question_with_reasoning.json")
        self.logger.info(f"[RephraseQuestionAgent] Start generating CTE for question_id: {self.question_id} ")
        print(f"[RephraseQuestionAgent] Start generating CTE for question_id: {self.question_id} ")
        external_knowledge_prompt_add = """The external knowledge is used to help clarify the question about:
        - How to get information to answer the question.
        - What columns could be used to answer the question 
        """
        rephrase_result = self.question_text    #
        for external_knowledge_str in self.external_knowledge_str_list:
            if len(external_knowledge_str) > 0:
                # external_knowledge_prompt_add :
                external_knowledge_str += external_knowledge_prompt_add
            request_kwargs = {
                "EXTERNAL_KNOWLEDGE_STR": external_knowledge_str,
                "EXAMPLES_BLOCK": self.example_block_text,
                # "EXTERNAL_KNOWLEDGE_STR": self.external_knowledge_str,
                "QUESTION_TEXT": self.question_text,
            }
            message = [{"role": "user", "content": self.template.format(**request_kwargs)}]
            message_with_reasoning = message.copy()
            response = self.llm.get_model_response_txt(messages=message)

            rephrase_result = response
            message.append({"role": "assistant", "content": response})
            message_with_reasoning.append({"role": "assistant", "content": response, "reasoning": self.llm.reasoning})
            self.all_message.append(message)
            self.all_message_with_reasoning.append(message_with_reasoning)
        self.rephrase_result = rephrase_result
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_message, f, ensure_ascii=False, indent=4)
        with open(history_path_with_reasoning, 'w', encoding='utf-8') as f:
            json.dump(self.all_message_with_reasoning, f, ensure_ascii=False, indent=4)

    def load_history(self, history_previous_run_folder: str):
        """
        Load previous message history from file

        :param history_previous_run_folder:
        :return:
        """
        history_path = os.path.join(self.question_log_dir, "rephrase_question.json")
        history_path_with_reasoning = os.path.join(self.question_log_dir, "rephrase_question_with_reasoning.json")
        element_log_dir = self.question_log_dir.split("/")
        idx_run = int(element_log_dir[-1])
        question_id = element_log_dir[-2]
        assert question_id == self.question_id

        history_previous_run_path = os.path.join(history_previous_run_folder, self.question_id, str(idx_run),
                                                 "rephrase_question.json")
        history_previous_run_path_with_reasoning = os.path.join(history_previous_run_folder, self.question_id,
                                                                str(idx_run), "rephrase_question_with_reasoning.json")
        assert os.path.exists(
            history_previous_run_path), f"[RephraseQuestionAgent] Previous history file not found: {history_previous_run_path}"
        if os.path.exists(history_previous_run_path):
            with open(history_previous_run_path, 'r', encoding='utf-8') as f:
                self.all_message = json.load(f)
        # import pdb; pdb.set_trace()
        self.message = self.all_message[-1]   # load the last message
        assert self.message[0]['role'] == 'user', "[Rewriter] The first message must be from user."
        assert self.question_text in self.message[0][
            'content'], "[RephraseQuestionAgent] The question text must be in the first user message."
        for one_msg in self.message:
            if one_msg['role'] == 'assistant':
                self.rephrase_result = one_msg['content']

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_message, f, ensure_ascii=False, indent=4)
        shutil.copyfile(history_previous_run_path_with_reasoning, history_path_with_reasoning)
        self.logger.info(f"[Rewriter] Loaded previous history from {history_previous_run_folder}")

    def get_external_knowledge(self):
        """
        Get external knowledge:
        - Spider : no external knowledge
        - BIRD : external knowledge = str evidence
        - Spider2.0 : read external knowledge file

        :return:
        """
        external_knowledge_str_list = []
        if self.dataset_name == "spider2" or "spider2" in self.dataset_name:
            # Read file
            #  : self.ext_knowledge = None
            if self.ext_knowledge:  # self.ext_knowledge not None
                external_knowledge_path = os.path.join(self.external_knowledge_folder_path, self.ext_knowledge)
                if os.path.isfile(external_knowledge_path) and external_knowledge_path.lower().endswith("md"):
                    with open(external_knowledge_path, 'r', encoding='utf-8') as f:
                        external_knowledge_str = f.read()
                    token_ids = DatabaseSchemaManager.get_instace().tokenizer.encode(external_knowledge_str, add_special_tokens=False)
                    external_knowledge_str_len_token = len(token_ids)
                    num_ext_know_part = math.ceil(external_knowledge_str_len_token/self.token_limit)
                    if num_ext_know_part > 1:
                        """
                        Split external_knowledge_str into several parts.
                        Split part by line : divide each part to equal line each part 
                        """
                        external_knowledge_str_split = external_knowledge_str.split("\n")
                        num_line_each_ext_know_part = math.ceil(len(external_knowledge_str_split)/num_ext_know_part)
                        idx_line_part = 0
                        ext_know_part_str = ""
                        for line in external_knowledge_str_split:
                            idx_line_part += 1
                            ext_know_part_str += "\n" + line
                            if idx_line_part % num_line_each_ext_know_part == 0:
                                external_knowledge_str_list.append(ext_know_part_str)
                                ext_know_part_str = ""
                    else:
                        external_knowledge_str_list.append(external_knowledge_str)
        else:
            external_knowledge_str = self.ext_knowledge
            external_knowledge_str_list.append(self.ext_knowledge)
        # print(len(external_knowledge_str_list))
        if len(external_knowledge_str_list) == 0:
            external_knowledge_str_list.append("")
        return external_knowledge_str_list
