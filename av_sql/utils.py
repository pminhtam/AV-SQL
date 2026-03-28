"""


"""


import os
import json
import logging
import glob
import random

def read_jsonl(file_path):
    json_list_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
    # with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            json_list_data.append(json.loads(line))
        # return [json.loads(line) for line in file]
    return json_list_data

import threading
def initialize_logger(log_path, logger_name=None):
    """

    Source : ReFoRCE/utils.py

    :param log_path:
    :param logger_name:
    :return:
    """
    if logger_name is None:
        logger_name = threading.current_thread().name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, mode='a+') # Avoid deleting previous logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    # https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
    logger.addHandler(logging.StreamHandler())  # to print to console as well
    return logger


def extract_all_blocks(text: str, code_format: str):
    """
    source : ReFoRCE/utils.py
    Extract all code blocks of a specific format from the given text.

    Args:
        text (str): The input text containing code blocks.
        code_format (str): The format of the code blocks to extract (e.g., 'sql', 'python').

    Returns:
        list: A list of extracted code blocks.
    """
    # pattern = rf"```{code_format}\s*(.*?)\s*```"
    # matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    # return [match.strip() for match in matches]
    format_blocks = []
    start = 0
    extract_time = 0    # BUG09112025 : infinite loop
    if type(text) is not str:
        # local_bird_1196 error sql_query_start = text.find(f"```{code_format}", start)
        # AttributeError: 'NoneType' object has no attribute 'find'
        ## When text == None , type(text) == <class 'NoneType'>
        # import pdb; pdb.set_trace()
        return format_blocks
    while True: # Keep extracting until no more blocks are found

        sql_query_start = text.find(f"```{code_format}", start)
        if sql_query_start == -1:
            break

        sql_query_end = text.find("```", sql_query_start + len(f"```{code_format}"))
        if sql_query_end == -1:
            break

        sql_block = text[sql_query_start + len(f"```{code_format}"):sql_query_end].strip()
        if len(sql_block) > 0:
            # Avoid empty block e.g : ```cte\n```\n\n```
            format_blocks.append(sql_block)

        start = sql_query_end + len("```")
        extract_time += 1
        if extract_time >= 10:  # Limit to 10 extractions to prevent infinite
            break
    return format_blocks


def get_in_context_examples(num_incontext_shot, incontext_shot_folder_path):
    """
    Get in-context examples for CTE generation
    For multi-shot prompting
    For now it use in :
    - CTEAgent
    - RephraseQuestionAgent
    :return:
    """
    example_block_text = ""
    if num_incontext_shot <=0:
        """
        No in-context examples needed
        """
        return example_block_text
    else:
        assert os.path.exists(incontext_shot_folder_path), f"incontext_shot_folder_path {incontext_shot_folder_path} not exists."
        sample_file_path_list = glob.glob(os.path.join(incontext_shot_folder_path, "*"))
        assert len(sample_file_path_list) > 0 , f"No sample file found in {incontext_shot_folder_path}"
        # Randomly choose num_incontext_shot files
        example_block_text += "\n"
        choosen_sample_file_path = random.sample(sample_file_path_list, min(num_incontext_shot, len(sample_file_path_list)))
        for sample_file_path in choosen_sample_file_path:
            with open(sample_file_path, 'r', encoding='utf-8') as f:
                sample_text = f.read()
                example_block_text += sample_text + "\n\n"
        # import pdb; pdb.set_trace()
        return example_block_text


