"""

chat model (gpt-3.5-turbo, gpt-4, ...)






"""

import os
import re
import time
import copy
from typing import List, Dict, Any, Tuple

from openai import OpenAI
from .utils import extract_all_blocks

class ChatLLM:
    def __init__(self, model_name='gpt-3.5-turbo', temperature=0.0, max_fix_attempt=3, max_tokens=2048, stream=False ,logger=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_fix_attempt = max_fix_attempt
        self.max_tokens = max_tokens
        self.stream = stream
        self.messages_ori = []
        self.api_key = self.get_api_key(model_name)
        self.client = self.get_client(model_name)
        self.logger = logger

        # Output contain
        self.reasoning = ""
    def __del__(self):
        del self.client
        del self.api_key
        del self.messages_ori

    def get_model_response_format(self, messages_inp: list[Dict], code_format_list=['sql']):
        """
        Re call LLM until get correct sql format or reach max_trys
        Do not append error message to history, only append final response
        """
        # self.messages.append({"role": "user", "content": prompt})
        fix_time = 0
        # main_content = ""
        self.reasoning = ""
        # need to use deepcopy
        self.messages_ori = copy.deepcopy(messages_inp)
        messages = copy.deepcopy(messages_inp)  #
        is_response_correct_format = False
        time_sleep_reset_seconds = 10
        while not is_response_correct_format and fix_time < self.max_fix_attempt:
            fix_time += 1
            fail_llm_calling = 0
            success_llm_calling = False
            self.logger.info("[ChatLLM] : start chat ")
            while fail_llm_calling < 7 and not success_llm_calling:
                try:
                    if "mistral-" in self.model_name.lower():
                        response = self.client.chat.complete(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=self.temperature,
                            # extra_body={"chat_template_kwargs": {"thinking": False}},
                            # extra_body={"reasoning": {"enabled": False}},
                            # reasoning_effort=None,
                            stream=self.stream,
                        )
                    if self.stream:
                        choices = True    # stream=True
                    else:
                        choices = response.choices    # not-stream | stream=False
                    if choices:
                        ## Enable stream response processing # stream=True
                        main_content = ""
                        self.reasoning = ""
                        if self.stream:
                            for delta in response:
                                # print("finish_reason  : ",delta.choices[0].finish_reason)
                                if not getattr(delta, "choices", None): # check if choices exist
                                    continue
                                if not delta.choices[0].finish_reason:
                                    word = delta.choices[0].delta.content or ""
                                    # import pdb; pdb.set_trace()
                                    word_reasong = getattr(delta.choices[0].delta, 'reasoning', "")
                                    if word_reasong == "":
                                        word_reasong = getattr(delta.choices[0].delta, 'reasoning_content', "") # deepseekr1 - nvidia-NIM , kimik2thinking - NIM
                                    if word_reasong == "":
                                        word_reasong = getattr(delta.choices[0].delta, 'reasoning_details', [{'text': ''}])[0][
                                            'text']  # gptoss20b - openrouter
                                    main_content = main_content + word if word else main_content
                                    self.reasoning = self.reasoning + word_reasong if word_reasong else self.reasoning
                            ##############################
                        else:
                            ############################# | stream=False
                            main_content = choices[0].message.content  # output
                            self.reasoning = getattr(choices[0].message, 'reasoning', "")  # reasoning gptoss120b - nvidia-NIM
                            if self.reasoning == "":
                                self.reasoning = getattr(choices[0].message, 'reasoning_content', "") # deepseekr1 - nvidia-NIM
                            if self.reasoning == "":
                                self.reasoning = getattr(choices[0].message, 'reasoning_details', [{'text': ''}])[0][
                                    'text']  # gptoss20b - openrouter
                            #############################
                        format_blocks_all = []
                        for code_format_item in code_format_list:
                            format_blocks = extract_all_blocks(main_content, code_format_item)
                            format_blocks_all.extend(format_blocks)
                        is_response_correct_format = True
                        if len(format_blocks_all) == 0:
                            is_response_correct_format = False
                        if self.logger:
                            self.logger.info(f"[ChatLLM] : successfully get response from LLM with len(format_blocks_all) = {len(format_blocks_all)} ")
                        success_llm_calling = True
                    else:
                        if self.logger:
                            self.logger.info(f"[ChatLLM] get_model_response_format  : Empty choices in response : {str(response)}")
                        main_content = ""
                        fail_llm_calling += 1
                        success_llm_calling = False
                        if len(str(response.error)) > 0 and "tokens" in str(response.error):
                            # time.sleep(60)
                            error_message = str(response.error)
                            prompt_truncate = self.truncate_message_history(error_message, messages)
                            if len(prompt_truncate) > 0:
                                if self.logger:
                                    self.logger.info(
                                    f"[ChatLLM]  truncate from len {len(messages[0]['content'])} to len {len(prompt_truncate)}")
                                messages[0]['content'] = prompt_truncate
                except Exception as err:

                    if self.logger:
                        self.logger.error(f"[ChatLLM] get_model_response_format Exception : {err}")
                    error_message = str(err)
                    prompt_truncate = self.truncate_message_history(error_message, messages)
                    if len(prompt_truncate) > 0:
                        if self.logger:
                            self.logger.info(
                            f"[ChatLLM]  truncate from len {len(messages[0]['content'])} to len {len(prompt_truncate)}")
                        messages[0]['content'] = prompt_truncate
                    try:
                        time_sleep_reset_seconds = getattr(err, "body", {'reset_seconds': 10}).get("reset_seconds", 10)
                    except AttributeError as eeeeee:
                        time_sleep_reset_seconds = 10
                    time.sleep(60 + time_sleep_reset_seconds)
                    fail_llm_calling += 1
                    success_llm_calling = False
                    main_content = ""
                finally:
                    if self.logger:
                        self.logger.info(f"[ChatLLM] finally get_model_response_format  {fail_llm_calling} ")
                    if fail_llm_calling > 3 and not success_llm_calling:    # =3 thì tạo lại
                        del self.client
                        time.sleep(60 * 3 + time_sleep_reset_seconds)
                        self.client = self.get_client(self.model_name)
                        if self.logger:
                            self.logger.info(f"[ChatLLM] fail too many times: {fail_llm_calling} times, change LLM client")
            if type(main_content) is not str or len(main_content) == 0:

                if self.logger:
                    self.logger.info("[ChatLLM] get_model_response_format : main_content is empty or not str")
                is_response_correct_format = False
                continue
            if not is_response_correct_format and  len(main_content) > 0:  # Sai format thì tiếp tục hỏi

                if self.logger:
                    self.logger.info(f"[ChatLLM]  {str(code_format_list)}  : fix_time: {fix_time}")
                messages.append(
                    {"role": "assistant", "content": main_content})  # đưa cả history để prompt cho những turn sau
                fix_content = f"The previous response is not in the correct format. Please answer in format :"
                for code_format_item in code_format_list:
                    fix_content += f"\n{code_format_item} within  ```{code_format_item} and ``` "
                messages.append({"role": "user", "content": fix_content})

        del messages
        return main_content

    def get_model_response_format_multi_candidate(self, messages_inp: list[Dict],n=3):
        """
        same with function get_model_response_format()
        but add n in parameter

        Input: prompt (str) --> câu hỏi + schema + lịch sử hội thoại
        Output: main_content (str)

        """

        """
        Re call LLM until get correct sql format or reach max_trys
        Do not append error message to history, only append final response
        """
        # self.messages.append({"role": "user", "content": prompt})
        # main_content = ""
        self.reasoning = ""
        # messages_inp là 1 list of dict -> use .copy() do not Creates a completely independent copy of all nested structures
        # need to use deepcopy
        self.messages_ori = copy.deepcopy(messages_inp)
        messages = copy.deepcopy(messages_inp)  # tránh thay đổi messages gốc
        time_sleep_reset_seconds = 10
        fail_llm_calling = 0
        success_llm_calling = False
        all_response = {}
        while fail_llm_calling < 7 and not success_llm_calling:
            try:
                # import pdb; pdb.set_trace()
                if "mistral-" in self.model_name.lower():
                    response = self.client.chat.complete(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        n=n,  # This requests 3 different completion choices
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        # extra_body={"chat_template_kwargs": {"thinking": True}},
                        # extra_body={"reasoning": {"enabled": False}},
                        # reasoning_effort=None,
                        stream=self.stream,
                        n=n,  # This requests 3 different completion choices

                    )
                if self.stream:
                    choices = True    # stream=True
                else:
                    choices = response.choices    # not-stream | stream=False
                if choices:
                    ## Enable stream response processing # stream=True
                    main_content = ""
                    self.reasoning = ""
                    if self.stream:
                        for delta in response:
                            # print("finish_reason  : ",delta.choices[0].finish_reason)
                            if not getattr(delta, "choices", None): # check if choices exist
                                continue
                            for idx_choice in range(len(delta.choices)):
                                if not delta.choices[idx_choice].finish_reason:
                                    word = delta.choices[idx_choice].delta.content or ""
                                    choice_index = delta.choices[idx_choice].index
                                    # print(delta.choices)
                                    word_reasong = getattr(delta.choices[idx_choice].delta, 'reasoning', "")
                                    if word_reasong == "":
                                        word_reasong = getattr(delta.choices[idx_choice].delta, 'reasoning_content', "") # deepseekr1 - nvidia-NIM , kimik2thinking - NIM
                                    if word_reasong == "":
                                        word_reasong = getattr(delta.choices[idx_choice].delta, 'reasoning_details', [{'text': ''}])[0][
                                            'text']  # gptoss20b - openrouter
                                    if choice_index not in all_response:
                                        all_response[choice_index] = {'content': str(word), 'reasoning': str(word_reasong)}
                                    else:
                                        all_response[choice_index]['content'] = all_response[choice_index]['content'] + str(word)
                                        all_response[choice_index]['reasoning'] = all_response[choice_index]['reasoning'] + str(word_reasong)
                        ##############################
                    else:
                        ############################# | stream=False
                        for idx_choice in range(len(choices)):
                            main_content = choices[idx_choice].message.content  # output
                            choice_index = choices[idx_choice].index
                            reasoning = getattr(choices[idx_choice].message, 'reasoning', "")  # reasoning gptoss120b - nvidia-NIM
                            if reasoning == "":
                                reasoning = getattr(choices[idx_choice].message, 'reasoning_content', "") # deepseekr1 - nvidia-NIM
                            if reasoning == "":
                                reasoning = getattr(choices[idx_choice].message, 'reasoning_details', [{'text': ''}])[0][
                                    'text']  # gptoss20b - openrouter
                            all_response[choice_index] = {'content': main_content, 'reasoning': reasoning}
                        #############################
                    success_llm_calling = True
                else:
                    #  : Empty choices in response
                    if self.logger:
                        self.logger.info(f"[ChatLLM] get_model_response_format_multi_candidate  : Empty choices in response : {str(response)}")
                    fail_llm_calling += 1
                    success_llm_calling = False
                    if len(str(response.error)) > 0 and "tokens" in str(response.error):
                        # time.sleep(60)
                        error_message = str(response.error)
                        prompt_truncate = self.truncate_message_history(error_message, messages)
                        if len(prompt_truncate) > 0:
                            if self.logger:
                                self.logger.info(
                                f"[ChatLLM]  truncate from len {len(messages[0]['content'])} to len {len(prompt_truncate)}")
                            messages[0]['content'] = prompt_truncate
                # print(response.choices[0].message.content)
            except Exception as err:
                print(err)
                if self.logger:
                    self.logger.error(f"[ChatLLM] get_model_response_format_multi_candidate Exception : {err}")
                # print(f"[ChatLLM] get_model_response_format Exception : {err}")
                error_message = str(err)
                prompt_truncate = self.truncate_message_history(error_message, messages)
                if len(prompt_truncate) > 0:
                    if self.logger:
                        self.logger.info(
                        f"[ChatLLM]  truncate from len {len(messages[0]['content'])} to len {len(prompt_truncate)}")
                    messages[0]['content'] = prompt_truncate

                try:
                    time_sleep_reset_seconds = getattr(err, "body", {'reset_seconds': 10}).get("reset_seconds", 10)
                except AttributeError as eeeeee:
                    time_sleep_reset_seconds = 10
                time.sleep(60 + time_sleep_reset_seconds)
                fail_llm_calling += 1
                success_llm_calling = False
                # continue
            finally:
                print("WTF")
                if self.logger:
                    self.logger.info(f"[ChatLLM] finally get_model_response_format_multi_candidate  {fail_llm_calling} ")
                if fail_llm_calling > 3 and not success_llm_calling:    # =3 thì tạo lại
                    del self.client
                    time.sleep(60 * 3 + time_sleep_reset_seconds)
                    self.client = self.get_client(self.model_name)
                    if self.logger:
                        self.logger.info(f"[ChatLLM] fail too many times: {fail_llm_calling} times, change LLM client")

        del messages
        return all_response
    def get_model_response_txt(self, messages):
        # self.messages.append({"role": "user", "content": prompt})
        main_content = ""
        self.reasoning = ""
        fail_llm_calling = 0
        messages = messages.copy()  # tránh thay đổi messages gốc
        success_llm_calling = False
        while fail_llm_calling < 6 and not success_llm_calling:
            try:
                # import pdb; pdb.set_trace()
                if "mistral-" in self.model_name.lower():
                    response = self.client.chat.complete(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        # extra_body={"chat_template_kwargs": {"thinking": False}},
                        # extra_body={"reasoning": {"enabled": False}},
                        # reasoning_effort=None,
                    )
                choices = response.choices
                if choices:
                    main_content = choices[0].message.content
                    self.reasoning = getattr(choices[0].message, 'reasoning', "")  # reasoning gptoss120b - nvidia-NIM ;
                    if self.reasoning == "":
                        self.reasoning = getattr(choices[0].message, 'reasoning_content', "")  # deepseekr1 - nvidia-NIM
                    if self.reasoning == "":
                        self.reasoning = getattr(choices[0].message, 'reasoning_details', [{'text': ''}])[0]['text'] # gptoss20b - openrouter
                    if main_content:    # avoid null/None response
                        success_llm_calling = True
                else:
                    if self.logger:
                        self.logger.info("[ChatLLM]  get_model_response_txt : Empty choices in response")
                    main_content = ""

            except Exception as err:
                success_llm_calling = False
                if self.logger:
                    self.logger.info(f"[ChatLLM]  get_model_response_txt Exception : {err}")
                else:
                    print(f"[ChatLLM]  get_model_response_txt Exception : {err}")
                error_message = str(err)
                main_content = error_message
                success_llm_calling += 1
                prompt_truncate = self.truncate_message_history(error_message, messages)
                if len(prompt_truncate) > 0:
                    if self.logger:
                        self.logger.info(f"[ChatLLM]  truncate from len {len(messages[0]['content'])} to len {len(prompt_truncate)}")
                    messages[0]['content'] = prompt_truncate
                try:
                    time_sleep_reset_seconds = getattr(err, "body", {'reset_seconds': 10}).get("reset_seconds", 10)
                except AttributeError as eeeeee:
                    time_sleep_reset_seconds = 10
                time.sleep(60+time_sleep_reset_seconds)
                if fail_llm_calling > 3:
                    del self.client
                    self.client = self.get_client(self.model_name)
            # return err
        # self.messages.append({"role": "assistant", "content": main_content})
        del messages
        return main_content
    def truncate_message_history(self, error_message: str, messages: list[Dict] ) -> str:
        """

        :param error_message: error message from LLM API, in try-except block
        :param messages: history messages
        :return:
        """
        if "token" not in error_message:
            return ""
        if self.logger:
            self.logger.info(f"[ChatLLM] error message: {error_message}")
        max_context_match = re.search(r'maximum context length is (\d+) tokens', error_message)
        max_context_length = int(max_context_match.group(1)) if max_context_match else None
        if not max_context_length:
            # Try alternative pattern for deepseek
            max_context_match = re.search(r'maximum sequence length of (\d+)', error_message)
            max_context_length = int(max_context_match.group(1)) if max_context_match else None

        # Parse requested tokens
        requested_match = re.search(r'request(.*?) (\d+) (.*?)tokens', error_message)
        # requested_tokens = int(requested_match.group(1)) if requested_match else None
        requested_tokens = int(requested_match.group(
            2)) if requested_match else None
        if not requested_tokens:
            # Try alternative pattern for deepseek
            requested_match = re.search(r'prompt contains (\d+) tokens', error_message)
            requested_tokens = int(requested_match.group(1)) if requested_match else None

        # Parse tokens in messages
        messages_match = re.search(r'(\d+) in the messages', error_message)
        tokens_in_messages = int(messages_match.group(1)) if messages_match else None

        if not max_context_length or not requested_tokens:  # For mistral error :
            int_token_match = [int(re_result) for re_result in re.findall(r' (\d+) ', error_message)]
            if len(int_token_match) < 2:
                int_token_match = [int(re_result) for re_result in re.findall(r'\b\d+\b', error_message)]
            int_token_match.sort()
            if len(int_token_match) >= 2:
                requested_tokens = int_token_match[-1]
                max_context_length = int_token_match[-2]
            if max_context_length < 32000:
                max_context_length = 100000
        if not tokens_in_messages:
            tokens_in_messages = requested_tokens
        # import pdb; pdb.set_trace()
        if self.logger:
            self.logger.info(f"[ChatLLM] truncate_message_history max_context_length: {max_context_length}, requested_tokens: {requested_tokens}, tokens_in_messages: {tokens_in_messages}")
        if "Assistant message must have either cont" in error_message:
            import pdb; pdb.set_trace()
        if not max_context_length or not requested_tokens or not tokens_in_messages:
            if self.logger:
                self.logger.info(f"Error during LLM API call: {error_message}. Retrying...")
            return ""
        """
        Reduce prompt size to fit within context length
        """
        prompt_ori = messages[0]['content']
        # max_str_prompt = len(prompt_ori) * (max_context_length - self.max_tokens) / tokens_in_messages
        max_str_prompt = len(prompt_ori) * (max_context_length - 1024) / tokens_in_messages
        max_str_prompt = int(max_str_prompt * 0.7)
        # prompt = prompt[:max_str_prompt]
        # messages[0]['content'] = prompt_ori[-max_str_prompt:]
        prompt_truncate = prompt_ori[-max_str_prompt:]
        # import pdb; pdb.set_trace()

        return prompt_truncate
    def get_api_key(self, model_name):
        if "deepseek-ai/deepseek-r1" == model_name or "qwen/qwen2.5-coder-32b-instruct" == model_name \
                or "meta/llama-3.3-70b-instruct" == model_name or\
                model_name == "deepseek-ai/deepseek-v3.1" or \
                model_name == "deepseek-ai/deepseek-v3.2" or \
                model_name == "moonshotai/kimi-k2-thinking" or \
                model_name == "qwen/qwen3-235b-a22b" or \
                model_name == "qwen/qwen2.5-coder-7b-instruct" or \
                model_name == "qwen/qwen3-coder-480b-a35b-instruct" or \
                model_name == "meta/llama-3.1-405b-instruct" or \
                model_name == "openai/gpt-oss-120b" :
            api_key = os.environ["NVIDIA_API_KEY"]
        elif ("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" ==  model_name
              or model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
            or model_name == "lgai/exaone-3-5-32b-instruct"
        ):
            api_key = os.environ["TOGETHER_API_KEY"]
        elif "mistral-" in model_name.lower():
            api_key = os.environ["MISTRAL_API_KEY"]
        # elif "gemini" in model_name.lower():
        #     api_key = os.environ["GOOGLE_API_KEY"]
        elif 'gpt' in model_name and 'gpt' == model_name[:3]:
            api_key = os.getenv('OPENAI_API_KEY')
        else:
            # raise NotImplementedError
            api_key = os.environ["OPENROUTER_API_KEY"]
        return api_key
    def get_client(self, model_name):
        if "deepseek-ai/deepseek-r1" == model_name or "qwen/qwen2.5-coder-32b-instruct" == model_name \
                or "meta/llama-3.3-70b-instruct" == model_name or\
                model_name == "deepseek-ai/deepseek-v3.1"  or \
                model_name == "deepseek-ai/deepseek-v3.2" or \
                model_name == "moonshotai/kimi-k2-thinking" or \
                model_name == "qwen/qwen3-235b-a22b"  or \
                model_name == "qwen/qwen2.5-coder-7b-instruct"  or \
                model_name == "qwen/qwen3-coder-480b-a35b-instruct"  or \
                model_name == "meta/llama-3.1-405b-instruct"  or \
                model_name == "openai/gpt-oss-120b" :
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.api_key
            )
        elif ("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" ==  model_name
              or model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
            or model_name == "lgai/exaone-3-5-32b-instruct"
        ):
            client = OpenAI(
                  api_key=self.api_key,
                  base_url="https://api.together.xyz/v1",
                )
        elif "mistral-" in model_name.lower():
            from mistralai import Mistral
            client = Mistral(api_key=self.api_key)
        # elif "gemini" in model_name.lower():
        #     client = OpenAI(
        #         base_url="https://generativelanguage.googleapis.com/v1beta/",
        #         api_key=self.api_key,
        #     )
        elif 'gpt' in model_name and 'gpt' == model_name[:3]:
            client = OpenAI(api_key=self.api_key)
        else:
            # raise NotImplementedError
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
        return client
    def get_max_output_tokens_num(self, model_name):
        """
        Max tokens in response from model

        :param model_name:
        :return:
        """
        if 'gpt-3.5-turbo' in model_name:
            max_tokens = 4096
        elif 'gpt-4' in model_name:
            max_tokens = 8192
        elif "qwen/qwen2.5-coder-32b-instruct" == model_name:
            max_tokens = 2000
            # max_str_prompt = 28000 * 4
        elif "deepseek" in model_name:
            max_tokens = 20000
        else:
            max_tokens = 20000
        return max_tokens


