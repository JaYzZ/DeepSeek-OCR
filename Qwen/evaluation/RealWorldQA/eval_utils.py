"""
RealWorldQA Evaluation Utilities

Evaluation utilities, fully independent of VLMEvalKit.
"""

import os
import requests
import time
import random
import string
import copy
import traceback
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Any
try:
    from .common_utils import encode_image_to_base64
except ImportError:
    from common_utils import encode_image_to_base64


def strip_thinking_tokens(prediction):
    """Strip thinking process wrapped in <thinking>...</thinking> tokens."""
    import re
    prediction_str = str(prediction)
    # Remove <thinking>...</thinking>
    prediction_str = re.sub(r'<thinking>.*?</thinking>', '', prediction_str, flags=re.DOTALL)
    return prediction_str.strip()


class OpenAIWrapper:
    """Wrapper for OpenAI API."""
    
    def __init__(self, model, api_base, api_key, timeout=60, retry=5, wait=5):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = 'Failed to obtain answer via API.'
    
    def generate(self, messages):
        """Generate a response from the API."""
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        
        # Format messages for API
        formatted_messages = []
        for msg in messages:
            if msg['type'] == 'text':
                formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg['value']}]})
            elif msg['type'] == 'image':
                # Load and encode the image
                image = Image.open(msg['value'])
                image_data = encode_image_to_base64(image)
                formatted_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                })
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": 4096,
            "temperature": 0
        }
        
        for i in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    return resp_json['choices'][0]['message']['content'].strip()
                
                time.sleep(self.wait)
            except Exception as e:
                print(f"API error: {e}")
                time.sleep(self.wait)
        
        return self.fail_msg

class DashScopeWrapper:
    """Wrapper for DashScope API."""
    
    def __init__(self, model, api_base, api_key, timeout=60, retry=5, wait=5):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = 'Failed to obtain answer via API.'
    
    def generate(self, messages):
        """Generate a response from the API."""
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        
        # Format messages for API
        formatted_messages = []
        for msg in messages:
            if msg['type'] == 'text':
                formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg['value']}]})
            elif msg['type'] == 'image':
                # Load and encode the image
                image = Image.open(msg['value'])
                image_data = encode_image_to_base64(image)
                formatted_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                })
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_completion_tokens": 4096,
            "n": 1,
            "temperature": 0,
            "stream": False
        }

        for i in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    
                    # Check finish reason
                    for output in resp_json['choices']:
                        if output['finish_reason'] not in ['stop', 'function_call']:
                            print(f"DashScope finished with error: {resp_json}")
                            time.sleep(self.wait)
                            continue
                    
                    return resp_json['choices'][0]['message']['content']
                else:
                    print(f"DashScope API error: HTTP {response.status_code}")
                    try:
                        error_content = response.json()
                        print(f"Error details: {error_content}")
                    except:
                        print(f"Raw error content: {response.content.decode('utf-8', errors='replace')}")
                
                time.sleep(self.wait)
            except requests.exceptions.ConnectionError as conn_err:
                print(f"DashScope: Connection error occurred: {conn_err}")
                time.sleep(self.wait)
            except requests.exceptions.Timeout as timeout_err:
                print(f"DashScope: Timeout error occurred: {timeout_err}")
                time.sleep(self.wait)
            except requests.exceptions.RequestException as req_err:
                print(f"DashScope: Request exception occurred: {req_err}")
                time.sleep(self.wait)
            except Exception as e:
                print(f"DashScope: An error occurred: {e}")
                print(traceback.format_exc())
                time.sleep(self.wait)
        
        return self.fail_msg

class CustomJudgeWrapper:
    """Wrapper for custom judge server with question/reference/prediction format."""

    def __init__(self, judge_server_url, timeout=60, retry=5, wait=5):
        self.judge_server_url = judge_server_url
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = 'Z'
        self.model = 'CustomJudge'

    def judge(self, question, reference, prediction):
        """Send request to custom judge server (text only, no images)."""
        # Truncate prediction if too long to avoid token limit issues
        max_chars = 50000
        prediction_str = str(prediction)
        if len(prediction_str) > max_chars:
            prediction_str = prediction_str[-max_chars:]  # Keep the END where final answer is

        payload = {
            "question": str(question),
            "reference": str(reference),
            "prediction": prediction_str
        }

        headers = {'Content-Type': 'application/json'}

        for i in range(self.retry):
            try:
                response = requests.post(
                    f"{self.judge_server_url}/judge",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    resp_json = response.json()
                    if resp_json.get('success', False):
                        correct = resp_json.get('correct', False)
                        verdict = resp_json.get('verdict', 'UNKNOWN')

                        if correct:
                            # For multiple choice, extract the option letter
                            import re
                            match = re.search(r'\b([A-D])\b', str(prediction).upper())
                            if match:
                                return match.group(1)
                            return prediction
                        else:
                            return self.fail_msg
                    else:
                        print(f"Judge server error: {resp_json.get('error', 'Unknown error')}")
                        time.sleep(self.wait)
                else:
                    print(f"Judge server HTTP error: {response.status_code}")
                    time.sleep(self.wait)
            except Exception as e:
                print(f"Judge server error: {e}")
                time.sleep(self.wait)

        return self.fail_msg

    def generate(self, messages):
        """Compatibility method - not used for custom judge."""
        return "Custom judge should use judge() method instead"

def build_judge(model, api_type, api_url=None, api_key=None):
    """Build a judge model for evaluation."""
    if api_type == 'custom':
        # Use custom judge server
        judge_url = api_url or os.environ.get('JUDGE_SERVER_URL', 'http://47.111.147.142:8600')
        print(f"Using custom judge server: {judge_url}")
        return CustomJudgeWrapper(judge_url)
    elif api_type == 'local':
        # Use local OpenAI-compatible API
        api_base = api_url or os.environ.get('LOCAL_API_URL', 'http://localhost:8016/v1/chat/completions')
        api_key = api_key or os.environ.get('LOCAL_API_KEY', 'EMPTY')
        print(f"Using local API: {api_base}")
        return OpenAIWrapper(model, api_base, api_key)
    elif api_type == 'mit':
        api_key = os.environ.get('MIT_SPIDER_TOKEN', '')
        api_base = os.environ.get('MIT_SPIDER_URL', '')
        return OpenAIWrapper(model, api_base, api_key)
    elif api_type == 'dash':
        api_key = os.environ.get('CHATGPT_DASHSCOPE_API_KEY', '')
        api_base = os.environ.get('DASHSCOPE_API_BASE', '')
        return DashScopeWrapper(model, api_base, api_key)
    else:
        raise ValueError

def can_infer_option(answer, choices):
    """Rule-based extraction of answer option."""
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False

def can_infer_text(answer, choices):
    """Extract answer by matching text content."""
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer(answer, choices):
    """Combined approach to infer answer choice."""
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)

def build_choices(item):
    """Build choices dictionary from item."""
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret

def build_option_str(option_dict):
    """Build option string for prompt."""
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s

def build_prompt(question, options, prediction):
    """Build prompt for answer extraction."""
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)

def extract_final_answer_smart(prediction):
    """Extract the final answer letter from prediction, stripping all thinking content."""
    import re

    prediction_str = str(prediction)

    # Pattern 1: Single letter (most common for RealWorldQA)
    if len(prediction_str.strip()) <= 3 and prediction_str.strip().isalpha() and prediction_str.strip().isupper():
        return prediction_str.strip()

    # Pattern 2: \boxed{D}
    match = re.search(r'\\boxed\{([A-Z])\}', prediction_str)
    if match:
        return match.group(1)

    # Pattern 3: **A** or **A. $123**
    match = re.search(r'\*\*([A-Z])(?:\.|\s|\$)', prediction_str)
    if match:
        return match.group(1)

    # Pattern 4: "Thus, the correct answer is **D**"
    match = re.search(r'(?:correct answer|final answer|answer|thus).*?:\s*\*{1,2}([A-Z])(?:\.|\s|\$|\*{2})', prediction_str, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern 5: Last 200 chars analysis
    end_text = prediction_str[-200:]
    match = re.search(r'\b([A-Z])\b(?=\s*(?:\)|\.|,|\n|$|\*\*|\\boxed))', end_text)
    if match:
        return match.group(1)

    # Pattern 6: "The answer is D"
    match = re.search(r'(?:answer|correct|final).*?(?:is|:)\s*([A-Z])(?:\.|,|\s)', end_text, re.IGNORECASE)
    if match:
        return match.group(1)

    return None

def extract_answer_from_item(model, item, wait=5):
    """Extract answer - ONLY validate final answer, no thinking content."""
    import re
    prediction = strip_thinking_tokens(item['prediction'])  # Strip thinking tokens
    gt_answer = item['answer']

    # Step 1: Smart extraction (works for all predictions)
    extracted_answer = extract_final_answer_smart(prediction)

    if extracted_answer:
        # Direct comparison
        is_correct = (extracted_answer == gt_answer.strip().upper())
        log = f"Smart extract: '{extracted_answer}' vs GT '{gt_answer}' = {is_correct}"
        return dict(opt=extracted_answer, log=log, extract_model='smart_extraction', extract_flag=True)

    # Step 2: Traditional rule-based as fallback
    choices = build_choices(item)
    ret = can_infer(prediction, choices)

    if ret:
        is_correct = (ret == gt_answer.strip().upper())
        log = f"Rule extract: '{ret}' vs GT '{gt_answer}' = {is_correct}"
        return dict(opt=ret, log=log, extract_model='rule', extract_flag=is_correct)

    # Step 3: Use judge if available (rarely needed)
    if model and hasattr(model, 'judge'):
        print(f"Using judge (prediction length: {len(str(prediction))})")
        result = model.judge(
            question=item['question'],
            reference=gt_answer,
            prediction=prediction
        )

        if result and isinstance(result, dict):
            correct = result.get('correct', False)
            match = re.search(r'\b([A-Z])\b', str(prediction).upper())
            extracted = match.group(1) if match else 'Z'
            log = f"Judge: {result.get('verdict')}, extracted: {extracted}"
            return dict(opt=extracted, log=log, extract_model='judge', extract_flag=correct)

    # Step 4: Fallback
    options = list(string.ascii_uppercase)[:4]
    log = "Extraction failed, random choice"
    return dict(opt=random.choice(options), log=log, extract_model='random', extract_flag=False)

def eval_single_sample(args):
    """Evaluate a single sample."""
    model, item = args
        
    # Extract answer using the combined approach
    result = extract_answer_from_item(model, item)
    
    # Get ground truth answer
    gt_answer = item['answer']
    
    # Determine if the answer is correct
    hit = 1 if result['opt'] == gt_answer else 0
    
    return {
        "index": item['index'],
        "question": item['question'],
        "prediction": item['prediction'],
        "extracted_answer": result['opt'],
        "extraction_method": result['extract_model'],
        "extraction_success": result['extract_flag'],
        "extraction_log": result['log'],
        "gt": gt_answer,
        "hit": hit
    }
