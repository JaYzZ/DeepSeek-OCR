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
        self.fail_msg = 'Z'  # Return Z on failure
        self.model = 'CustomJudge'

    def extract_final_answer(self, prediction):
        """Extract the final answer letter from prediction, stripping all thinking content."""
        import re

        # Pattern 1: \boxed{D} at the end
        match = re.search(r'\\boxed\{([A-Z])\}', str(prediction))
        if match:
            return match.group(1)

        # Pattern 2: **A** or **A. $123** (bold answer)
        match = re.search(r'\*\*([A-Z])(?:\.|\s|\$)', str(prediction))
        if match:
            return match.group(1)

        # Pattern 3: "Thus, the correct answer is **D**" or similar
        match = re.search(r'(?:correct answer|final answer|answer|thus).*?:\s*\*{1,2}([A-Z])(?:\.|\s|\$|\*{2})', str(prediction), re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 4: Just the letter at the very end (fallback)
        # Look at last 500 chars for the final answer
        end_text = str(prediction)[-500:]
        match = re.search(r'\b([A-Z])\b(?=\s*(?:\)|\.|,|\n|$|\*\*|\\boxed))', end_text)
        if match:
            return match.group(1)

        # Pattern 5: Find any single letter option in the last part
        # Look for "The answer is D" patterns
        match = re.search(r'(?:answer|correct|final).*?(?:is|:)\s*([A-Z])(?:\.|,|\s)', str(prediction[-500:]), re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def judge(self, question, reference, prediction):
        """Send request to custom judge server with stripped thinking content."""
        import re

        # Step 1: Try to extract the final answer locally (fast, no API call needed)
        extracted_answer = self.extract_final_answer(prediction)

        if extracted_answer:
            # Compare extracted answer with reference
            if extracted_answer == str(reference).strip().upper():
                # Direct match found - correct!
                return {'success': True, 'verdict': 'CORRECT', 'correct': True, 'extracted': extracted_answer}
            else:
                # Extracted but doesn't match - incorrect
                return {'success': True, 'verdict': 'INCORRECT', 'correct': False, 'extracted': extracted_answer}

        # Step 2: If extraction fails, use judge (but with minimal content)
        # Only send the last 2000 chars where the conclusion is
        prediction_minimal = str(prediction)[-2000:] if len(str(prediction)) > 2000 else str(prediction)

        payload = {
            "question": str(question),
            "reference": str(reference),
            "prediction": prediction_minimal
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
                        verdict = resp_json.get('verdict', 'UNKNOWN')
                        correct = resp_json.get('correct', False)

                        # Map verdict to answer option
                        if correct and verdict == 'CORRECT':
                            # For multiple choice, the prediction should contain the option
                            # Extract option letter if present
                            match = re.search(r'\b([A-D])\b', str(prediction_minimal).upper())
                            if match:
                                return match.group(1)
                            # If no option letter found, return the prediction as-is
                            return prediction_minimal
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
        # This won't be called for custom judge, but return something sensible
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
        api_base = api_url or os.environ.get('LOCAL_API_URL', 'http://localhost:8000/v1/chat/completions')
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
        raise ValueError(f"Unsupported API type: {api_type}")

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
                # print(f'A might be a quantifier in the string: {answer}.')
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
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret

def build_option_str(option_dict):
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s

def build_prompt(question, options, prediction):
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

    # Step 1: Remove thinking content wrapped in special tags
    # Remove </think> or <|thinking|> tags with content
    prediction_str = re.sub(r'<\|thinking\|>.*?<\|/thinking\|>', '', prediction_str, flags=re.DOTALL)
    # Remove <|thinking|>...<|/thinking|> tags
    prediction_str = re.sub(r'<\|thinking\|>.*?<\|/thinking\|>', '', prediction_str, flags=re.DOTALL)
    # Remove any other XML-like thinking tags
    prediction_str = re.sub(r'<\|[^|]+\|>.*?<\|/\|[^|]+\|>', '', prediction_str, flags=re.DOTALL)

    # Remove common thinking/chain-of-thought markers
    lines = prediction_str.split('\n')
    content_lines = [l for l in lines if l.strip() and
                     not l.strip().startswith(('Let me think', 'I need to', 'To solve', 'First, ', 'Step ', 'Thinking:'))]
    final_content = '\n'.join(content_lines) if content_lines else prediction_str

    # Now extract from cleaned content
    # Pattern 1: \boxed{D} - LaTeX boxed answer (most reliable)
    match = re.search(r'\\boxed\{([A-Z])\}', final_content)
    if match:
        return match.group(1)

    # Pattern 2: **A** - Bold answer
    match = re.search(r'\*\*([A-Z])(?:\.|\s|\$|\n)', final_content)
    if match:
        return match.group(1)

    # Pattern 3: "Thus, the correct answer is **D**" or similar
    match = re.search(r'(?:correct answer|final answer|answer|thus).*?:\s*\*{1,2}([A-Z])(?:\.|\s|\$|\*{2})', final_content, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern 4: Last 200 chars - very end where final answer lives
    end_text = final_content[-200:]
    match = re.search(r'\b([A-Z])\b(?=\s*(?:\)|\.|,|\n|$|\*\*|\\boxed))', end_text)
    if match:
        return match.group(1)

    # Pattern 5: "The answer is D" patterns
    match = re.search(r'(?:answer|correct|final).*?(?:is|:)\s*([A-Z])(?:\.|,|\s)', end_text, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern 6: Single letter in last line (ultimate fallback)
    last_line = final_content.strip().split('\n')[-1] if final_content.strip() else final_content
    match = re.match(r'^\s*([A-Z])\s*$', last_line)
    if match:
        return match.group(1)

    return None

def extract_answer_from_item(model, item, wait=5):
    """Extract answer from model prediction - ONLY validate the final answer, not thinking."""
    import re
    prediction = strip_thinking_tokens(item['prediction'])  # Strip thinking tokens
    gt_answer = item['GT']

    # Step 1: Try smart extraction (works for all predictions, short or long)
    extracted_answer = extract_final_answer_smart(prediction)

    if extracted_answer:
        # Direct comparison with ground truth
        is_correct = (extracted_answer == gt_answer.strip().upper())

        log = f"Smart extraction: found '{extracted_answer}', GT: '{gt_answer}', correct: {is_correct}"
        return dict(
            opt=extracted_answer,
            log=log,
            extract_model='smart_extraction',
            extract_flag=True
        )

    # Step 2: If smart extraction fails, try traditional rule-based
    choices = build_choices(item)
    ret = can_infer(prediction, choices)

    if ret:
        if ret == 'Z':
            log = f"Rule extract failed with rule result: {ret}"
            extract_flag = False
        else:
            is_correct = (ret == gt_answer.strip().upper())
            log = f"Rule extraction: {ret}, GT: {gt_answer}, correct: {is_correct}"
            extract_flag = is_correct
        return dict(opt=ret, log=log, extract_model='rule', extract_flag=extract_flag)

    # Step 3: If both fail, use judge (only if available)
    if model and hasattr(model, 'judge'):
        print(f"Extraction failed, using judge (prediction length: {len(str(prediction))})")
        result = model.judge(
            question=item['question'],
            reference=gt_answer,
            prediction=prediction
        )

        if result and isinstance(result, dict):
            correct = result.get('correct', False)
            verdict = result.get('verdict', 'UNKNOWN')

            # Extract answer for return
            match = re.search(r'\b([A-Z])\b', str(prediction).upper())
            extracted_opt = match.group(1) if match else 'Z'

            log = f"Judge: {verdict}, extracted: {extracted_opt}, correct: {correct}"
            return dict(opt=extracted_opt, log=log, extract_model='judge', extract_flag=correct)

    # Step 4: Ultimate fallback
    options = list(string.ascii_uppercase)[:5]  # A, B, C, D, E
    log = "All extraction methods failed, returning random"
    return dict(opt=random.choice(options), log=log, extract_model='random', extract_flag=False)

def eval_single_sample(args):
    """Evaluate a single sample."""
    model, item = args
        
    # Extract answer using the combined approach
    result = extract_answer_from_item(model, item)
    
    # Determine if the answer is correct
    hit = 1 if result['opt'] == item['GT'] else 0
    
    return {
        "index": item['index'],
        "split": item['split'],
        "question": item['question'],
        "prediction": item['prediction'],
        "extracted_answer": result['opt'],
        "extraction_method": result['extract_model'],
        "extraction_success": result['extract_flag'],
        "extraction_log": result['log'],
        "gt": item['GT'],
        "hit": hit
    }