import os
import re
import numpy as np
import pandas as pd
import copy
import os
import glob
import json

RED = '\033[91m'
RESET = '\033[0m'

os.environ["CMBAGENT_DEBUG"] = "false"
os.environ["ASTROPILOT_DISABLE_DISPLAY"] = "false"

import cmbagent
path_to_targets = '/Users/milind24/cmbagentmain/cmbagent'

from datasets import load_dataset
import traceback

# 1. Load top 10 samples from DS-1000
ds = load_dataset("xlangai/DS-1000", split="test")
#samples = ds.shuffle(seed=42).select(range(1))  # Top 10 random samples
#samples = ds.select(range(1))
#samples = ds.select([35])
samples = ds.select(list(range(300, 305)))

def extract_from_tags(text: str, start_tag: str, end_tag: str) -> str:
    start_index = len(start_tag) if text.startswith(start_tag) else 0
    end_index = text.find(end_tag, len(start_tag))
    end_index = end_index if end_index != -1 else len(text)
    return text[start_index:end_index]

def postprocess(code: str) -> str:
    
    code = extract_from_tags(code, "```python\n", "\n```")
    code = extract_from_tags(code, "```\n", "\n```")  # new pattern
    code = extract_from_tags(code, "<code>", "</code>")
    code = extract_from_tags(code, "", "</code>")
    code = extract_from_tags(code, "", "\nEND SOLUTION")
    code = extract_from_tags(code, "", "\n### END SOLUTION")
    return code.strip()

def extract_between_closing_and_opening(text: str, closing_tag: str, opening_tag: str) -> str:
    start_index = text.find(closing_tag)
    if start_index == -1:
        return ""
    start_index += len(closing_tag)
    end_index = text.find(opening_tag, start_index)
    if end_index == -1:
        return text[start_index:].strip()
    return text[start_index:end_index].strip()

import re

def get_result(cmbagent_results):
    chat_history = cmbagent_results['chat_history']
    try:
        for obj in reversed(chat_history):
            if obj['name'] == 'researcher_response_formatter':
                result = obj['content']
                break
        else:
            return None  # No matching object found
        task_result = result
    except Exception as e:
        print(f"Error: {e}")
        return None

    # Extract content between <code>...</code> tags
    match = re.search(r"<code>(.*?)</code>", task_result, re.DOTALL)
    if match:
        code_str = match.group(1).strip()
        return code_str
    else:
        print("No <code>...</code> block found.")
        return None

def my_agent(task, metadata):

    results = cmbagent.one_shot(task,
                                max_rounds=10,
                                agent='researcher',
                                #agent='engineer',
                                #initial_agent=metadata['initial_agent'],
                                researcher_model='gpt-4.1',
                                #engineer_model='claude-opus-4-20250514',
                                work_dir="/Users/milind24/cmbagentmain/cmbagent/output/data/"
                                )


    return get_result(results)

import traceback
from multiprocessing import Pool
from tqdm import tqdm

# You must define or import these somewhere in your code:
# samples = [...]
# def my_agent(prompt_text, metadata): ...
# def test_execution(code): ...
# def test_string(code): ...  # Optional

def process_sample(i_sample):
    i, sample = i_sample
    print(f"### Sample {i+1}: {sample['metadata']['problem_id']} ###")

    prompt = (
        "Write a short code following the given format and indentation. "
        "Place the executable code between <code> and </code> tags, without any other non-executable things \n"
        "Also save the code you place between <code> and </code> tags in a result.txt file \n"
        "Only provide the code completion needed. Don't repeat the context code or add any unnecessary lines or comments.. \n"
        f"Prompt:\n {sample['prompt']}\n"
    )

    try:
        raw_output = my_agent(prompt, sample["metadata"])
        print(raw_output)  # Check structure

        solution_code = raw_output
        # solution_code = extract_between_closing_and_opening(raw_output, "<code>", "</code>")

        print("\nAgent Solution:\n", solution_code)

        code_context = sample["code_context"]
        full_code = (
            f"{code_context}\n"
            + f"solution = '''{solution_code}'''\n"
            + "test_execution(solution)\n"
        )
        if "test_string(" in code_context:
            full_code += "test_string(solution)\n"

        print("\n--- Running Test ---")
        exec_locals = {}
        exec(full_code, exec_locals, exec_locals)
        print("✅ Passed")
        return ("C", sample["metadata"]["problem_id"])

    except Exception:
        print("❌ Failed")
        traceback.print_exc()
        return ("I", sample["metadata"]["problem_id"])


def main():

    results = []
    max_workers = 5
    with Pool(max_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_sample, enumerate(samples)), total=len(samples)):
            results.append(result)

    print(f"\n{'Problem Number':<15} {'Result':<10}")
    print("-" * 26)
    for status, number in results:
        if status == 'C':
            result_str = "Passed"
        else:
            result_str = f"{RED}Failed{RESET}"
        print(f"{number:<15} {result_str:<10}")


if __name__ == "__main__":
    main()