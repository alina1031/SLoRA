import os 
import json 

import argparse 
from tqdm import tqdm
import re 

_ANSWER_PREFIX_RE = re.compile(
    r'(?:^|\b)(?:the\s+answer\s+is|final\s*answer\s*:?)\s*',
    flags=re.IGNORECASE
)


def extract_answer_from_text(text):
    if not isinstance(text, str):
        text = str(text)

    last = None
    for m in _ANSWER_PREFIX_RE.finditer(text):
        last = m
    answer_part = text[last.end():].strip() if last else text.strip()

    answer_part = re.sub(r'^(?:the\s+answer\s+is\s*)+', '', answer_part, flags=re.IGNORECASE).strip()

    answer_part = answer_part.strip().strip('"\'')

    answer_part = re.sub(r'[\s\.\,\;\:!\?]+$', '', answer_part)

    m = re.fullmatch(r'\{(\d+)\}', answer_part)
    if m:
        return int(m.group(1))

    m = re.match(r'^\[(\d+)\]\s*(.*)$', answer_part)
    if m:
        idx = int(m.group(1))
        txt = m.group(2).strip()
        txt = re.sub(r'[\.\,\;\:!\?]+$', '', txt).strip()
        return (idx, txt if txt else None)

    m = re.match(r'^(\d+)\D*$', answer_part)
    if m:
        return int(m.group(1))

    return answer_part

def normalize_answer(answer):
    if isinstance(answer, str):
        return answer.replace("and", "&").replace("or", "&").strip().lower()
    return answer


def continual_acc(line):
    pred = line['text']
    target = line['additional_info']['answer']
    target_idx = line['additional_info']['answer_idx']

    pred_answer = extract_answer_from_text(pred)
    line['additional_info']['pred_answer'] = pred_answer

    pred_answer_normalized = normalize_answer(pred_answer)
    target_normalized = normalize_answer(target)

    if isinstance(pred_answer, tuple):
        pred_idx, pred_text = pred_answer
        pred_text_normalized = normalize_answer(pred_text)
        return 1 if pred_idx == target_idx or pred_text_normalized == target_normalized else 0
    elif isinstance(pred_answer, int):
        return 1 if pred_answer == target_idx else 0
    elif isinstance(pred_answer, str):
        return 1 if pred_answer_normalized == target_normalized else 0
    else:
        return 0


METRIC_FUNC_MAPPING = {
    "dbpedia": continual_acc,
    "amazon": continual_acc,
    "yahoo": continual_acc,
    "agnews": continual_acc,
    "yelp": continual_acc,
    "mnli": continual_acc,
    "CB": continual_acc,
    "WiC": continual_acc,
    "MultiRC": continual_acc,
    "COPA": continual_acc,
    "QQP": continual_acc,
    "BoolQA": continual_acc,
    "RTE": continual_acc,
    "IMDB": continual_acc,
    "SST-2": continual_acc
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=False)
    args = parser.parse_args()

    # input_file is a jsonl file with the following format:
    # questions = client.read_jsonl(args.input_file)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.input_file), "r")]
    
    total_num = len(questions)
    total_score = 0


    dataset_name  = args.input_file.split("/")[-2].replace(".jsonl", "")
    if dataset_name not in METRIC_FUNC_MAPPING:
        dataset_name  = args.input_file.split("/")[-1].replace(".jsonl", "")
    acc_func = METRIC_FUNC_MAPPING[dataset_name]
    wrong_idx = []
    for line in tqdm(questions, total=total_num):
        scores = acc_func(line)
        if scores is None:
            total_num -= 1
            wrong_idx.append(line)
            continue
        total_score += scores
        if scores == 0:
            wrong_idx.append(line)
    avg_acc = total_score / total_num
    print(f"Acc in {dataset_name}: {avg_acc}")
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(wrong_idx, f, ensure_ascii=False)
    
