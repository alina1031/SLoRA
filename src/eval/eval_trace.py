from src.eval.metrics import calculate_sari, calculate_bleu, calculate_rouge, calculate_accuracy, calculate_f1, calculate_fuzz
import re
import argparse
import json
import os
from tqdm import tqdm


def eval_20Minuten(input_sequences, predicted_sequences, ground_truths):
    predicted_sequences = _strip_thought_blocks(predicted_sequences)
    if '\nAnswer:' in predicted_sequences:
        predicted_sequences = predicted_sequences.split('\nAnswer:')[1]
    bleu_1 = calculate_bleu(predicted_sequences, ground_truths, 1)
    bleu_4 = calculate_bleu(predicted_sequences, ground_truths, 4)
    rouge = calculate_rouge(predicted_sequences, ground_truths)
    sari = calculate_sari(input_sequences, predicted_sequences, ground_truths)
    return {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "sari": sari}

def eval_CStance(predicted_sequences, ground_truths):
    predicted_sequences = _strip_thought_blocks(predicted_sequences)
    if '\nAnswer:' in predicted_sequences:
        predicted_sequences = predicted_sequences.split('\nAnswer:')[1]
    if '\n回答:' in predicted_sequences:
        predicted_sequences = predicted_sequences.split('\n回答:')[1]
    if not predicted_sequences or not ground_truths:
        return {"accuracy": 0.0}
    accuracy = calculate_accuracy(predicted_sequences[0], ground_truths)
    return {"accuracy": accuracy}


def extract_option(paragraph):
    match = re.search(r'choose option\s+([A-Za-z])', paragraph, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def _strip_thought_blocks(s: str) -> str:
    pattern = r'(?is)<think\b[^>]*>.*?</think\s*>|<think\b[^>]*>\s*|</think\s*>'
    return re.sub(pattern, '', s)

def _to_text(predicted_sequences) -> str:
    ps = predicted_sequences
    if isinstance(ps, list) and ps:
        first = ps[0]
        if isinstance(first, dict) and 'text' in first:
            ps = first['text']
        else:
            ps = first
    elif isinstance(ps, dict) and 'text' in ps:
        ps = ps['text']
    if not isinstance(ps, str):
        ps = str(ps)
    return ps

def extract_policy_stance(text: str):
    t = _strip_thought_blocks(text)

    if '\nAnswer:' in t:
        t = t.split('\nAnswer:')[-1]

    t = re.sub(r'[ \t]+', ' ', t)

    if "dovish" in t.lower():
        return "A"
    elif "hawkish" in t.lower():
        return "B"
    elif "neutral" in t.lower():
        return "C"

    patterns = [
        r'\bstance\s*[:：]\s*([A-D])\b',        # Stance: B
        r'\banswer\s*(?:is|:)?\s*([A-D])\b',    # Answer is B / Answer: B
        r'\boption\s*([A-D])\b',                # option B
        r'\b选项\s*[:：]?\s*([A-D])\b',           # 选项：B
        r'\b答案\s*[:：]\s*([A-D])\b',            # 答案：B
        r'\b预测\s*[:：]\s*([A-D])\b',            # 预测：B
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()


    last_line = t.strip().splitlines()[-1] if t.strip().splitlines() else ''
    m = re.match(r'^\s*([A-D])\s*$', last_line)
    if m:
        return m.group(1).upper()

    m = re.search(r'(?<![A-Z])\b([A-D])\b(?![A-Z])', t)
    if m:
        return m.group(1).upper()

    return None

def eval_FOMC(predicted_sequences, ground_truths):
    text = _to_text(predicted_sequences)
    predicted_answer = extract_policy_stance(text)

    if predicted_answer is None:
        accuracy = 0.0
    else:
        accuracy = calculate_accuracy(predicted_answer, ground_truths)

    return {"accuracy": accuracy}

def extract_summary(text):
    patterns = [
        r"### Summary of Meeting Transcript[s]*\s*\n*\n*", 
        r"\*\*Summary of Meeting Transcripts:\*\*\s*\n*\n*", 
        r"Here is a summary of the meeting transcript[s]*:\s*\n*\n*",  
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    if 'Answer:' in text:
        parts = text.split('Answer:', 1)
        text = parts[1]
    
    text = re.sub(r"^####\s*", "", text)
    return text.strip()

def eval_MeetingBank(predicted_sequences, ground_truths):
    predicted_sequences = _strip_thought_blocks(predicted_sequences)
    predicted_sequences = extract_summary(predicted_sequences)
    if not predicted_sequences or not ground_truths:
        return {"bleu-1": 0.0, "bleu-4": 0.0, "rouge-L": 0.0}
    if not predicted_sequences.strip() or not ground_truths.strip():
        return {"bleu-1": 0.0, "bleu-4": 0.0, "rouge-L": 0.0}
    bleu_1 = calculate_bleu(predicted_sequences, ground_truths, 1)
    bleu_4 = calculate_bleu(predicted_sequences, ground_truths, 4)
    rouge = calculate_rouge(predicted_sequences, ground_truths)
    return {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge}



def is_solution_in_last_sentence(solution, text):

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    last_sentence = sentences[-1] if sentences else ""

    numbers_in_last_sentence = re.findall(r'\d+', last_sentence)
    
    return solution.strip() in numbers_in_last_sentence

def eval_NumGLUE(predicted_sequences, ground_truths):
    predicted_sequences = _strip_thought_blocks(predicted_sequences)
    if len(predicted_sequences) > 5:
        accuracy = is_solution_in_last_sentence(ground_truths, predicted_sequences)
        return {"accuracy": accuracy}
    predicted_sequences = predicted_sequences.strip()
    accuracy = calculate_accuracy(predicted_sequences, ground_truths)
    return {"accuracy": accuracy}

def eval_PapyrusF(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)
    f1 = calculate_f1(outputs, gts)
    return {"F1": f1}

def eval_Py150(predicted_sequences, ground_truths):
    predicted_sequences = _strip_thought_blocks(predicted_sequences)
    if  '\nReasoning:' in predicted_sequences:
        predicted_sequences = predicted_sequences.split('\nReasoning:')[0]
    outputs = [postprocess(predicted_sequences)]
    gts = [postprocess(ground_truths)]
    fuzz = calculate_fuzz(outputs, gts)
    return {"similarity": fuzz}


def extract_chosen_option(paragraph):
    patterns = [
        r'(?i)Answer:\s*([A-D])\b',
        r'^\s*([A-D])\s*$', 
        r'\b([A-D])[\.:,\)]',
        r'[Tt]he answer is\s+([A-D])\b', 
        r'[Ii] would choose\s+([A-D])\b', 
        r'[Mm]y answer is\s+([A-D])\b',    
    ]
    for pattern in patterns:
        match = re.search(pattern, paragraph, flags=re.MULTILINE)
        if match:
            return match.group(1)
    return None

def eval_ScienceQA(predicted_sequences, ground_truths):
    if predicted_sequences == "":
        return {"accuracy": 0.0}
    extracted_option = extract_chosen_option(predicted_sequences)
    
    if extracted_option:
        accuracy = calculate_accuracy(extracted_option, ground_truths[0])
        return {"accuracy": accuracy}

    if isinstance(predicted_sequences, str) and predicted_sequences.strip() in ['A', 'B', 'C', 'D']:
        accuracy = calculate_accuracy(predicted_sequences.strip(), ground_truths[0])
        return {"accuracy": accuracy}
    
    accuracy = calculate_accuracy(predicted_sequences[0], ground_truths[0])
    return {"accuracy": accuracy}


def resolve(dataset):
    answers, reasonings = [], []
    for datium in dataset:
        answers.append(datium[0])  # first char is the answer, e.g., A, B, ...
        reasonings.append(datium[2:])  # Reasoning starts from 2nd character
    return {"answers": answers, "reasonings": reasonings}

def postprocess(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code

# TRACE dataset evaluation
def eval_trace(dataset, input_sequences, predicted_sequences, ground_truths):
    if dataset == '20Minuten':
        return eval_20Minuten(input_sequences, predicted_sequences, ground_truths)
    elif dataset == 'C-STANCE':
        return eval_CStance(predicted_sequences, ground_truths)
    elif dataset == 'FOMC':
        return eval_FOMC(predicted_sequences, ground_truths)
    elif dataset == 'MeetingBank':
        return eval_MeetingBank(predicted_sequences, ground_truths)
    elif dataset == 'NumGLUE-cm' or dataset == 'NumGLUE-ds':
        return eval_NumGLUE(predicted_sequences, ground_truths)
    elif dataset == 'Py150':
        return eval_Py150(predicted_sequences, ground_truths)
    elif dataset == 'ScienceQA':
        return eval_ScienceQA(predicted_sequences, ground_truths)
    else:
        print('dataset:', dataset)
        print("The dataset is not in TRACE benchmark")
        return {}
    
    
def evaluate(questions, dataset_name):
    total_num = len(questions)
    total_score = 0
    rouge_score = [[], [], []]
    sari_score = []
    similarity_score = []

    wrong_idx = []
    for line in tqdm(questions, total=total_num):
        input_sequences = line["prompt"]
        predicted_sequences = line["text"]
        if '<think>' in predicted_sequences:
            predicted_sequences = re.sub(r'<think>.*?</think>', '', predicted_sequences, flags=re.DOTALL).strip()
        ground_truths = line["solution"]
        
        scores = eval_trace(dataset_name, input_sequences, predicted_sequences, ground_truths)
        
        if "accuracy" in scores:
            total_score += scores["accuracy"]
            if scores["accuracy"] == 0:
                wrong_idx.append(line)
        if "bleu-1" in scores:
            rouge_score[0].append(scores["bleu-1"])
        if "bleu-4" in scores:
            rouge_score[1].append(scores["bleu-4"])
        if "rouge-L" in scores:
            rouge_score[2].append(scores["rouge-L"])
        if "sari" in scores:
            sari_score.append(scores["sari"])
        if "F1" in scores:
            total_score += scores["F1"]
        if "similarity" in scores:
            similarity_score.append(scores["similarity"])

    result = {
        "rouge-1": sum(rouge_score[0]) / len(rouge_score[0]) if rouge_score[0] else 0.0,
        "rouge-4": sum(rouge_score[1]) / len(rouge_score[1]) if rouge_score[1] else 0.0,
        "rouge-l": sum(rouge_score[2]) / len(rouge_score[2]) if rouge_score[2] else 0.0,
        "sari": sum(sari_score) / len(sari_score) if sari_score else 0.0,
        "accuracy": total_score / total_num if total_num > 0 else 0.0,
        "similarity": sum(similarity_score) / len(similarity_score) if similarity_score else 0.0,
    }

    print(f"In {dataset_name}: {result}")
    return result, wrong_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=False)
    args = parser.parse_args()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.input_file), "r")]


    dataset_name = args.input_file.split("/")[-2].replace(".jsonl", "")

    eval_result, wrong_idx = evaluate(questions, dataset_name)
        
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for item in wrong_idx:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
