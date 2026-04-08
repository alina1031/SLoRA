import argparse
import torch
import os
import json
from tqdm import tqdm, trange

from src.conversations import conv_templates
from src.model.builder import load_continual_pretrained_model
from src.utils import disable_torch_init, get_model_name_from_path
from torch.utils.data import DataLoader
import pandas as pd 
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset:
    def __init__(self, questions, batch_size, conv_mode, task_specific_prompt):
        self.questions = questions
        self.batch_size = batch_size
        self.size = len(questions)
        self.conv_mode = conv_mode
        self.conv = conv_templates[conv_mode].copy()
        self.task_specific_prompt = task_specific_prompt

    def __getitem__(self, index):
        bz = self.batch_size

        # return question, ansewr, additional info
        questions = []
        prompts = []
        answers = []
        additional_infos = []
        for i in range(index*bz, (index+1)*bz):
            if i < self.size:
                conv = self.conv.copy()

                line = self.questions[i]
                question = line['conversations'][0]['value']
                questions.append(question)
                # print('before_promt:', conv.get_prompt(), flush=True)
                # conv.append_message(conv.roles[0], question+self.task_specific_prompt)
                if  self.conv_mode == 'qwen':
                    conv.append_message(conv.roles[0], "/no_think\n" + question + self.task_specific_prompt)
                else:
                    conv.append_message(conv.roles[0], question+self.task_specific_prompt)
                conv.append_message(conv.roles[1], None)
                # print(conv, flush=True)/
                # print('after_promt:', conv.get_prompt(), flush=True)
                prompts.append(conv.get_prompt())
                answers.append(line['conversations'][1]['value'] if len(line['conversations']) > 1 else None)
                additional_infos.append(line['eval'] if 'eval' in line else None)

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return questions, prompts, answers, additional_infos

    def __len__(self):
        return len(self.questions) // self.batch_size + 1

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < len(self.questions):
            item = self.questions[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration


# DataLoader
def create_data_loader(questions, tokenizer, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def convert_to_json(questions):
    # questions is a pandas dataframe, which is to be converted to a list object
    # each element in the list is a dictionary
    # the column name of questions is the key of the dictionary
    # the value of the dictionary is the value of the corresponding column
    questions = questions.to_dict(orient='records')
    return questions

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # dataset_name = args.question_file.split("/")[-1].split(".")[0]
    dataset_name = args.question_file.split("/")[-2]
    print(dataset_name, flush=True)

    print("Load LoRA")
    tokenizer, model, _, tokenizer_with_prefix_space = load_continual_pretrained_model(model_path, args.model_base, model_name, args.denoised_mode, use_logit_bias=args.use_logit_bias, test_order=args.test_order, denoised_num=args.denoised_num)

    tokenizer.padding_side = "left"
    tokenizer_with_prefix_space.padding_side = "left"

    # load args.question_file, which is a csv file
    if args.question_file.endswith(".csv"):
        questions = pd.read_csv(args.question_file)
        questions = convert_to_json(questions)
    elif args.question_file.endswith(".jsonl"):
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    else:
        # a json file
        with open(args.question_file, 'r') as f:
            questions = json.load(f)
    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    if args.resume and os.path.exists(answers_file):
        current_file_num = 0
        with open(answers_file, 'r') as f:
            for line in f:
                current_file_num += 1
        questions = questions[current_file_num:]
        ans_file = open(answers_file, "a", encoding='utf-8')
    else:
        ans_file = open(answers_file, "w", encoding='utf-8')

    model: torch.nn.Module
    sequence_bias = None

    task_specific_prompt = ""
    if dataset_name == "Py150":
        task_specific_prompt = "\n\nPlease output only the next line of code, without adding anything else. Do not include explanations or comments. Treat <EOL> as a line break in the original code."
    elif dataset_name in ["NumGLUE-cm", "NumGLUE-ds"]:
        task_specific_prompt = "\n\nSolve the math problem and output only the final answer. Do not include any explanation, reasoning, or extra words."
        
    print("task_specific_prompt:", task_specific_prompt, flush=True)

    dataset = CustomDataset(questions, batch_size=args.batch_size, conv_mode=args.conv_mode, task_specific_prompt=task_specific_prompt)
    
    for idx in trange(len(dataset)):
        questions, prompts, answers, additional_infos = dataset[idx]
        if len(questions) == 0:
            break
        
        # print("[FIRST INPUT]: ", prompt)
        input_ids = tokenizer(prompts, return_tensors='pt', padding=True).input_ids
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)
        if args.conv_mode == 'llama3':
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            terminators = tokenizer.eos_token_id

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=terminators,
                sequence_bias=sequence_bias,
                use_cache=True)
        # print(input_ids.shape, output_ids.shape)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # print("original outputs: ",tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True) )
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        for prompt, output, answer, additional_info in zip(prompts, outputs, answers, additional_infos):
            if "assistant" in output:
                output = output.split("assistant", 1)[0] 
            ans_file.write(json.dumps({"prompt": prompt,
                                    "text": output,
                                    "solution": answer,
                                    "additional_info": additional_info,
                                    "model_id": model_name,
                                    "metadata": {}}, ensure_ascii=False) + "\n",)
        ans_file.flush()
        
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_logit_bias", action="store_true", default=False)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--denoised_mode", type=str, default="max")
    parser.add_argument("--test_order", type=int, default=0)
    parser.add_argument("--denoised_num", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)