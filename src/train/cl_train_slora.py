import argparse
import json
import os
from tqdm import tqdm

import torch
from safetensors.torch import save_file, load_file

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, Trainer
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
from datasets import Dataset
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config
)

from src.model.builder import load_denoised_lora

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ScriptArguments(ScriptArguments):
    dataset_name: str = field(metadata={"help": "Dataset name."}, default=None)
    train_data_path: str = field(metadata={"help": "Path to the training data."}, default=None)
    val_data_path: str = field(metadata={"help": "Path to the validation data."}, default=None)
    task_id: int = field(default=0)
    mode: str = field(default='max', metadata={"help": "Mode for denoising."})

@dataclass
class ModelArguments(ModelConfig):
    model: str = field(metadata={"help": "name of model."}, default=None)

def return_prompt_and_responses(samples):
    new_list = []
    for sample in samples:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": sample["conversations"][0]["value"]},
            {"role": "assistant", "content": sample["conversations"][1]["value"]},
        ]   
        new_list.append({"messages": messages})
    return new_list

def obtain_dataset(data_path):
    with open(data_path, "r") as f:
        datas = json.load(f)
    data_list = return_prompt_and_responses(datas)
    dataset = Dataset.from_list(data_list)
    return dataset 

def safe_save_model_for_hf_trainer(trainer: Trainer,
                                   output_dir: str):

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def calculate_similarity(U_c_r, base_U_r, mode=None):
    if mode == 'l2':
        return torch.norm(U_c_r - base_U_r, p=2).item()
    elif mode == 'cosine':
        return torch.cosine_similarity(U_c_r.flatten(), base_U_r.flatten(), dim=0).item()
    else:
        return torch.norm(torch.mm(U_c_r.T, base_U_r), p="fro").item()

def perform_similarity_search(delta_W_tensor, base_U_r, rank, mode):
    print(f"rank: {rank}", flush=True)
    ratio_candidates = [0.1 * i for i in range(1, 11)]
    max_similarity = -float("inf")
    min_similarity = float("inf")
    best_c = rank
    new_lora_A, new_lora_B = None, None

    for ratio in ratio_candidates:
        c = max(int(rank * ratio), 1)
        Omega = torch.randn(delta_W_tensor.shape[1], c, device=delta_W_tensor.device)
        Y = torch.mm(delta_W_tensor, Omega)
        Q, _ = torch.linalg.qr(Y, mode='reduced')
        B_ = torch.mm(Q.T, delta_W_tensor)
        U_hat, Sigma, Vt = torch.linalg.svd(B_, full_matrices=False)
        U = torch.mm(Q, U_hat)

        if mode == 'minor':
            U_c = U[:, -c:]
            Sigma_c = Sigma[-c:]
            V_c = Vt[-c:, :]
        else:
            U_c = U[:, :c]
            Sigma_c = Sigma[:c]
            V_c = Vt[:c, :]

        delta_W_c = torch.mm(torch.mm(U_c, torch.diag(Sigma_c)), V_c)
        U_c, _, _ = torch.linalg.svd(delta_W_c, full_matrices=False)
        U_c_r = U_c[:, :rank]

        total_similarity = 0
        if base_U_r is not None:
            similarity_base = calculate_similarity(U_c_r, base_U_r, mode)
            total_similarity += similarity_base

        if mode == 'min':
            if total_similarity < min_similarity:
                min_similarity = total_similarity
                best_c = c
                new_lora_A = V_c
                new_lora_B = torch.mm(U, torch.diag(Sigma_c))           
        else:
            if total_similarity > max_similarity:
                max_similarity = total_similarity
                best_c = c
                new_lora_A = V_c
                new_lora_B = torch.mm(U, torch.diag(Sigma_c))
        print(f"Best c determined by Sim-Search: {best_c}", flush=True)
    return new_lora_A, new_lora_B

def denoising(base_model, delta_weights, mode='max'):
    if base_model is not None:
        with torch.no_grad():
            base_weights = dict(base_model.named_parameters())

    denoised_delta_weights = {}
    for name, delta_W in tqdm(delta_weights.items()):
        if "lora_A" not in name:
            continue
        
        lora_B_name = name.replace("lora_A", "lora_B")
        if lora_B_name not in delta_weights:
            print(f"Warning: {lora_B_name} not found in LoRA weights. Skipping this parameter.", flush=True)
            continue

        lora_A = delta_W.detach()
        lora_B = delta_weights[lora_B_name].detach()
        rank = lora_A.shape[0]
        base_U_r = None
        if base_model is not None:
            base_weight_name = name.replace("base_model.model.", "").replace(".lora_A.weight", ".base_layer.weight")
            base_weight_name_1 = name.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            if base_weight_name in base_weights:
                base_W_tensor = base_weights[base_weight_name].detach()
            elif base_weight_name_1 in base_weights:
                base_W_tensor = base_weights[base_weight_name_1].detach()
            else:
                print(f"Warning: {base_weight_name}  or {base_weight_name_1} not found in base weights. Skipping this parameter.", flush=True)
                
                continue
            base_U, _, _ = torch.linalg.svd(base_W_tensor.to(torch.float32), full_matrices=False)
            base_U_r = base_U[:, :rank]


        delta_W_tensor = torch.mm(lora_B, lora_A)
        delta_W_tensor_32 = delta_W_tensor.to(torch.float32)

        new_lora_A, new_lora_B = perform_similarity_search(delta_W_tensor_32, base_U_r, rank, mode)

        denoised_delta_weights[name] = new_lora_A.to(delta_W_tensor.dtype).contiguous()
        denoised_delta_weights[lora_B_name] = new_lora_B.to(delta_W_tensor.dtype).contiguous()

    return denoised_delta_weights


def train_continual_learning():
    global local_rank

    parser = make_parser()
    script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.dataset_num_proc = 8
    local_rank = training_args.local_rank
    model_config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    task_id = script_args.task_id
    rank0_print('task', script_args.train_data_path)

    ################
    # Model & Tokenizer
    ################
    config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, config=config)

    model = base_model
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        if "llama3" in model_config.model:
            tokenizer.pad_token="<|finetune_right_pad_id|>"
            tokenizer.pad_token_id=128004
        else:
            raise NotImplementedError
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    output_dir_parent = os.path.dirname(training_args.output_dir)
    
    rank0_print(f"Training task {task_id}: {script_args.train_data_path}")  
    if task_id > 1:
        for prev_order in range(1, task_id):
            previous_task_model_path = os.path.join(output_dir_parent, f"order{prev_order}")
            lora_config = LoraConfig.from_pretrained(previous_task_model_path)
            previous_task_lora_path = os.path.join(previous_task_model_path, f"{script_args.mode}.safetensors")
            print('Load LoRA from previous task:', previous_task_lora_path, flush=True)
            state_dict = load_file(previous_task_lora_path)
            model = load_denoised_lora(model, state_dict, lora_config)
    rank0_print("Adding LoRA adapters...")
    model = get_peft_model(model, get_peft_config(model_config))


   
    ################
    # Dataset
    ################
    train_data = obtain_dataset(script_args.train_data_path)
    eval_data = obtain_dataset(script_args.val_data_path) if script_args.val_data_path is not None else None

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    ################
    # After Training - Denoising
    ################
    if training_args.local_rank == 0:
        rank0_print(f"Pruning LoRA weights after task {task_id}...")
        state_dict = load_file(os.path.join(training_args.output_dir, "adapter_model.safetensors"))
        state_dict = {key: value.to(base_model.device) for key, value in state_dict.items()}
        denoised_state_dict = denoising(base_model, state_dict, script_args.mode)
        save_file(denoised_state_dict, os.path.join(training_args.output_dir, f"{script_args.mode}.safetensors"))

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelArguments)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

if __name__ == "__main__":
    train_continual_learning()

