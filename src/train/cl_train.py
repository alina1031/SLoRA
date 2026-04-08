import argparse
import pathlib
import json
import os

import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, Trainer, set_seed
from peft import PeftModel, get_peft_model
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
    """收集模型状态字典并保存到磁盘"""

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

def train_continual_learning():
    global local_rank

    parser = make_parser()
    script_args, training_args, model_config = parser.parse_args_and_config()
    # training_args.seed = 2025
    set_seed(training_args.seed)
    training_args.dataset_num_proc = 8
    local_rank = training_args.local_rank
    model_config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    task_id = script_args.task_id
    rank0_print('task', script_args.train_data_path)

    ################
    # Model & Tokenizer
    ################
    config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, config=config)
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

    rank0_print(f"Training task {task_id}: {script_args.train_data_path}")  
    if task_id > 1:
        for prev_order in range(1, task_id):
            output_dir_parent = os.path.dirname(training_args.output_dir)    
            previous_task_model_path = os.path.join(output_dir_parent, f"order{prev_order}")
            model = PeftModel.from_pretrained(model, previous_task_model_path)
            model = model.merge_and_unload()
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

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelArguments)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

if __name__ == "__main__":
    train_continual_learning()