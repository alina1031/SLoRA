import os
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch
from peft import PeftModel, LoraConfig, get_peft_model
from safetensors.torch import save_file, load_file
import re 



def load_state_dict(path):
    state_dict = {}
    for file_name in os.listdir(path):
        if file_name.endswith('.safetensors') and 'max' not in file_name:
            print(f'Load Parameters from {file_name}')
            file_path = os.path.join(path, file_name)
            state_dict.update(load_file(file_path))
    return state_dict



def load_denoised_lora(model, delta_weight, lora_config):
    memory = {}
    delta_weight = {key.replace("weight", "default.weight"):name for key, name in delta_weight.items()}
    for key, param in delta_weight.items():
        if "lora_A" in key:
            cur_r = param.shape[0]
            memory[cur_r] = memory.get(cur_r, [])
            memory[cur_r].extend([key, key.replace("lora_A", "lora_B")])
    
    for rank, key_list in tqdm(memory.items()):
        print(rank)
        related_layers = sorted(list(set([int(x.split(".")[4]) for x in key_list])))
        print(related_layers)
        related_modules = list(set([".".join(x.split(".")[5:7]) for x in key_list]))
        print(related_modules)
        print("=" * 100)
        cur_config = deepcopy(lora_config)
        cur_config.r = rank 
        cur_config.lora_alpha = rank # NOTE: this is a non-trivial trial from empirical results
        target_modules = list(set([".".join(x.split(".")[3:7]) for x in key_list]))
        cur_config.target_modules = target_modules 
        model = get_peft_model(model, cur_config)
        load_delta_weight = {name: delta_weight[name] for name in key_list}
        model.load_state_dict(load_delta_weight, strict=False)
        # model.load_state_dict(load_delta_weight, strict=False, assign=True)
        model = model.merge_and_unload()
        
    print('Convert to FP16...')
    model.to(torch.float16)
    return model


def to_safetensors_format(state_dict):
    safe_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            v = v.detach()
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"{k} is not a tensor")
        safe_dict[k] = v.contiguous()
    return safe_dict


def calculate_similarity(U_c_r, base_U_r, mode, rank=None):
    if mode == 'l2':
        return torch.norm(U_c_r - base_U_r, p=2).item()
    elif mode == 'cosine':
        return torch.cosine_similarity(U_c_r.flatten(), base_U_r.flatten(), dim=0).item()
    else:
        similarity = torch.norm(torch.mm(U_c_r.T, base_U_r), p="fro").item()
        return similarity

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
            similarity_base = calculate_similarity(U_c_r, base_U_r, mode, rank)
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
        print(f"Max similarity: {max_similarity}", flush=True)
    return new_lora_A, new_lora_B

def denoising(base_model, delta_weights, mode='max'):
    if base_model is not None:
        with torch.no_grad():
            base_weights = dict(base_model.named_parameters())

    denoised_delta_weights = {}
    for name, delta_W in tqdm(delta_weights.items()):
        if "lora_A" not in name:
            continue

        print(f"Processing parameter: {name}", flush=True)

        lora_B_name = name.replace("lora_A", "lora_B")
        if lora_B_name not in delta_weights:
            print(f"Warning: {lora_B_name} not found in LoRA weights. Skipping this parameter.", flush=True)
            continue

        lora_A = delta_W.detach()
        lora_B = delta_weights[lora_B_name].detach()
        rank = lora_A.shape[0]
        base_U_r = None
        if base_model is not None:
            base_weight_name = name.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            if base_weight_name not in base_weights:
                print(f"Warning: {base_weight_name} not found in base weights. Skipping this parameter.", flush=True)
                continue

            base_W_tensor = base_weights[base_weight_name].detach()
            base_U, _, _ = torch.linalg.svd(base_W_tensor.to(torch.float32), full_matrices=False)
            base_U_r = base_U[:, :rank]

        delta_W_tensor = torch.mm(lora_B, lora_A)
        delta_W_tensor_32 = delta_W_tensor.to(torch.float32)

        new_lora_A, new_lora_B = perform_similarity_search(delta_W_tensor_32, base_U_r, rank, mode)

        denoised_delta_weights[name] = new_lora_A.to(delta_W_tensor.dtype).contiguous()
        denoised_delta_weights[lora_B_name] = new_lora_B.to(delta_W_tensor.dtype).contiguous()

    return denoised_delta_weights

def resolve_task_paths(model_path):
    if not os.path.isdir(model_path):
        raise ValueError(f"model_path does not exist or is not a directory: {model_path}")

    items = os.listdir(model_path)

    order_dirs = []
    for item in items:
        item_path = os.path.join(model_path, item)
        if os.path.isdir(item_path) and re.fullmatch(r"order\d+", item):
            order_id = int(item.replace("order", ""))
            order_dirs.append((order_id, item_path))

    if order_dirs:
        order_dirs = sorted(order_dirs, key=lambda x: x[0])
        return [path for _, path in order_dirs]


    checkpoint_markers = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin"
    ]
    if any(os.path.exists(os.path.join(model_path, f)) for f in checkpoint_markers):
        return [model_path]

    raise ValueError(
        f"Cannot find valid task folders under {model_path}. "
        f"Expected either order* subfolders or a checkpoint directory itself."
    )


def load_continual_pretrained_model(model_path, model_base, model_name, denoised_mode, load_8bit=False, load_4bit=False, use_logit_bias=False, device_map="auto", device="cuda", test_order=0, denoised_num=0):
    kwargs = {"device_map": device_map}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    
    if model_base is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, trust_remote_code=True,**kwargs)
        base_model = model
        print(f"Loading LoRA weights from {model_path}")
        # task_num = len(os.listdir(model_path))
        task_paths = resolve_task_paths(model_path)
        task_num = len(task_paths)
        print(f"There are {task_num} tasks in total.")
        # for idx, item in enumerate(os.listdir(model_path)):
        for idx, item_path in enumerate(task_paths):
            order_id = idx + 1
            if test_order != 0 and order_id == test_order+1:
                break

            # item_path = os.path.join(model_path, 'order'+str(order_id))
            print("model_path", item_path)
            lora_config = LoraConfig.from_pretrained(item_path)

            if denoised_mode=='slora_pre':
                denoised_lora_path = os.path.join(item_path, f"max.safetensors")
                if test_order !=0 and denoised_num !=0 and order_id < denoised_num:
                    denoised_lora_path = os.path.join(item_path, "adapter_model.safetensors")
                print('Load LoRA from previous task:', denoised_lora_path, flush=True)
                state_dict = load_file(denoised_lora_path)
                model = load_denoised_lora(model, state_dict, lora_config)
            elif denoised_mode == 'slora_post':
                denoised_lora_path = os.path.join(item_path, f"max.safetensors")
                if test_order !=0 and denoised_num !=0 and order_id < denoised_num:
                    denoised_lora_path = os.path.join(item_path, "adapter_model.safetensors")
                if denoised_lora_path and not os.path.exists(denoised_lora_path):
                    print(f"Loading LoRA weights from {item_path}")
                    lora_config = LoraConfig.from_pretrained(item_path)

                    state_dict = load_file(os.path.join(item_path, "adapter_model.safetensors"))
                    state_dict = {key: value.to(base_model.device) for key, value in state_dict.items()}

                    new_state_dict = denoising(base_model, state_dict)

                    # Save the optimized model
                    save_file(new_state_dict, denoised_lora_path)
                print(f"Loading pre-optimized model from {denoised_lora_path}")
                lora_config = LoraConfig.from_pretrained(item_path)
                denoised_state_dict = load_file(denoised_lora_path)
                model = load_denoised_lora(model, denoised_state_dict, lora_config)
            else:
                model = PeftModel.from_pretrained(model, item_path, config=lora_config)            
                model = model.merge_and_unload()
                print(f"Loading LoRA weights from {item_path}")
                print('Convert to FP16...')
                model.to(torch.float16)


    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
        
        model.to(torch.float16)


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if use_logit_bias:
        if model_base is not None:
            # lora case
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base , add_prefix_space=True, trust_remote_code=True)
        else:
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)
        if tokenizer_with_prefix_space.pad_token_id is None:
            tokenizer_with_prefix_space.pad_token_id = tokenizer_with_prefix_space.eos_token_id
    else:
        tokenizer_with_prefix_space = None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 
        
    return tokenizer, model, context_len, tokenizer  