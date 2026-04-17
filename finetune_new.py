#!/usr/bin/env python
# coding=utf-8
import time
import os
import sys
sys.path.append(".")
import argparse
import logging
import math
import random
import datasets
import torch
import copy
from functools import partial
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from run_src.ft.utils.helper_wandb import init_wandb_training
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional, Dict, Sequence
import json
import torch
torch.cuda.empty_cache()


import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM
)
from peft import LoraConfig, TaskType, get_peft_model

logger = get_logger(__name__)

def ft_parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="The entity to store runs under.",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="The project to store runs under.",
    )
    parser.add_argument(
        "--ft_type",
        type=str,
        default=None,
        choices = ["qp_wo_ins", "qp_w_ins", "q_gold", "gold"],
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the ?? Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--use_special_tokens",
        action="store_true",
        help=(
            "Use special tokens."
        ),
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args

def _tokenize_fn(text: str, tokenizer: transformers.PreTrainedTokenizer, max_seq_length: int) -> Dict:
    """Tokenize a list of strings."""
    input_ids = labels = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
    ).input_ids
    input_ids_lens = labels_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()
    # print(input_ids_lens)

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    # "prompt_no_input": (
    #     "### Instruction:\n{instruction}\n\n### Response:\n"
    # ),
    "prompt_no_input": (
        "{instruction}"
    ),
}

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, ft_type, context_markups=None):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space

    # print(context_markups)
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if ft_type == "q_gold":
        example["instruction"] = example["instruction2"]
        example["output"] = example["output_gold"]
    
    elif ft_type == "gold":
        # ins = """A chat between a curious user and an AI assistant. The assistant gives step-by-step solutions to the user\'s questions. In the end of assistant\'s response, a final answer is given in the format of "The answer is: <ANSWER>.", where <ANSWER> should be a numeric number.\n\n### Instruction:\n"""
        # ins = """A chat between a curious user and an AI assistant. The assistant gives step-by-step solutions to the user\'s questions. In the end of assistant\'s response, a final answer is given in the format of "The answer is: <ANSWER>.", where <ANSWER> should be a numeric number.\n\n### Q: """        
        ins = """### Instruction: Please solve the following question step-by-step, showing your complete reasoning process. Conclude your response with "The answer is: <ANSWER>." where <ANSWER> is a numeric value without any additional text.\n### Question: """        
        example["instruction"] = ins + example["instruction"] + "\n### Answer: Let's think step by step."
        example["output"] = example["response"]
        
    elif ft_type == "qp_wo_ins":
        example["instruction"] = example["instruction2"]
        example["output"] = example["output2"]

    source_text = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    target_text = example['output'] + tokenizer.eos_token
    # print(f"source_text: {source_text}")
    # print("===========================")
    # print(f"target_text: {target_text}")
    # import pdb
    # pdb.set_trace()
    examples_tokenized = _tokenize_fn(source_text + target_text, tokenizer, max_seq_length)
    sources_tokenized = _tokenize_fn(source_text, tokenizer, max_seq_length)

    input_ids = examples_tokenized["input_ids"].flatten()
    source_len = sources_tokenized["input_ids_lens"]
    labels = copy.deepcopy(input_ids)
    labels[ :source_len-1] = -100

    if context_markups is not None:
        context_start = False
        for j, orig_token in enumerate(labels[source_len:]):
            if context_start is False and orig_token == context_markups[0]:
                context_start = True
                assert labels[source_len+j] == context_markups[0]
                start_idx = j+source_len
                end_idx = None
                for k, orig_token_2 in enumerate(labels[start_idx:]):
                    if orig_token_2 == context_markups[1]:
                        end_idx = start_idx + k
                if end_idx is None:
                    end_idx =  start_idx + k
                else:
                    assert labels[end_idx] == context_markups[1]
                labels[start_idx+1:end_idx] = -100
                context_start = False
    attention_mask = torch.ones_like(input_ids)

    # print("======================")
    # print(f"input_ids: {input_ids.shape}")
    # print(f"input_ids: {input_ids}")
    # print(f"labels: {labels.shape}")
    # print(f"labels: {labels}")
    # print(f"attention_mask: {attention_mask.shape}")
    # print(f"attention_mask: {attention_mask}")
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten()
    }

def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
        

def obtain_dataset_name(file):
    if "GSM8K" in file:
        dataset = "GSM8K"
    elif "MATH" in file:
        dataset = "MATH"
    elif "STG" in file:
        dataset = "STG"
    elif "SVAMP" in file:
        dataset = "SVAMP"
    return dataset

def run_ft(args):
    if "wandb" in args.report_to or "all" in args.report_to:
        init_wandb_training(args)
    
    # A hacky way to make llama work with flash attention
    if args.use_flash_attn:
        from run_src.ft.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    # import pdb
    # pdb.set_trace()
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # import pdb
    # pdb.set_trace()
    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=True)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=True
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)


    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    tokenizer.pad_token = tokenizer.eos_token
    # print(args.use_special_tokens)
    # import pdb 
    # pdb.set_trace()
    # if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast): # or isinstance(tokenizer, PreTrainedTokenizerFast)
    #     special_token_dict = {}
    #     if args.use_special_tokens is True:
    #         special_token_dict = {"additional_special_tokens": ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]", "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]", "[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]}
    #     special_token_dict["bos_token"] = "<s>"
    #     special_token_dict["eos_token"] = "</s>"
    #     special_token_dict["unk_token"] = "<unk>"
    #     special_token_dict["pad_token"] = "<pad>"
    #     num_added_tokens = tokenizer.add_special_tokens(special_token_dict)
    #     if args.use_special_tokens is False:
    #         assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    #     else:
    #         assert num_added_tokens > 10, "special tokens must be added to the original tokenizers."
        
    # elif isinstance(tokenizer, GPTNeoXTokenizerFast):
    #     num_added_tokens = tokenizer.add_special_tokens({
    #         "pad_token": "<pad>",
    #     })
    #     assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    # elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
    #     num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().num_embeddings  # embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        logger.info("Initializing LORA model...")
        modules_to_save = ["embed_tokens"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_rank, 
            #modules_to_save=modules_to_save,
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    context_markups = []
    encode_function = partial(
        encode_with_prompt_completion_format,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        ft_type = args.ft_type, 
        context_markups=context_markups if args.use_special_tokens is True else None
    )
    
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())  # filter examples with labels are all -100

    # import pdb
    # pdb.set_trace()
    train_dataset = lm_datasets["train"]
    train_dataset = Dataset.from_dict({
        'input_ids': [item['input_ids'].tolist() for item in train_dataset],
        'labels': [item['labels'].tolist() for item in train_dataset],
        'attention_mask': [item['attention_mask'].tolist() for item in train_dataset]
    })
    # with open("processed.json", "w") as outfile:
    #     new_data = []
    #     for item in train_dataset:
    #         # print(item)
    #         labels = [int(i) for i in item["labels"]]
    #         input_ids = [int(i) for i in item["input_ids"]]
    #         new_data.append({"labels": labels, "input_ids": input_ids})
    #     json.dump(new_data, outfile)
    # # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # import pdb
    # pdb.set_trace()
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("open_instruct", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    # import pdb
    # pdb.set_trace()
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and completed_steps < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= args.max_train_steps:
                    break
        
        
        if args.checkpointing_steps == "epoch" and epoch == args.num_train_epochs:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    # import pdb
    # pdb.set_trace()
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        print(accelerator.is_main_process)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        unwrapped_model = accelerator.unwrap_model(model)
        # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
        # Otherwise, sometimes the model will be saved with only part of the parameters.
        # Also, accelerator needs to use the wrapped model to get the state_dict.
        state_dict = accelerator.get_state_dict(model)
        if args.use_lora:
            # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
            # and has its own save_pretrained function for only saving lora modules.
            # We have to mannually specify the is_main_process outside the save_pretrained function.
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
        else:
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
            )


def run_test(args):
    from models.IO_System import IO_System
    from common.utils import read_txt, read_json, save_json
    # from eval_src.Evaluator import GSM8KEvaluator, MATHEvaluator, STGEvaluator, SVAMPEvaluator
    from common.arguments import get_parser
    import ast
    parser = get_parser()
    args_test = parser.parse_args()
    if "GSM" in args.test_name and "HARD" not in args.test_name:
        evaluator = eval("GSM8KEvaluator()")
    elif "MATH" in args.test_name:
        evaluator = eval("MATHEvaluator()")
    else:
        evaluator = eval(f"{args.test_name}Evaluator()")
    
    if args_test.args.api == "huggingface":
        from models.HuggingFace_API import load_HF_model
        tokenizer, model = load_HF_model(args.output_dir)
    elif args_test.args.api == "vllm":
        from models.vLLM_API import load_vLLM_model
        tokenizer, model = load_vLLM_model(args.output_dir, args_test.args.seed, args_test.args.tensor_parallel_size, args_test.args.half_precision)
    elif args_test.args.api == "gpt3.5-turbo":
        from models.OpenAI_API import load_OpenAI_model
        tokenizer, model = load_OpenAI_model(args.output_dir)
    
    io = IO_System(args_test, tokenizer, model)
    data_item_list = read_json(args.test_file)
    
    instruction = "### Instruction: Given a question, you are required to solve it step by step. For each step, first select the most appropriate action from the following five predefined actions, then generate a corresponding reasoning step. Each step should end in the format of 'The answer is: <ANSWER>.' The five actions are defined as follows:\n1. Action 1: System Analysis -- Identify the conditions necessary for solving the problem and rephrase the problem for clarity.\n2. Action 2: One-Step Thought -- Present a single, focused idea that could help move toward solving the problem.\n3. Action 3: Chain-of-Thought -- Break down the problem step-by-step and reason through each part logically.\n4. Action 4: Divide and Conquer -- Break the problem into smaller subproblems and solve them individually.\n5. Action 5: Self-Refine -- Reflect on your progress, refine your approach, and ensure your solution is coherent and complete.\nNote that, only output concise reasoning steps without explanations or unnecessary content.\n\n### Question: "
    io_inputs = []
    correct, total_correct = 0, 0
    for i, data_item in enumerate((pbar := tqdm(data_item_list))):
        total_correct += 1
        try:
            problem_id, problem, gt_solution = data_item["id"], data_item["problem"], data_item["solution"]
        except:
            problem_id = 0
            try:
                problem, gt_solution = data_item["problem"], data_item["solution"]
            except:
                problem, gt_solution = data_item["question"], data_item["answer"]
        gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)
        
        if args.ft_type == "q_gold":
            io_inputs.append(f"### Question: {problem}\n### Answer: ")
        elif args.ft_type == "qp_wo_ins":
            io_inputs.append(f"### Question: {problem}\n### Answer: ")
        elif args.ft_type == "qp_w_ins":
            io_inputs.append(f"{instruction}{problem}\n\n### Answer: ")
        
        if i % 50 == 0:
            io_outputs = io.generate(model_input=io_inputs, max_tokens=2048, num_return=1, stop_tokens=["\n\n"])
            for io_output in io_outputs:
                if args.ft_type == "q_gold":
                    if args.test_name == "GSM8K":
                        output = io_output.split("####")[-1]
                    elif args.test_name == "MATH":
                        output = io_output.replace(io_output.split("\\boxed")[0], "")
                    elif args.test_name == "STG":
                        output = io_output
                    elif args.test_name == "SVAMP":
                        output = io_output
                
                elif args.ft_type in ["qp_wo_ins", "qp_w_ins"]:
                    output = io_output.split("\nStep ")[-1].split("answer is")[-1].strip()
                
                correct += evaluator.check_answers_equiv(output, gt_answer)
            
            acc = 1.0*correct / total_correct
            print(f"===> 0-{i} acc: {acc:.4f}")
    
    print(f"===> Final acc: {acc:.4f}")
        

if __name__ == "__main__":
    args = ft_parse_args()
    args.tokenizer_name = args.model_name_or_path
    model_id = args.model_name_or_path.split("/")[-1]
    args.train_name = obtain_dataset_name(args.train_file)
    args.test_name = obtain_dataset_name(args.test_file)
    args.output_dir = f"run_outputs/structure_all/ft_checkpoint/{model_id}/epoch{args.num_train_epochs}_type_{args.ft_type}_train_{args.train_name}/"
    
    # run fine-tuning
    start_time = time.time()
    # import pdb
    # pdb.set_trace()
    run_ft(args)
    end_time = time.time()
    # 计算总时间的小时、分钟和秒
    total_seconds = end_time - start_time
    total_hours = int(total_seconds // 3600)
    total_minutes = int((total_seconds % 3600) // 60)
    total_seconds_remainder = total_seconds % 60
    print(f"==> Total time: {total_hours}h {total_minutes}m {total_seconds_remainder:.2f}s")
    
    
    
    # run testing
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # import pdb
    # pdb.set_trace()
    # run_test(args)
    print(args.output_dir)