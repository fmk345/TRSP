#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import warnings
from typing import List, Optional, Tuple, Union
import os
import random
import datasets
import torch
import torch.nn as nn
import copy
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset,Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional, Dict, Sequence
import json
from utils.model_utils import get_llm
from utils.onoff_utils.onoff import block_replace,scaling_replace, turn_off, turn_on
from utils.data_utils import *
from utils.block_remove import block_remove
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM
)
from peft import LoraConfig, TaskType, get_peft_model
from utils.opt.modeling_opt import ScaleOPTForCausalLM

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1024,
        help="The number of samples.",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.01,
        help="The lamda of scaling_factor.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=32,
        help="The number of layers.",
    )
    parser.add_argument(
        "--scale",
        action='store_true',
        default=False,
        help="Whether use scale?",
    )
    parser.add_argument(
        "--eval_zeroshot",
        action='store_true',
        default=False,
        help="Whether use scale?",
    )
    parser.add_argument(
        "--eval_ppl",
        action='store_true',
        default=False,
        help="Whether evalaute ppl?",
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Whether train?",
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
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
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
        default=1,
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
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        default=None,
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
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         assert extension in ["json", "jsonl","parquet"], "`train_file` should be a json/jsonl file."
    return args

class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids


def main():
    args = parse_args()

    # A hacky way to make llama work with flash attention
    if args.use_flash_attn:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,**accelerator_log_kwargs)

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
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if 'opt' in args.model_name_or_path.lower():
        def skip(*args, **kwargs):
            pass
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        model = ScaleOPTForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
        model.name=args.model_name_or_path
        num_layers = len(model.model.decoder.layers)
    # elif 'llama' in args.model_name_or_path.lower():
    #     def skip(*args, **kwargs):
    #         pass
    #     torch.nn.init.kaiming_uniform_ = skip
    #     torch.nn.init.uniform_ = skip
    #     torch.nn.init.normal_ = skip
    #     model = ScaleLlamaForCausalLM.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #         low_cpu_mem_usage=args.low_cpu_mem_usage,
    #     )
    #     model.name=args.model_name_or_path
    #     num_layers = len(model.model.layers)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
    #add the pad_token
    if 'opt' in model.name.lower():
        num_layers = len(model.model.decoder.layers)
    elif 'llama' in model.name.lower():  
         num_layers = len(model.model.layers)
    tokenizer.pad_token = tokenizer.eos_token
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
    with accelerator.main_process_first():
        train_dataset=get_loaders(args.dataset_name,nsamples=args.nsamples,seed=0, seqlen=2048, tokenizer=tokenizer)[0]
        # dataset = Dataset.from_dict({'input_ids': [item['input_ids'].tolist() for item in train_dataset],
        # 'labels': [item['labels'].tolist() for item in train_dataset],
        # 'attention_mask': [item['attention_mask'].tolist() for item in train_dataset]})
    
    # train_dataset = dataset
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
    with open('/home/fmk/retrieval_lm/model_parameters.txt', 'w') as f:
        for n, p in model.named_parameters():
            f.write(f"{n}: {p}\n")  
    # import pdb
    # pdb.set_trace()
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
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
    # import pdb
    # pdb.set_trace()
    # Prepare everything with `accelerator`.
    # import pdb
    # pdb.set_trace()
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
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
    # use_cache = model.config.use_cache
    use_cache=True
    #define lamda_reg for scaling_factor
    # alive_list = [i for i in range(num_layers)]
    # alive_list_select= alive_list[3:-4]  
    removal_list = []
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        scaling_list=[]
        for step, batch in enumerate(train_dataloader):
            l1_norm=0
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and completed_steps < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                for i in alive_list:
                    l1_norm+= torch.norm(model.model.decoder.layers[i].scaling_factor.weight, p=1)
                # scaling_list=outputs.scaling_factors
                # loss= outputs.loss
                import pdb
                pdb.set_trace()
                loss = outputs.loss+100*l1_norm
                print(loss)
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                # print(f"Gradient of scaling_factor: {model.model.decoder.layers[0].scaling_factor}")
                # print(f"Gradient of scaling_factor: {model.model.decoder.embed_tokens}")
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
        import pdb
        pdb.set_trace()
        for i in alive_list_select:
            scaling_list.append((model.model.decoder.layers[i].scaling_factor.weight,i))
        scaling_list.sort(key=lambda x: x[0])
        min_layer_index=scaling_list[0][1]
        print("======================")
        print("the min scale layer is:",min_layer_index)
        print(scaling_list)
        print("======================")
        alive_list.remove(min_layer_index)
        alive_list_select.remove(min_layer_index)
        model.model.decoder.layers[min_layer_index].turn_off()
        removal_list.append(min_layer_index)
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
        print(removal_list[0])
        print(model.model.decoder.layers[removal_list[0]].pass_layer)        
        print(model.model.decoder.layers[removal_list[0]].scaling_factor.weight)
    # removal_list=[2,5,12]
    for epoch in range(starting_epoch,args.num_train_epochs):
        model.train()
        model.model.decoder.layers[removal_list[epoch]].res()
        total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch:
            if resume_step is not None and completed_steps < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                    completed_steps += 1
                continue
        
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
                # scaling_list=outputs.scaling_factors
                # loss= outputs.loss
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                # print(f"Gradient of scaling_factor: {model.model.decoder.layers[0].scaling_factor}")
                # print(f"Gradient of scaling_factor: {model.model.decoder.embed_tokens}")
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
        model.model.decoder.layers[removal_list[epoch]].turn_off()

    # if args.eval_ppl:
    #     print(f"Starting PPL evaluation...")
    #     model.eval()
    #     model_ppl = block_remove(model, copy.deepcopy(removal_list))
    #     model_ppl.use_cache= use_cache
    #     model_ppl.seqlen=args.max_seq_length
    #     import pdb
    #     pdb.set_trace()
    #     w2_ppl = load_and_eval_ppl(model_ppl, device=torch.device("cuda:0"), dataset='wikitext2')
    #     print(f"WikiText-2 PPL = {w2_ppl:.2f}")
    #     c4_ppl = load_and_eval_ppl(model_ppl, device=torch.device("cuda:0"), dataset='c4')
    #     print(f"C4 PPL = {c4_ppl:.2f}")
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
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
    # if  args.eval_zeroshot:
    #     print(f"Starting Zero-shot tasks evaluation...")
    #     if '30b' in args.output_dir or '66b' in args.output_dir or '70b' in args.output_dir:
    #         parallelize = True
    #     else:
    #         parallelize = False

    #     tasks = ['piqa','winogrande','hellaswag','arc_challenge','arc_easy']

    #     results = eval_zero_shot(model.name, copy.deepcopy(removal_list), tasks, parallelize=parallelize)
    #     results = results['results']

    #     for task in tasks:
    #         print(f"{task}: {results[task]}")
    import pdb
    pdb.set_trace()
    if args.with_tracking:
        accelerator.end_training()



if __name__ == "__main__":
    main()