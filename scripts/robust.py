""" Align the embeddings of noised data"""
import argparse
import json
import pdb
import logging
import copy
import math
import os
import random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AdamW,
    set_seed,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import numpy as np

logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description="Align the embeddings of a comrpessed encoder and a target")
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="The name of the train file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="loss temperature",
    )
    parser.add_argument(
        "--kldiv",
        action="store_true",
        help="If passed, will use a kldiv instead of cosine distance.",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default=None,
        help="The name of the dev file.",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="The name of the test file.",
    )
    parser.add_argument(
        "--cosine_margin",
        type=float,
        default=0.2,
        help="cosine loss"
    )
    parser.add_argument(
        "--positive_temp",
        type=float,
        default=0.6,
        help="positive temperature for cosine loss"
    )
    parser.add_argument(
        "--negative_temp",
        type=float,
        default=0.2,
        help="positive temperature for cosine loss"
    )
    parser.add_argument(
        "--anchor_temp",
        type=float,
        default=0.2,
        help="positive temperature for cosine loss"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='spacemanidol/esci-all-distilbert-base-uncased-5e-5',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the :hugging_face: Tokenizers library).",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="random Seed",
   )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    print(args)
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    set_seed(args.seed)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    data_files = {
            "train": args.train_file,
            "dev": args.dev_file,
            "test": args.test_file
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    padding = "max_length"
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['query'], max_length=args.max_length, padding=padding, truncation=True)
        positive_targets_inputs = tokenizer(examples['positive_anchor'], max_length=args.max_length, padding=padding, truncation=True)
        negative_targets_inputs = tokenizer( examples['negative_anchor'], max_length=args.max_length, padding=padding, truncation=True)
        model_inputs["positive_anchor_input_ids"] = positive_targets_inputs["input_ids"]
        model_inputs["negative_anchor_input_ids"] = negative_targets_inputs["input_ids"]
        model_inputs["positive_anchor_attention_mask"] = positive_targets_inputs["attention_mask"]
        model_inputs["negative_anchor_attention_mask"] = negative_targets_inputs["attention_mask"]
        return model_inputs

    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets['train'].column_names,
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets[ "dev"]
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size,drop_last =True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size,drop_last=True)

    no_decay = ["bias", "LayerNorm.weight"]
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    num_eval_steps = len(eval_dataloader)
    if args.checkpointing_steps == 'epoch':
        args.checkpointing_steps = num_update_steps_per_epoch

    total_batch_size = args.batch_size  * args.gradient_accumulation_steps    
    # Only show the progress bar once on each machine.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    completed_steps = 0
    cosine_embedding_loss_positive = torch.nn.CosineEmbeddingLoss(margin=0)
    kl_loss = nn.KLDivLoss(reduction="sum")
    cosine_embedding_loss_negative = torch.nn.CosineEmbeddingLoss(margin=args.cosine_margin)
    undrifted_model = copy.deepcopy(model)
    undrifted_model  = undrifted_model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    target_positive = torch.ones(args.batch_size).to(device)
    target_positive_eval = torch.ones(args.batch_size).to(device)
    target_negative = target_positive * -1
    target_negative_eval = target_positive * -1
    target_negative_eval = target_negative_eval.to(device)
    target_negative = target_negative.to(device)
    eval_loss = []
    model.eval()

    eval_progress_bar = tqdm(range(num_eval_steps))
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            anchor = model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device))
            negative = model(input_ids=batch['negative_anchor_input_ids'].to(device),attention_mask=batch['negative_anchor_attention_mask'].to(device))
            positive = model(input_ids=batch['positive_anchor_input_ids'].to(device),attention_mask=batch['positive_anchor_attention_mask'].to(device))
            with torch.no_grad():
                original = undrifted_model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device))
            if args.kldiv == False:
                loss_pos = cosine_embedding_loss_positive(anchor.last_hidden_state[:,0], positive.last_hidden_state[:,0],target_positive_eval) * args.temperature
                loss_orig = cosine_embedding_loss_positive(original.last_hidden_state[:,0], positive.last_hidden_state[:,0],target_positive_eval) * args.temperature
                loss_neg = cosine_embedding_loss_negative(anchor.last_hidden_state[:,0], negative.last_hidden_state[:,0],target_positive) * args.temperature
            else:
                input_orig = positive.last_hidden_state[:,0]
                target_orig = anchor.last_hidden_state[:,0]
                inputs = F.log_softmax(input_orig/ args.temperature, dim=-1)
                targets = F.softmax(target_orig/ args.temperature, dim=-1)
                loss_pos = kl_loss(inputs, targets) * (args.temperature ** 2) / (input_orig.numel() / input_orig.shape[-1])
                target = original.last_hidden_state[:,0]
                targets = F.softmax(target_orig/ args.temperature, dim=-1)
                loss_orig = kl_loss(inputs, targets) * (args.temperature ** 2) / (input_orig.numel() / input_orig.shape[-1])
                target = negative.last_hidden_state[:,0]
                targets = F.softmax(target_orig/ args.temperature, dim=-1)
                loss_neg = kl_loss(inputs, targets) * (args.temperature ** 2) / (input_orig.numel() / input_orig.shape[-1])
            loss = (args.positive_temp * loss_pos) + (args.negative_temp * loss_neg) + (args.anchor_temp * loss_orig)
            eval_loss.append(loss.cpu())
            eval_progress_bar.update(1)
    eval_metric = np.mean(eval_loss)
    logger.info(f"initial eval loss: {eval_metric}")
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps))
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            anchor = model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device))
            negative = model(input_ids=batch['negative_anchor_input_ids'].to(device),attention_mask=batch['negative_anchor_attention_mask'].to(device))
            positive = model(input_ids=batch['positive_anchor_input_ids'].to(device),attention_mask=batch['positive_anchor_attention_mask'].to(device))
            with torch.no_grad():
                original = undrifted_model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device))
            if args.kldiv == False:
                loss_pos = cosine_embedding_loss_positive(anchor.last_hidden_state[:,0], positive.last_hidden_state[:,0],target_positive_eval) * args.temperature
                loss_orig = cosine_embedding_loss_positive(original.last_hidden_state[:,0], positive.last_hidden_state[:,0],target_positive_eval) * args.temperature
                loss_neg = cosine_embedding_loss_negative(anchor.last_hidden_state[:,0], negative.last_hidden_state[:,0],target_positive) * args.temperature
            else:
                input_orig = positive.last_hidden_state[:,0]
                target_orig = anchor.last_hidden_state[:,0]
                inputs = F.log_softmax(input_orig/ args.temperature, dim=-1)
                targets = F.softmax(target_orig/ args.temperature, dim=-1)
                loss_pos = kl_loss(inputs, targets) * (args.temperature ** 2) / (input_orig.numel() / input_orig.shape[-1])
                target = original.last_hidden_state[:,0]
                targets = F.softmax(target_orig/ args.temperature, dim=-1)
                loss_orig = kl_loss(inputs, targets) * (args.temperature ** 2) / (input_orig.numel() / input_orig.shape[-1])
                target = negative.last_hidden_state[:,0]
                targets = F.softmax(target_orig/ args.temperature, dim=-1)
                loss_neg = kl_loss(inputs, targets) * (args.temperature ** 2) / (input_orig.numel() / input_orig.shape[-1])

            loss = (args.positive_temp * loss_pos) + (args.negative_temp * loss_neg) + (args.anchor_temp * loss_orig)
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            if step % 100 == 0:
                logger.info(f"Loss : {loss.item()}")
            if isinstance(args.checkpointing_steps, int):
                if completed_steps % args.checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    if isinstance(model,type(nn.DataParallel(None))):
                        model.module.save_pretrained(output_dir)
                    else:
                        model.save_pretrained(output_dir)
            if completed_steps >= args.max_train_steps:
                break
        eval_progress_bar = tqdm(range(num_eval_steps))
        model.eval()
        eval_loss = []
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                anchor = model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device))
                negative = model(input_ids=batch['negative_anchor_input_ids'].to(device),attention_mask=batch['negative_anchor_attention_mask'].to(device))
                positive = model(input_ids=batch['positive_anchor_input_ids'].to(device),attention_mask=batch['positive_anchor_attention_mask'].to(device))
                with torch.no_grad():
                    original = undrifted_model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device))
                if args.kldiv == False:
                    loss_pos = cosine_embedding_loss_positive(anchor.last_hidden_state[:,0], positive.last_hidden_state[:,0],target_positive_eval) * args.temperature
                    loss_orig = cosine_embedding_loss_positive(original.last_hidden_state[:,0], positive.last_hidden_state[:,0],target_positive_eval) * args.temperature
                    loss_neg = cosine_embedding_loss_negative(anchor.last_hidden_state[:,0], negative.last_hidden_state[:,0],target_positive) * args.temperature
                else:
                    input_orig = positive.last_hidden_state[:,0]
                    target_orig = anchor.last_hidden_state[:,0]
                    inputs = F.log_softmax(input_orig/ args.temperature, dim=-1)
                    targets = F.softmax(target_orig/ args.temperature, dim=-1)
                    loss_pos = kl_loss(inputs, targets) * (args.temperature ** 2) / (input_orig.numel() / input_orig.shape[-1])
                    target = original.last_hidden_state[:,0]
                    targets = F.softmax(target_orig/ args.temperature, dim=-1)
                    loss_orig = kl_loss(inputs, targets) * (args.temperature ** 2) / (input_orig.numel() / input_orig.shape[-1])
                    target = negative.last_hidden_state[:,0]
                    targets = F.softmax(target_orig/ args.temperature, dim=-1)
                    loss_neg = kl_loss(inputs, targets) * (args.temperature ** 2) / (input_orig.numel() / input_orig.shape[-1])
                loss = (args.positive_temp * loss_pos) + (args.negative_temp * loss_neg) + (args.anchor_temp * loss_orig)
                eval_loss.append(loss.cpu())
                eval_progress_bar.update(1)
        eval_metric = np.mean(eval_loss)
        logger.info(f"epoch {epoch}: {eval_metric}")
    
    output_dir = "final"
    output_dir = os.path.join(args.output_dir, output_dir)    
    with open(output_dir + '/eval_results.txt','w') as w:
        w.write("Eval Loss :{}".format(eval_metric))
    if isinstance(model,type(nn.DataParallel(None))):
        model.module.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)
if __name__ == "__main__":
    main()
