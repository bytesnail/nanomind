#!/usr/bin/env python3
"""
Nanomind 预训练脚本
支持 DeepSpeed、Accelerate 和混合精度训练
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import load_dataset
import wandb

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: dict):
    """初始化模型和 tokenizer"""
    model_config = config["model"]
    tokenizer_config = config["tokenizer"]

    # 加载或创建 tokenizer
    tokenizer_path = tokenizer_config["name_or_path"]
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"加载 tokenizer: {tokenizer_path}")
    else:
        raise ValueError(f"Tokenizer 路径不存在: {tokenizer_path}")

    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 创建模型配置
    model_kwargs = {k: v for k, v in model_config.items()}
    model_config_obj = AutoConfig.for_model("llama", **model_kwargs)

    # 创建模型
    model = AutoModelForCausalLM.from_config(model_config_obj)
    model.resize_token_embeddings(len(tokenizer))

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"模型总参数量: {total_params / 1e6:.2f}M")
    logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    return model, tokenizer


def prepare_dataset(config: dict, tokenizer):
    """准备数据集"""
    data_config = config["data"]

    # 加载数据集
    if os.path.exists(data_config["train_file"]):
        dataset = load_dataset(
            "json",
            data_files={
                "train": data_config["train_file"],
            },
        )

        if os.path.exists(data_config["validation_file"]):
            dataset["validation"] = load_dataset(
                "json", data_files={"validation": data_config["validation_file"]}
            )["validation"]
        else:
            # 从训练集分割验证集
            split_dataset = dataset["train"].train_test_split(
                test_size=data_config.get("validation_split_percentage", 1) / 100,
                seed=42,
            )
            dataset["train"] = split_dataset["train"]
            dataset["validation"] = split_dataset["test"]
    else:
        raise ValueError(f"训练文件不存在: {data_config['train_file']}")

    max_seq_length = min(
        data_config.get("max_seq_length", 2048), tokenizer.model_max_length
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    # Tokenize 数据集
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_config.get("preprocessing_num_workers", 4),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    return tokenized_datasets


def get_training_arguments(config: dict) -> TrainingArguments:
    """创建 TrainingArguments"""
    training_config = config["training"]

    # 确定混合精度设置
    fp16 = training_config.get("fp16", False)
    bf16 = training_config.get("bf16", False)

    # 自动检测 BF16 支持
    if bf16 == "auto":
        bf16 = torch.cuda.is_bf16_supported()
        fp16 = not bf16

    return TrainingArguments(
        output_dir=training_config["output_dir"],
        overwrite_output_dir=True,
        # 训练步数
        max_steps=training_config.get("max_steps", -1),
        num_train_epochs=training_config.get("num_train_epochs", 1),
        # Batch size
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config.get(
            "per_device_eval_batch_size", training_config["per_device_train_batch_size"]
        ),
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        # 优化器
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.01),
        adam_beta1=training_config.get("adam_beta1", 0.9),
        adam_beta2=training_config.get("adam_beta2", 0.999),
        adam_epsilon=training_config.get("adam_epsilon", 1e-8),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        # 学习率调度
        warmup_ratio=training_config.get("warmup_ratio", 0.01),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        # 日志与保存
        logging_steps=training_config.get("logging_steps", 100),
        save_steps=training_config.get("save_steps", 1000),
        eval_steps=training_config.get("eval_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        logging_first_step=training_config.get("logging_first_step", True),
        # 评估
        evaluation_strategy=training_config.get("evaluation_strategy", "steps"),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        # 混合精度
        fp16=fp16,
        bf16=bf16,
        # DeepSpeed
        deepspeed=training_config.get("deepspeed"),
        # 梯度检查点
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        # WandB
        report_to=training_config.get("report_to", "wandb"),
        run_name=training_config.get("run_name", None),
        # 数据加载
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        dataloader_pin_memory=training_config.get("dataloader_pin_memory", True),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        # 其他
        seed=42,
        data_seed=42,
        ddp_find_unused_parameters=False,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Nanomind 预训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed 配置文件路径",
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置随机种子
    set_seed(42)

    # 初始化 wandb (如果启用)
    if config["training"].get("report_to") == "wandb":
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "nanomind-pretraining"),
            name=config["training"].get("run_name"),
            config=config,
        )

    # 初始化模型和 tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # 准备数据集
    tokenized_datasets = prepare_dataset(config, tokenizer)

    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言建模
    )

    # 创建训练参数
    training_args = get_training_arguments(config)

    # 覆盖 DeepSpeed 配置 (如果命令行提供)
    if args.deepspeed:
        training_args.deepspeed = args.deepspeed

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 保存最终模型
    trainer.save_model(os.path.join(training_args.output_dir, "final"))

    # 结束 wandb
    if config["training"].get("report_to") == "wandb":
        wandb.finish()

    logger.info("训练完成!")


if __name__ == "__main__":
    main()
