import datasets
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import transformers
from transformers import DataCollatorWithPadding
from sklearn.metrics import f1_score
import torch
import numpy as np
import os
import torch.nn as nn
from head import MyClassifier

SEED=42
LR=2e-5
BATCH_SIZE=8
WARMUP_RATIO=0.1
INTERVAL=20*8//BATCH_SIZE
MODEL_NAME = "albert-base-v2"
DATASET_NAME = "glue" # 一组NLP评测任务
DATASET_TASK = "mrpc" # MRPC 是其中一个子任务 -- Microsoft Research Paraphrase Corpus

# 加载数据集对应的评估方法
glue_metric = datasets.load_metric(DATASET_NAME, DATASET_TASK)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return glue_metric.compute(predictions=predictions, references=labels)

# 加载数据集
raw_datasets = load_dataset(DATASET_NAME,DATASET_TASK)

# 训练集
raw_train_dataset = raw_datasets["train"]
# 验证集
raw_valid_dataset = raw_datasets["validation"]

columns = raw_train_dataset.column_names

# 设置随机种子
transformers.set_seed(SEED)

# 定义tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 定义数据处理函数，把原始数据转成input_ids, attention_mask, labels
def process_fn(examples):
    inputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=128)
    examples["input_ids"] = inputs["input_ids"]
    examples["attention_mask"] = inputs["attention_mask"]
    examples["labels"] = examples["label"]
    return examples



tokenized_train_dataset = raw_train_dataset.map(
    process_fn,
    batched=True,
    remove_columns=columns
)

tokenized_valid_dataset = raw_valid_dataset.map(
    process_fn,
    batched=True,
    remove_columns=columns
)


# 定义数据校准器（自动生成batch）
collater = DataCollatorWithPadding(
    tokenizer=tokenizer, return_tensors="pt",
)

# 定义模型 -- 其实Transformer可以直接用AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 我手工写了分类器层，为了方便大家理解什么叫在Transformer上面做分类任务
# backbone = AutoModel.from_pretrained(MODEL_NAME)
# model = MyClassifier(backbone)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./output",        # checkpoint保存路径
    evaluation_strategy="steps",    # 每N步做一次eval
    overwrite_output_dir=True,
    num_train_epochs=1,             # 训练epoch数
    per_device_train_batch_size=BATCH_SIZE,  # 每张卡的batch大小
    gradient_accumulation_steps=1,   # 累加几个step做一次参数更新
    per_device_eval_batch_size=BATCH_SIZE,  # evaluation batch size
    logging_steps=INTERVAL,             # 每20步eval一次
    save_steps=INTERVAL,                # 每20步保存一个checkpoint
    learning_rate=LR,             # 学习率
    warmup_ratio=WARMUP_RATIO,               # 预热（可选）
)

# 定义训练器
trainer = Trainer(
    model=model, # 待训练模型
    args=training_args, # 训练参数
    data_collator=collater, # 数据校准器
    train_dataset=tokenized_train_dataset, # 训练集
    eval_dataset=tokenized_valid_dataset, # 验证集
    compute_metrics=compute_metrics, # 评价指标
)

# 禁用wandb（与huggingface.co同步的机制）
os.environ["WANDB_DISABLED"] = "true"

# 开始训练
trainer.train()
