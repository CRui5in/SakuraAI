import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 1. 从本地加载模型和分词器
model_path = "./Qwen2.5-7B-Instruct"  # 本地模型路径
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 减少显存占用
    low_cpu_mem_usage=True  # 减少 CPU 到 GPU 的内存拷贝
)
tokenizer = AutoTokenizer.from_pretrained(model_path)  # 从本地加载分词器
tokenizer.pad_token = tokenizer.eos_token  # 设置padding token

# 2. 加载数据集
dataset = load_dataset('json', data_files='data/processed_data/converted_data.jsonl')  # 加载数据集

# 3. 数据预处理
def preprocess_function(examples):
    """
    数据预处理函数：将多轮对话转换为模型训练格式，并进行tokenize。
    """
    # 将多轮对话转换为模型训练格式
    formatted_texts = []
    for messages in examples['messages']:
        formatted_text = ""
        for msg in messages:
            if msg['role'] == 'user':
                formatted_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'assistant':
                formatted_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        formatted_texts.append(formatted_text)

    # 对整个对话进行编码
    model_inputs = tokenizer(
        formatted_texts,
        truncation=True,  # 截断超过 max_length 的文本
        padding='max_length',  # 填充到 max_length
        max_length=1024,  # 最大长度
        return_tensors="pt"  # 返回 PyTorch 张量
    )

    # 创建训练标签
    model_inputs['labels'] = model_inputs['input_ids'].clone()
    # 将padding位置的标签设为-100，使其在计算loss时被忽略
    pad_mask = model_inputs['attention_mask'] == 0
    model_inputs['labels'][pad_mask] = -100

    return model_inputs

# 对数据集进行预处理
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

# 4. 配置 LoRA
lora_config = LoraConfig(
    r=8,  # LoRA 的秩
    lora_alpha=32,  # LoRA 的缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的目标模块
    lora_dropout=0.1,  # Dropout 概率
    bias="none",  # 是否训练偏置参数
    task_type="CAUSAL_LM"  # 任务类型
)

# 将 LoRA 应用到模型
model = get_peft_model(model, lora_config).to("cuda")  # 将模型移动到 GPU

# 5. 启用梯度检查点并禁用 use_cache
model.gradient_checkpointing_enable()
model.config.use_cache = False  # 禁用 use_cache 以减少显存占用

# 6. 启用输入梯度需求
model.enable_input_require_grads()

# 7. 准备 DataCollator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True  # 动态填充批次数据
)

# 8. 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    per_device_train_batch_size=2,  # 每个设备的批次大小
    num_train_epochs=3,             # 训练轮数
    logging_dir="./logs",           # TensorBoard 日志目录
    logging_steps=10,               # 每 10 步记录一次日志
    save_steps=500,                 # 每 500 步保存一次模型
    eval_strategy="no",             # 不进行评估（如果没有验证集）
    learning_rate=5e-5,             # 学习率
    fp16=True,                      # 启用混合精度训练（fp16）
    gradient_accumulation_steps=4,  # 梯度累积步数
    save_total_limit=2,             # 最多保存 2 个检查点
    report_to="tensorboard"         # 使用 TensorBoard 记录日志
)

# 9. 实例化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 10. 开始训练
trainer.train()

# 11. 保存微调后的模型
model.save_pretrained("./lora-weights")  # 保存 LoRA 权重