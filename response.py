import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型和tokenizer
base_model_path = "./Qwen2.5-7B-Instruct"
lora_weights_path = "./lora-weights"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, lora_weights_path)
model.eval()

def chat_response(message, history):
    """
    处理用户输入并生成回复
    """
    # 构建对话上下文
    full_prompt = ""
    for hist in history:
        full_prompt += f"<|im_start|>user\n{hist[0]}<|im_end|>\n<|im_start|>assistant\n{hist[1]}<|im_end|>\n"
    full_prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

    # 对输入进行编码
    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        )

    # 解码模型输出
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    # 移除结束标记
    response = response.split("<|im_end|>")[0].strip()

    return response

def agent_response(travel_agent, message, history):
    """
    处理用户输入并生成回复
    """
    try:
        # 首先通过travel agent处理输入
        response = travel_agent.process_input(message)

        # 如果需要，可以将agent的响应再通过原有的模型处理
        full_prompt = ""
        for hist in history:
            full_prompt += f"<|im_start|>user\n{hist[0]}<|im_end|>\n<|im_start|>assistant\n{hist[1]}<|im_end|>\n"
        full_prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"

        # 使用原有的模型生成最终响应
        inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        response = response.split("<|im_end|>")[0].strip()

        return response

    except Exception as e:
        return f"抱歉，处理您的请求时出现错误：{str(e)}"