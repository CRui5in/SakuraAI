import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class ModelHandler:
    def __init__(self):
        # 初始化分类和分析模型
        self.classifier_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1-distill-qwen-1.5b")
        self.classifier_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-r1-distill-qwen-1.5b")

        # 初始化行程生成模型
        self.planner_tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-7B-Instruct")
        self.planner_tokenizer.pad_token = self.planner_tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            "./Qwen2.5-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.planner_model = PeftModel.from_pretrained(base_model, "./lora-weights")
        self.planner_model.eval()

    def get_task_classification(self, message: str) -> str:
        """使用DeepSeek模型进行任务分类"""
        prompt = f"判断以下用户输入是否是旅游规划相关的请求，若是则回复'requirement_analysis'，不是则回复'general_query'。\n用户输入: {message}"

        inputs = self.classifier_tokenizer(prompt, return_tensors="pt")
        outputs = self.classifier_model.generate(**inputs, max_new_tokens=50)
        response = self.classifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.strip()

    def get_requirements_analysis(self, message: str) -> str:
        """使用DeepSeek模型进行需求分析"""
        prompt = f"""分析以下用户的旅游需求，提取以下信息：
        - 目的地(destination)
        - 旅行天数(duration)
        - 特定景点要求(attractions)
        - 住宿偏好(hotel_preference)
        - 交通偏好(transport_preference)
        按JSON格式返回结果。

        用户输入: {message}"""

        inputs = self.classifier_tokenizer(prompt, return_tensors="pt")
        outputs = self.classifier_model.generate(**inputs, max_new_tokens=200)
        response = self.classifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.strip()

    def get_travel_plan(self, requirements: dict, attractions: List[str]) -> str:
        """使用Qwen模型生成旅游计划"""
        prompt = f"""根据以下信息生成详细的旅游计划：
        目的地：{requirements.get('destination', '未指定')}
        天数：{requirements.get('duration', '未指定')}
        {'推荐景点：' + '、'.join(attractions) if attractions else ''}
        住宿偏好：{requirements.get('hotel_preference', '未指定')}
        交通偏好：{requirements.get('transport_preference', '未指定')}

        请生成每天的具体行程安排。"""

        inputs = self.planner_tokenizer(prompt, return_tensors="pt")
        outputs = self.planner_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
        response = self.planner_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.strip()