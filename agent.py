import operator
import traceback
from typing import List, Dict, Optional, Any, Annotated
import torch
from langgraph.constants import START
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing_extensions import TypedDict
import json
import googlemaps

from tool import extract_json_block, merge_dicts

REDUCER_METADATA = {"reducer": operator.add}

class TravelRequirements(TypedDict):
    destination: List[str]
    duration: Optional[int]
    attractions: List[str]
    hotel_preference: List[str]
    transport_preference: List[str]

class AgentState(TypedDict):
    messages: List[Any]
    current_step: str
    travel_requirements: Dict
    generated_plan: Optional[str]
    status: str

    @classmethod
    def create_initial(cls, input_message: str) -> Dict:
        return {
            "messages": [HumanMessage(content=input_message)],
            "current_step": "start",
            "travel_requirements": {
                "destination": [],
                "duration": None,
                "attractions": [],
                "hotel_preference": [],
                "transport_preference": []
            },
            "generated_plan": None,
            "status": "active"
        }

class ModelManager:
    def __init__(self):
        # 加载DeepSeek模型用于分类和需求分析
        # self.deepseek_tokenizer = AutoTokenizer.from_pretrained("./DeepSeek-R1-Distill-Qwen-7B")
        # self.deepseek_model = AutoModelForCausalLM.from_pretrained(
        #     "./DeepSeek-R1-Distill-Qwen-7B",
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto"
        # )

        # 加载Qwen模型用于计划生成
        self.qwen_tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-7B-Instruct")
        self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            "./Qwen2.5-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.qwen_model = PeftModel.from_pretrained(base_model, "./lora-weights")
        self.qwen_model.eval()

    def generate_deepseek_response(self, prompt: str) -> str:
        print("Input prompt:", prompt)
        inputs = self.deepseek_tokenizer(prompt, return_tensors="pt").to(self.deepseek_model.device)
        with torch.no_grad():
            outputs = self.deepseek_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        response = self.deepseek_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        print("Decoded output:", response)

        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        return response

    def generate_qwen_response(self, message: str, history: List = None) -> str:
        if history is None:
            history = []

        full_prompt = ""
        for hist in history:
            full_prompt += f"<|im_start|>user\n{hist[0]}<|im_end|>\n<|im_start|>assistant\n{hist[1]}<|im_end|>\n"
        full_prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.qwen_tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.qwen_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.qwen_tokenizer.pad_token_id,
                eos_token_id=self.qwen_tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
            )

        response = self.qwen_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        response = response.split("<|im_end|>")[0].strip()
        return response

class TravelKnowledgeBase:
    """空的知识库类，用于占位"""
    def get_attractions(self, destination: str) -> str:
        return "未配置景点信息"


class TravelAgentNodes:
    def __init__(self, knowledge_base: 'TravelKnowledgeBase', gmaps_client: Optional['googlemaps.Client']):
        self.model_manager = ModelManager()
        self.knowledge_base = knowledge_base
        self.gmaps_client = gmaps_client
        self.use_gmaps = gmaps_client is not None and hasattr(gmaps_client,
                                                              'key') and gmaps_client.key != 'YOUR_GOOGLE_MAPS_API_KEY'
        self.use_knowledge_base = knowledge_base is not None and not isinstance(knowledge_base, type)

    def task_classifier(self, state: Dict) -> Dict:
        print("===== Starting task classifier =====")
        user_message = state["messages"][-1].content

        prompt = f"""作为旅游助手，判断以下输入是否是旅游规划相关的请求，并从以下两种回复中选择：
        如果是旅游规划请求或者请求中包含国家/地区/城市/地点/景点名字，只需要回复一个'是'字，
        如果不是则只回复一个'否'字。

        用户输入：{user_message}"""

        response = self.model_manager.generate_qwen_response(prompt)
        print("Raw classifier response:", response)

        next_step = 'requirement_analysis' if response.strip() == "是" else 'general_query'
        classification_result = f"根据分析，这是一个{'旅游规划' if next_step == 'requirement_analysis' else '普通查询'}请求。"

        new_messages = state["messages"] + [AIMessage(content=classification_result)]

        return {
            "messages": new_messages,
            "current_step": next_step,
            "travel_requirements": state["travel_requirements"].copy(),
            "generated_plan": state["generated_plan"],
            "status": state["status"]
        }

    def requirement_analyzer(self, state: Dict) -> Dict:
        print("=== Starting requirement analyze ===")
        user_message = next((msg.content for msg in reversed(state['messages']) if isinstance(msg, HumanMessage)), None)
        print("Raw classifier response:", user_message)

        prompt = f"""
        注意事项：
        1. 需要提取的信息的字段及其定义：
           destination（目的地）：国家/城市/地区名称，用列表格式。
           duration（天数）：天数（中文数字需转为阿拉伯数字）。
           attractions（景点）：景点或地点名称，用列表格式。
           hotel_preference（住宿偏好）：住宿要求，用列表格式。
           transport_preference（交通偏好）：交通方式，用列表格式。
           示例输入格式：我想去重庆玩3天，要去解放碑和洪崖洞，住宿要求五星级豪华酒店。
           示例输出格式：
           {{
              "destination": ["重庆"],
              "duration": 3,
              "attractions": ["解放碑", "洪崖洞"],
              "hotel_preference": ["五星级豪华酒店"],
              "transport_preference": []
            }}
        2. 字段提取条件：
           只要是字段定义相关的信息都可以进行提取，不管是否和旅游有关。
           只有未明确提到的字段需要填充默认值，禁止填充推测值。
        3. 严格禁止行为：
           禁止推测用户意图。
           禁止补充未提及的信息。
           禁止填充推测值。
           禁止联想相关场景。
           禁止输出除JSON数据以外语句。
        
        你是一个从用户输入中提取信息的助手，需要从用户输入的信息整合成满足注意事项的JSON格式数据进行下一步处理。
        
        用户输入：{user_message}
        """

        response = self.model_manager.generate_qwen_response(prompt)
        print("Analyzer response:", response)
        response = extract_json_block(response)

        new_requirements = json.loads(response)
        requirements = state["travel_requirements"].copy()
        try:
            new_requirements = merge_dicts(requirements, new_requirements)
            next_step = "ask_destination" if not new_requirements.get("destination") else (
                "route_planning" if self.use_gmaps else "plan_generation"
            )
            analysis_result = f"""我已经分析了您的需求：
                                目的地：{new_requirements.get('destination')}
                                天数：{new_requirements.get('duration')}
                                推荐景点：{new_requirements.get('attractions')}
                                住宿偏好：{new_requirements.get('hotel_preference')}
                                交通偏好：{new_requirements.get('transport_preference')}"""
        except json.JSONDecodeError:
            next_step = "error_handling"
            analysis_result = "抱歉，在分析您的需求时遇到了问题。"

        new_messages = state["messages"] + [AIMessage(content=analysis_result)]

        return {
            "messages": new_messages,
            "current_step": next_step,
            "travel_requirements": new_requirements,
            "generated_plan": state["generated_plan"],
            "status": state["status"]
        }

    def destination_asker(self, state: Dict) -> Dict:
        print("==== Starting destination asker ====")
        question = "我注意到您没有指定目的地，请问您想去哪个地方旅游呢？"
        new_messages = state["messages"] + [AIMessage(content=question)]

        return {
            "messages": new_messages,
            "current_step": "wait_destination",
            "travel_requirements": state["travel_requirements"].copy(),
            "generated_plan": state["generated_plan"],
            "status": "waiting_input"
        }

    def wait_destination_handler(self, state: Dict) -> Dict:
        print("=== Processing destination input ===")
        if state["status"] == "waiting_input":
            return state

        user_input = state["messages"][-1].content
        new_requirements = state["travel_requirements"].copy()
        new_requirements["destination"] = user_input

        confirmation = f"好的，已经记录您想去 {user_input} 旅游。让我继续分析其他需求。"
        new_messages = state["messages"] + [AIMessage(content=confirmation)]

        return {
            "messages": new_messages,
            "current_step": "requirement_analysis",
            "travel_requirements": new_requirements,
            "generated_plan": state["generated_plan"],
            "status": "active"
        }

    def plan_generator(self, state: Dict) -> Dict:
        print("===== Starting plan generator ======")
        requirements = state["travel_requirements"]
        attractions = (self.knowledge_base.get_attractions(requirements["destination"])
                       if self.use_knowledge_base else "未指定景点")

        prompt = f"""请根据以下信息生成详细的旅游计划：
        目的地：{requirements.get('destination')}
        天数：{requirements.get('duration')}
        推荐景点：{requirements.get('attractions')}
        住宿偏好：{requirements.get('hotel_preference')}
        交通偏好：{requirements.get('transport_preference')}

        请生成每天的具体行程安排。"""

        generated_plan = self.model_manager.generate_qwen_response(prompt)
        new_messages = state["messages"] + [AIMessage(content=generated_plan)]
        next_step = "route_planning" if self.use_gmaps else END

        return {
            "messages": new_messages,
            "current_step": next_step,
            "travel_requirements": requirements,
            "generated_plan": generated_plan,
            "status": state["status"]
        }

    def route_planner(self, state: Dict) -> Dict:
        print("====== Starting route planner ======")
        if not self.use_gmaps:
            return {
                "messages": state["messages"],
                "current_step": END,
                "travel_requirements": state["travel_requirements"].copy(),
                "generated_plan": state["generated_plan"],
                "status": state["status"]
            }

        try:
            directions = self.gmaps_client.directions(
                origin="当前位置",
                destination=state["travel_requirements"]["destination"],
                mode="transit"
            )

            route_info = f"交通路线信息：\n{directions}"
            new_messages = state["messages"] + [AIMessage(content=route_info)]
            next_step = END

        except Exception as e:
            error_message = f"在获取路线信息时遇到错误：{str(e)}"
            new_messages = state["messages"] + [AIMessage(content=error_message)]
            next_step = "error_handling"

        return {
            "messages": new_messages,
            "current_step": next_step,
            "travel_requirements": state["travel_requirements"].copy(),
            "generated_plan": state["generated_plan"],
            "status": state["status"]
        }

    def error_handler(self, state: Dict) -> Dict:
        print("====== Starting error handler ======")
        error_message = "抱歉，处理您的请求时出现了问题。请重新描述您的需求。"
        new_messages = state["messages"] + [AIMessage(content=error_message)]

        return {
            "messages": new_messages,
            "current_step": "task_classifier",
            "travel_requirements": {},
            "generated_plan": None,
            "status": "active"
        }

    def general_query_handler(self, state: Dict) -> Dict:
        print("========Starting general query handler========")
        user_message = state["messages"][-1].content

        prompt = f"""作为旅游助手，请回答用户的问题：

        用户问题：{user_message}"""

        response = self.model_manager.generate_qwen_response(prompt)
        new_messages = state["messages"] + [AIMessage(content=response)]

        return {
            "messages": new_messages,
            "current_step": END,
            "travel_requirements": state["travel_requirements"].copy(),
            "generated_plan": state["generated_plan"],
            "status": state["status"]
        }


class TravelAgent:
    def __init__(self, knowledge_base: Optional[TravelKnowledgeBase] = None,
                 gmaps_api_key: Optional[str] = None):
        self.knowledge_base = knowledge_base or TravelKnowledgeBase()
        self.gmaps_client = None if not gmaps_api_key or gmaps_api_key == 'YOUR_GOOGLE_MAPS_API_KEY' else googlemaps.Client(
            key=gmaps_api_key)
        self.workflow = create_travel_agent_workflow(self.knowledge_base, self.gmaps_client)
        self.current_state = None

    def reset(self):
        """重置代理状态"""
        self.current_state = None

    def process_input(self, user_input: str) -> List[tuple]:
        try:
            if self.current_state is not None and self.current_state["status"] == "waiting_input":
                new_messages = self.current_state["messages"] + [HumanMessage(content=user_input)]

                self.current_state = {
                    "messages": new_messages,
                    "current_step": self.current_state["current_step"],
                    "travel_requirements": self.current_state["travel_requirements"].copy(),
                    "generated_plan": self.current_state["generated_plan"],
                    "status": "active"  # 改变状态为active
                }

            else:
                self.current_state = AgentState.create_initial(user_input)

            final_state = self.workflow.invoke(self.current_state)
            self.current_state = final_state

            dialogue_history = []
            for message in final_state["messages"]:
                if isinstance(message, HumanMessage):
                    dialogue_history.append(("user", message.content))
                elif isinstance(message, AIMessage):
                    dialogue_history.append(("assistant", message.content))

            return dialogue_history

        except Exception as e:
            print(f"Error in process_input: {traceback.format_exc()}")
            return [("user", user_input), ("assistant", f"处理请求时发生错误: {str(e)}")]

def create_travel_agent_workflow(knowledge_base: TravelKnowledgeBase,
                               gmaps_client: Optional['googlemaps.Client']) -> StateGraph:
    """创建旅行代理工作流"""
    workflow = StateGraph(AgentState)
    nodes = TravelAgentNodes(knowledge_base, gmaps_client)

    # 添加所有节点
    workflow.add_node("task_classifier", nodes.task_classifier)
    workflow.add_node("requirement_analysis", nodes.requirement_analyzer)
    workflow.add_node("ask_destination", nodes.destination_asker)
    workflow.add_node("wait_destination", nodes.wait_destination_handler)
    workflow.add_node("plan_generation", nodes.plan_generator)
    workflow.add_node("route_planning", nodes.route_planner)
    workflow.add_node("error_handling", nodes.error_handler)
    workflow.add_node("general_query", nodes.general_query_handler)

    # 配置工作流入口
    workflow.add_edge(START, "task_classifier")

    # task_classifier 的条件边
    workflow.add_conditional_edges(
        "task_classifier",
        lambda x: x["current_step"],  # 修改为使用字典访问
        {
            "requirement_analysis": "requirement_analysis",
            "general_query": "general_query"
        }
    )

    # requirement_analysis 的条件边
    workflow.add_conditional_edges(
        "requirement_analysis",
        lambda x: x["current_step"],  # 修改为使用字典访问
        {
            "ask_destination": "ask_destination",
            "plan_generation": "plan_generation",
            "error_handling": "error_handling"
        }
    )

    # ask_destination 到 wait_destination 的边
    workflow.add_edge("ask_destination", "wait_destination")

    # wait_destination 的条件边
    workflow.add_conditional_edges(
        "wait_destination",
        lambda x: x["status"],  # 修改为使用字典访问
        {
            "waiting_input": END,  # 如果是等待输入状态，结束当前流程
            "active": "requirement_analysis"  # 如果是活动状态，继续需求分析
        }
    )

    # plan_generation 的条件边
    if nodes.use_gmaps:
        workflow.add_conditional_edges(
            "plan_generation",
            lambda x: x["current_step"],  # 修改为使用字典访问
            {
                "route_planning": "route_planning",
                "error_handling": "error_handling"
            }
        )

        # route_planning 的条件边
        workflow.add_conditional_edges(
            "route_planning",
            lambda x: x["current_step"],  # 修改为使用字典访问
            {
                "error_handling": "error_handling",
                END: END
            }
        )
    else:
        workflow.add_edge("plan_generation", END)

    # error_handling 返回到 task_classifier
    workflow.add_edge("error_handling", "task_classifier")

    # general_query 直接结束
    workflow.add_edge("general_query", END)

    return workflow.compile()

def main():
    # 创建旅行代理实例
    basic_agent = TravelAgent()

    print("欢迎使用旅行助手！请输入您的需求，输入'退出'结束对话。")

    while True:
        # 获取用户输入
        user_input = input("\n=========== HUMAN MESSAGE ===========\n")

        # 检查是否退出
        if user_input.lower() in ['退出', 'quit', 'exit']:
            print("\n=========== AI MESSAGE ===========")
            print("感谢使用旅行助手，再见！")
            break

        # 处理用户输入
        try:
            dialogue_history = basic_agent.process_input(user_input)

            # 打印对话历史
            for role, content in dialogue_history:
                if role == "user":
                    print(f"\n=========== HUMAN MESSAGE ===========")
                    print(content)
                elif role == "assistant":
                    if content.startswith("正在使用工具") or content.startswith("工具返回结果"):
                        print(f"\n=========== TOOLS MESSAGE ===========")
                    else:
                        print(f"\n=========== AI MESSAGE ===========")
                    print(content)

        except Exception as e:
            print(f"\n========== AI MESSAGE ==========")
            print(f"发生错误: {str(e)}")
            print(traceback.format_exc())

if __name__ == "__main__":
    main()