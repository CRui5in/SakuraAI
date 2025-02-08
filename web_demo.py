import gradio as gr
from typing import List, Tuple
from agent import TravelAgent
import time


class WebTravelAgent(TravelAgent):
    def __init__(self):
        super().__init__()
        self.chat_history: List[Tuple[str, str]] = []

    def process_web_input(self, user_input: str) -> List[Tuple[str, str]]:
        # 处理用户输入
        dialogue_history = self.process_input(user_input)

        # 将用户输入添加到历史记录
        self.chat_history.append((user_input, None))
        return self.chat_history

    def add_assistant_message(self, message: str) -> List[Tuple[str, str]]:
        """添加助手消息到历史记录"""
        if self.chat_history and self.chat_history[-1][1] is None:
            self.chat_history[-1] = (self.chat_history[-1][0], message)
        else:
            self.chat_history.append((None, message))
        return self.chat_history

    def clear_history(self):
        """清除对话历史并重置代理状态"""
        self.chat_history = []
        self.reset()
        return []


# 创建代理实例
web_agent = WebTravelAgent()

# 创建Gradio界面
with gr.Blocks(title="智能旅游助手") as demo:
    gr.Markdown("""
    # 智能旅游助手
    欢迎使用智能旅游助手！我可以帮您:
    - 制定旅游计划
    - 推荐景点和路线
    - 回答旅游相关问题
    """)

    chatbot = gr.Chatbot(
        label="对话历史",
        height=600,
        show_label=True,
    )

    with gr.Row():
        msg = gr.Textbox(
            label="输入消息",
            placeholder="请输入您的旅游需求或问题...",
            lines=2,
            show_label=True,
        )
        submit = gr.Button("发送", variant="primary")

    with gr.Row():
        clear = gr.Button("清除对话")


    def respond(user_input: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        if not user_input:
            return "", chat_history
        try:
            # 首先添加用户输入并更新界面
            new_history = web_agent.process_web_input(user_input)
            yield "", new_history

            # 获取AI响应
            responses = web_agent.get_responses(user_input)  # 假设这个方法返回AI响应列表

            # 逐条发送AI响应
            for response in responses:
                time.sleep(0.5)  # 添加短暂延迟使消息显示更自然
                new_history = web_agent.add_assistant_message(response)
                yield "", new_history

        except Exception as e:
            error_message = f"处理消息时发生错误: {str(e)}"
            if chat_history:
                chat_history.append((user_input, error_message))
            else:
                chat_history = [(user_input, error_message)]
            yield "", chat_history


    # 事件处理
    msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=True)
    submit.click(respond, [msg, chatbot], [msg, chatbot], queue=True)
    clear.click(web_agent.clear_history, None, chatbot)

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False,
        debug=True
    )