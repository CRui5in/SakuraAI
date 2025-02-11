import time
import gradio as gr
from typing import List, Tuple, Optional

import uvicorn
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from agent import TravelAgent


class ChatState:
    def __init__(self):
        self.chats = [[]]
        self.current_chat_index = 0
        self.summaries = ["新对话"]
        self.dialogue_lengths = [0]


class WebTravelAgent(TravelAgent):
    def __init__(self):
        super().__init__()
        self.chat_history: List[Tuple[Optional[str], Optional[str]]] = []

    def process_web_input(self, user_input: str, chat_state: ChatState) -> List[Tuple[str, str]]:
        if chat_state.current_chat_index >= 0:
            self.set_conversation_id(chat_state.current_chat_index)
            chat_state.chats[chat_state.current_chat_index].append((user_input, None))
            return chat_state.chats[chat_state.current_chat_index]
        return []

    def clear_history(self, chat_state: ChatState):
        if chat_state.current_chat_index >= 0:
            self.set_conversation_id(chat_state.current_chat_index)
            chat_state.chats[chat_state.current_chat_index] = []
            chat_state.dialogue_lengths[chat_state.current_chat_index] = 0
            self.reset()
        return []

    def get_chat_summary(self, chat_history: List[Tuple[str, str]]) -> str:
        """获取对话摘要"""
        if not chat_history:
            return "新对话"

        first_msg = next((msg for msg, _ in chat_history if msg), "新对话")
        prompt = f"""请用不超过8个字简要总结用户的请求内容，并且只需要输出总结内容，最后不需要加句号。
                    示例：
                    输入：我想去北京旅游。
                    输出：北京旅游规划
                    用户输入：{first_msg}"""

        try:
            response = self.model_manager.generate_qwen_response(prompt)
            return response.strip()[:10]
        except:
            return first_msg[:10]

    def add_assistant_message(self, message: str, chat_state: ChatState) -> List[Tuple[str, str]]:
        """添加助手消息到对话历史"""
        if chat_state.current_chat_index >= 0:
            chat_state.chats[chat_state.current_chat_index].append((None, message))
            return chat_state.chats[chat_state.current_chat_index]
        return []

def create_iframe_html():
    return """
    <iframe src="/app.html" style="width:100%; height:100vh; border:none;"></iframe>
    """

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="智能旅游助手") as demo:
        gr.HTML(create_iframe_html())
        global chat_btns
        chat_btns = []

        with gr.Sidebar(open=False):
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown("# Sakura🌸AI")
                with gr.Column(scale=2):
                    toggle_dark = gr.Button(value="日间/夜间模式")

            toggle_dark.click(
                fn=None,
                inputs=None,
                outputs=None,
                js="() => { document.body.classList.toggle('dark'); }"
            )

            with gr.Tab("对话历史"):
                for i in range(6):
                    visible = i == 0
                    value = "新对话" if i == 0 else ""
                    chat_btns.append(gr.Button(value=value, visible=visible))
                new_chat_btn = gr.Button("新建对话", variant="primary")

        # 主界面
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            height="60vh",
            show_label=False,
            group_consecutive_messages=False,
            show_copy_button=True,
            avatar_images=(
                "static/user.png",
                "static/bot.png"
            )
        )
        with gr.Row():
            use_finetuned = gr.Checkbox(
                label="模型强化",
                value=True,
            )

        with gr.Row():
            with gr.Column(scale=20):
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="请输入您的旅游问题...",
                    container=False,
                )
            with gr.Column(scale=1, min_width=50):
                submit = gr.Button("🚀", variant="primary")
            with gr.Column(scale=1, min_width=50):
                clear = gr.Button("🗑️", variant="secondary")

        with gr.Row():
            examples = [
                ["我想去重庆玩3天，要去解放碑和洪崖洞"],
                ["我想去北京旅游"],
                ["我想去旅游七天"],
            ]
            gr.Examples(
                examples=examples,
                inputs=msg,
                examples_per_page=3
            )

        def create_new_chat():
            """创建新对话"""
            chat_state.chats.append([])
            chat_state.dialogue_lengths.append(0)
            chat_state.current_chat_index = len(chat_state.chats) - 1
            chat_state.summaries.append("新对话")

            btn_updates = []
            for i in range(6):
                if i < len(chat_state.summaries):
                    btn_updates.append(gr.update(
                        value=chat_state.summaries[i],
                        visible=True
                    ))
                else:
                    btn_updates.append(gr.update(
                        value="新对话",
                        visible=False
                    ))

            return [gr.update(value=[])] + btn_updates

        def switch_chat(index: int):
            if 0 <= index < len(chat_state.chats):
                chat_state.current_chat_index = index
                return chat_state.chats[index]
            return []

        def respond(user_input: str, use_finetuned: bool, chat_history: List[Tuple[str, str]]):
            if not user_input or chat_state.current_chat_index < 0:
                return "", chat_history

            try:
                web_agent.model_manager.use_finetuned = use_finetuned
                chat_history = web_agent.process_web_input(user_input, chat_state)
                yield "", chat_history, *[gr.update()] * 6

                dialogue_history = web_agent.process_input(user_input)

                current_length = chat_state.dialogue_lengths[chat_state.current_chat_index]
                new_responses = dialogue_history[current_length:]

                # 更新对话长度
                chat_state.dialogue_lengths[chat_state.current_chat_index] = len(dialogue_history)

                # 逐条添加回复
                should_generate_summary = False
                for role, content in new_responses:
                    if role == "assistant":
                        time.sleep(0.5)
                        chat_history = web_agent.add_assistant_message(content, chat_state)
                        yield "", chat_history, *[gr.update()] * 6

                        if len(chat_history) == 2:
                            should_generate_summary = True

                if should_generate_summary:
                    summary = web_agent.get_chat_summary(chat_history)
                    chat_state.summaries[chat_state.current_chat_index] = summary
                    updates = [gr.update()] * 6
                    updates[chat_state.current_chat_index] = gr.update(value=summary)
                    yield "", chat_history, *updates
                else:
                    yield "", chat_history, *[gr.update()] * 6

            except Exception as e:
                print(f"Error in respond: {str(e)}")
                if chat_state.current_chat_index >= 0:
                    chat_state.chats[chat_state.current_chat_index].append(
                        (user_input, f"处理消息时发生错误: {str(e)}")
                    )
                    chat_history = chat_state.chats[chat_state.current_chat_index]
                yield "", chat_history, *[gr.update()] * 6

        msg.submit(
            respond,
            [msg, use_finetuned, chatbot],
            [msg, chatbot] + chat_btns,
            queue=True
        )

        submit.click(
            respond,
            [msg, use_finetuned, chatbot],
            [msg, chatbot] + chat_btns,
            queue=True
        )

        clear.click(lambda: web_agent.clear_history(chat_state), None, chatbot)

        new_chat_btn.click(
            create_new_chat,
            None,
            [chatbot] + chat_btns
        )

        for i, btn in enumerate(chat_btns):
            btn.click(
                switch_chat,
                inputs=[gr.State(i)],
                outputs=[chatbot]
            )

        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="""
                function() {
                    const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                    if (isDarkMode && document.body.classList.contains('dark')) {
                        document.body.classList.toggle('dark');
                    }
                    window.scrollTo(0, 0);
                }
            """
        )

    return demo


if __name__ == "__main__":
    web_agent = WebTravelAgent()
    chat_state = ChatState()
    demo = build_ui()

    app = FastAPI()

    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/app.html")
    async def serve_scroll_demo():
        return FileResponse("app.html")

    app = gr.mount_gradio_app(app, demo, path="/", show_api=False, favicon_path="favicon.ico")

    uvicorn.run(app, host="127.0.0.1", port=7879, ssl_keyfile=None, ssl_certfile=None)