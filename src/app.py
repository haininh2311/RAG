import gradio as gr
from answer import QASystem
from config import load_config

config = load_config()
qa_system = QASystem.from_config(config)


def answer_question(question):
    result = qa_system.answer_question(question)
    return result["answer"]


# Giao diện
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Hệ thống Hỏi Đáp")
    question = gr.Textbox(
        label="Câu hỏi", placeholder="Ví dụ: ĐHQGHN được thành lập năm nào?"
    )
    answer = gr.Textbox(label="Câu trả lời", lines=5)
    submit = gr.Button("Hỏi")
    submit.click(fn=answer_question, inputs=question, outputs=answer)

demo.launch()
