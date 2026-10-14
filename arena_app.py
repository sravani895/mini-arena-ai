import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
groq_client   = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL_A = "openai/gpt-oss-120b"
# Prompt Guard (llama-prompt-guard-2-22m) is NOT a chat model — it only
# labels text as benign/malicious. Use a real LLM so Model B returns prose.
MODEL_B = "qwen/qwen3.8-27b"


def message_text(msg):
    """Return the user-visible answer, not tool-call parameters / reasoning blobs."""
    if msg.content:
        return msg.content
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if reasoning:
        return reasoning
    if msg.tool_calls:
        names = [tc.function.name for tc in msg.tool_calls]
        return f"(Model tried to call tools instead of answering: {', '.join(names)})"
    return "(empty response)"


def ask(client, model, prompt):
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tool_choice="none",
    )
    # Groq-only flag: the OpenAI SDK rejects include_reasoning as a kwarg.
    if model.startswith("openai/gpt-oss"):
        kwargs["extra_body"] = {"include_reasoning": False}
    r = client.chat.completions.create(**kwargs)
    return message_text(r.choices[0].message)


def battle(prompt):
    if not (prompt or "").strip():
        return "Type a prompt first.", "Type a prompt first."
    try:
        a = ask(openai_client, MODEL_A, prompt)
    except Exception as e:
        a = f"**Model A error:** {e}"
    try:
        b = ask(groq_client, MODEL_B, prompt)
    except Exception as e:
        b = f"**Model B error:** {e}"
    return a, b

def vote(label):
    return f"🗳️ Thanks! You voted: **{label}**"   # in real apps, save this to a file/DB

with gr.Blocks(title="LLM Arena") as demo:
    gr.Markdown("# 🥊 LLM Arena — one prompt, two models")
    prompt = gr.Textbox(label="Ask both models the same thing")
    go = gr.Button("⚔️ Battle!", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"### 🤖 Model A — `{MODEL_A}`")
            out_a = gr.Markdown()
            with gr.Row():
                up_a   = gr.Button("👍");  down_a = gr.Button("👎")
        with gr.Column():
            gr.Markdown(f"### 🤖 Model B — `{MODEL_B}`")
            out_b = gr.Markdown()
            with gr.Row():
                up_b   = gr.Button("👍");  down_b = gr.Button("👎")

    verdict = gr.Markdown()

    go.click(battle, inputs=prompt, outputs=[out_a, out_b])
    up_a.click(lambda: vote("👍 Model A"), outputs=verdict)
    down_a.click(lambda: vote("👎 Model A"), outputs=verdict)
    up_b.click(lambda: vote("👍 Model B"), outputs=verdict)
    down_b.click(lambda: vote("👎 Model B"), outputs=verdict)

demo.launch(share=True)