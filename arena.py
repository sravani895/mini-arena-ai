def battle(prompt):
    msgs = [{"role": "user", "content": prompt}]

    a = openai_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=msgs,
        tool_choice="none",
        extra_body={"include_reasoning": False},
    )

    b = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=msgs,
        tool_choice="none",
    )

    return a.choices[0].message.content, b.choices[0].message.content