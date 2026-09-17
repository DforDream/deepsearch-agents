from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())


llm = init_chat_model(model="qwen-max", model_provider="openai")

current_dir = Path(__file__).parent.resolve()
file_backend = FilesystemBackend(
    root_dir=current_dir,
    virtual_mode=True,
)

main_agent = create_deep_agent(
    model=llm,
    backend=file_backend,
    skills=[
        "skills",
    ],
    system_prompt="你是一个智能助手，可以使用 SKILL 技能",
)

query = "我早上起床晚了，赶公交车差点摔倒，还好最后到了公司。请你只用表情翻译技能。"
result = main_agent.invoke({"messages": [{"role": "user", "content": query}]})

print(f"最终输出结果：{result['messages'][-1].content}")