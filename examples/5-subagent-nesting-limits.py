import os
from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai",
)

coder_config = {
    "name": "Coder",
    "description": "高级 Python 工程师，负责接收具体的编码任务并实现代码。",
    "system_prompt": """
    你是一名高级 Python 工程师。
    你的职责是接收具体的编码任务，并给出对应的代码实现。
    """,
    "tools": [],  # Coder 拥有默认的文件操作工具
}

cto_config = {
    "name": "CTO",
    "description": "技术总监，负责将战略需求转化为技术任务，并分配给工程师。",
    # system_prompt 可以约束 CTO 的角色，但不能让 dict 自动支持嵌套子智能体
    "system_prompt": """
    你是技术总监。
    你不直接编写代码。
    你的职责包括：
    1. 分析 CEO 的需求。
    2. 设计技术方案。
    3. 调用 Coder 子代理完成具体的代码编写工作。
    """,
    "tools": [],
    # 关键边界：普通 dict 子智能体的标准字段里没有 subagents。
    # 这一行是“反例式演示”，用于观察底层是否忽略该字段，而不是推荐用法。
    "subagents": [coder_config],
}

ceo_agent = create_deep_agent(
    model=llm,
    name="CEO",
    # 顶层只注册 CTO。即便 CTO 配置里写了 subagents，也不代表 Coder 会被自动注册成功。
    system_prompt="""
    你是 CEO，负责公司战略决策。
    你不直接编写代码或操作文件。
    请将所有技术相关的开发任务委派给 CTO 处理。
    你的工作是验收 CTO 提交的结果。
    """,
    subagents=[cto_config],
)

print(">>> 开始执行任务链...")

stream = ceo_agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我开发一个贪吃蛇游戏，要求用 Python 实现，直接提供代码字符串即可。",
            }
        ]
    }
)

# 不推荐写法 子智能体 不能再嵌套 子智能体
print("\n>>> 最终结果：")
for chunk in stream:
    print(chunk)