import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv(find_dotenv())

llm_name = os.getenv("LLM_QWEN_MAX")
tavily_key = os.getenv("TAVILY_API_KEY")

# Tavily 客户端负责真正的联网搜索，工具函数中会复用这个客户端
tavily_client = TavilyClient(api_key=tavily_key)

@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["news", "finance", "general"] = "general",
    include_raw_content: bool = False,
):
    """
    互联网搜索工具

    DeepAgent 会根据工具描述和参数签名，自动决定是否调用该工具
    :param query: 搜索关键词
    :param max_results: 返回结果数量
    :param topic: 查询主题，可选 news、finance、general
    :param include_raw_content: 是否返回更详细的原文内容，include_raw_content=False 时返回摘要内容；True 时会尝试返回更完整的网页原文
    :return: Tavily 搜索结果
    """
    print(
        f"开始调用网络搜索工具，核心参数为：{query},{max_results},{topic},{include_raw_content}"
    )
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )

llm = init_chat_model(model=llm_name, model_provider="openai")

deep_agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    subagents=[],
    system_prompt="""
    你是一名严谨的研究员，可以使用 internet_search 工具检索网络信息。
    请根据检索结果进行归纳、分析和交叉验证，生成一份结构清晰、信息可靠的中文报告。
    """,
)

result = deep_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请查询人工智能和机器人领域的热门新闻信息，并整理为一份简要报告。",
            }
        ]
    }
)

print(result)

print(result["messages"][-1].content)
