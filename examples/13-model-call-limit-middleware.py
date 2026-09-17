import os
from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(find_dotenv())

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai",
)

@tool
def delete_database(table_name: str):
    """
    删除指定数据库表

    :param table_name: 要删除的表名
    :return: 操作的返回结果
    """
    print(f"调用了删除了delete_database工具。删除了{table_name}表！！")
    return f"删除了表{table_name}！"


@tool
def delete_file(file_name: str):
    """
    删除指定文件

    :param file_name: 要删除的文件名
    :return: 操作的返回结果
    """
    print(f"调用了删除了delete_file工具。删除了{file_name}文件")
    return f"删除了文件{file_name}！"


@tool
def select_database(table_name: str):
    """
    查询指定数据库表

    :param table_name: 要查询的表名
    :return: 查询结果
    """
    print(f"调用了select_database工具。查询了{table_name}表数据！")
    return f"查询了表{table_name}的数据！"

# checkpointer 用来记录同一条线程的执行状态
# thread_id 也是 thread_limit 判断“同一个会话线程”的依据
checkpointer = InMemorySaver()
thread_config = {"configurable": {"thread_id": "erdaye"}}

main_agent = create_deep_agent(
    model=llm,
    tools=[delete_database, delete_file, select_database],
    checkpointer=checkpointer,
    system_prompt="回答使用中文，调用对应的工具实现对应的功能",
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=1,  # 同一个 thread_id 下累计最多调用 1 次模型
            run_limit=1,  # 当前这次 invoke 内最多调用 1 次模型
            exit_behavior="error",  # 超限后抛出异常，便于后端统一捕获处理
        )
    ],
    # 本章重点是 middleware 调用限制，因此这里关闭人工审批拦截
    # 如果要演示危险动作审批，可以把高风险工具配置为 True 或 allowed_decisions
    interrupt_on={
        "delete_database": False,
        "delete_file": False,
        "select_database": False,
    },
)

result_1 = main_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "先查询product表的数据！再删除user表，最后，删除zhaoweifeng.txt文件",
            }
        ]
    },
    config=thread_config,
)

print(f"最终结果{result_1['messages'][-1].content}")