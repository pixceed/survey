import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI

# .envファイルから環境変数を読み込み
load_dotenv()

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

# tools = [add, multiply, divide]
# llm = AzureChatOpenAI(
#         openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#         azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
#         temperature=0
#     )

# llm_with_tools = llm.bind_tools(tools)