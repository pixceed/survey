'''
python実行機能があるエージェントを作成
過去の会話の記憶を保持
'''


import os
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import chat_agent_executor
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import PromptTemplate

# .envファイルから環境変数を読み込み
load_dotenv()

# PythonREPLインスタンスの作成
python_repl = PythonREPL()

# Pythonコードをキャプチャし、実行して結果を返す関数
def execute_python_code(code):
    print('#'*80)
    print(f"実行されたPythonコード:\n{code}\n{'-'*80}")
    try:
        result = python_repl.run(code)
        if not result:  # 結果が空の場合のチェック
            result = "結果が返されませんでした。"
        print(f"実行結果:\n{result}\n{'#'*80}")
    except Exception as e:
        result = f"エラーが発生しました: {str(e)}"
        print(result)
    return result

# OpenAIのLLMインスタンス作成
chat_model = ChatOpenAI(
    model="gpt-4",
    temperature=0  # 応答の一貫性を高めるため温度は0に設定
)

# Pythonコードを実行できるツールを作成
python_tool = Tool(
    name="Python_REPL",
    func=execute_python_code,
    description="Pythonコードを実行して結果を表示するツール"
)

# 使用するツールをリスト化
tools = [python_tool]

# Memoryの準備
memory = MemorySaver()

# チャットエージェントの作成
agent_executor = chat_agent_executor.create_tool_calling_executor(
    chat_model,
    tools=tools,
    checkpointer=memory
)

# 質問テンプレートの作成
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    以下の<質問></質問>に回答してください。
    Pythonプログラムを実行する場合は、実行結果をprintで出力してください。

    <質問>
    {question}
    </質問>
    """
)


# コンフィグの準備
config = {"configurable": {"thread_id": "abc123"}}


# 質問1
question = "300かける12は？"
formatted_question = prompt_template.format(question=question)

# エージェントに質問
response = agent_executor.invoke(
    {"messages": [("human", formatted_question)]},
    config=config
    )

# 結果を出力
print()
print("="*80)
print(response["messages"][-1].content)
print("="*80)
print()


# 質問2
question = "さっきの答えを100で割って"
formatted_question = prompt_template.format(question=question)

# エージェントに質問
response = agent_executor.invoke(
    {"messages": [("human", formatted_question)]},
    config=config
    )

# 結果を出力
print()
print("="*80)
print(response["messages"][-1].content)
print("="*80)
print()
