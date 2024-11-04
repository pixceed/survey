'''
エージェントの作成（LangGraphを使わない）

エージェントの作成には、LangChainとLangGraphどっち使うべき？
・https://qiita.com/YutaroOgawa2/items/cb5b1db9f07a1c4f3f54

'''

from langchain import hub
from dotenv import load_dotenv  # .envファイルからAPIキーなどを読み込むためのライブラリ
from langchain_openai import ChatOpenAI  # OpenAIの言語モデルを使用するためのモジュール

from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain_experimental.utilities import PythonREPL  # Pythonのコードを実行するためのモジュール

# .envファイルから環境変数（APIキーなど）を読み込み
load_dotenv()


# OpenAIのLLMインスタンスを作成
chat_model = ChatOpenAI(
    model="gpt-4o",  # GPT-4モデルを使用
    temperature=0   # 一貫した応答のために温度を0に設定 
)

# Pythonコードを実行するREPL環境を作成
python_repl = PythonREPL()

# Pythonコードを実行して結果を返す関数
def execute_python_code(code):
    try:
        # Pythonコードに必ずprint文を追加する
        code = f"print({code})" if "print" not in code else code
        
        # Pythonコードを実行
        result = python_repl.run(code)
        # 実行結果が空の場合
        if not result:
            result = "結果が返されませんでした。"
        # 実行されたコードと結果を表示
        print("pythonが実行されました。")
        print(f"コード:\n{code}\n\n結果:\n{result}\n\n")
    except Exception as e:
        # エラーハンドリング
        result = f"エラーが発生しました: {str(e)}"
    return result

# Pythonコードを実行できるツールを作成
python_tool = Tool(
    name="Python_REPL",
    func=execute_python_code,  # 実行する関数
    description="Pythonコードを実行して結果を表示するツール"
)

# 作成したツールをリストに追加
tools = [python_tool]


instructions = \
"""
You are an assistant for question-answering tasks. 
"""

base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

# print("-"*80)
# print(base_prompt)
# print("-"*80)
# print(prompt)
# print("-"*80)

# チャットエージェントの作成（システムプロンプトを追加）
agent = create_openai_functions_agent(
    chat_model,
    tools,
    prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)



result = agent_executor.invoke({"input": "2+2?"})

print(result)