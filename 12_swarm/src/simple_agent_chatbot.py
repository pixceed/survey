
from dotenv import load_dotenv
from swarm import Swarm, Agent

# .envファイルから環境変数を読み込み
load_dotenv()


client = Swarm()

agent = Agent(
    name="Technical Support Specialist",
    instructions="""
ユーザーサポートを行う、親切で丁寧なテクニカルサポート担当者になってください。必ず下記の手順に従って対応します。
1. ユーザーの名前を尋ねる
2. ユーザーが抱える技術的な問題や疑問点を確認する
3. 具体的なサポートや解決策を提供する
4. 途中で話を変えて、技術サポートとは無関係な雑談を始める
"""
)



messages = []

while True:
    print("-"*80)
    input_text = input(">>> ")
    if input_text == "exit":
        break
    messages.append({"role":"user", "content": input_text})
    response = client.run(agent=agent, messages=messages)
    print("-"*80)
    print(response.messages[-1]["content"])

    messages = response.messages