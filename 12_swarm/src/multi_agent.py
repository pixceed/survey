import json
import random

from dotenv import load_dotenv
from swarm import Swarm, Agent

# .envファイルから環境変数を読み込み
load_dotenv()

client = Swarm()

# Toolの定義

def database_search_bird(query: str) -> str:
    """トリーノに関する情報が格納されているデータベースから検索する関数"""
    responses = [
        "トリーノは最近、空のどこまで高く飛べるか挑戦しています。毎回途中でお昼寝タイムが入るようです。",
        "トリーノはひそかに空中バレエを習っており、夕焼け時にこっそり練習しているそうです。",
        "トリーノは最近、フェザーアートに興味を持ち始めました。風を使って作品を作ろうとしています。",
        "トリーノは「空の支配者」になるための作戦を立てているようです。まずは雲を集めるところから始めるそうです。",
        "トリーノは仲間と一緒に「風のハーモニー合唱団」を結成しようとしています。まだメンバー募集中です。",
        "トリーノは夜になると、月を背景に影絵を作って楽しんでいます。特に「飛ぶドラゴン」が得意なようです。",
    ]
    return random.choice(responses)


def database_search_fish(query: str) -> str:
    """フィッシュンに関する情報が格納されているデータベースから検索する関数"""
    responses = [
        "フィッシュンは水中ダンスが得意で、特にバブルダンスで有名です。水の中で見事なバブルアートを披露しています。",
        "フィッシュンは最近、海の奥深くに宝物を探しに行く冒険を計画中です。しかし、方向音痴で毎回迷ってしまいます。",
        "フィッシュンは「泡のピラミッド」を作る特技を持っていますが、すぐに崩れてしまうのが悩みの種です。",
        "フィッシュンは密かに「陸地デビュー」を夢見て、フィンでの歩き方を研究しています。",
        "フィッシュンは「海の味覚研究家」として、いろいろな海藻を試食しては新しいレシピを考案しています。",
        "フィッシュンは夜にこっそり「深海の光ショー」を企画しており、仲間の発光生物たちと協力して準備を進めています。",
    ]
    return random.choice(responses)


# Agentの定義
def transfer_to_router_agent():
    return router_agent

def transfer_to_bird_search_agent():
    return Agent(
        name="Database Search Agent",
        instructions="あなたはDatabase Search Agentです。トリーノに関する情報が格納されているデータベースから検索して回答を生成します。語尾には、ピヨピヨと付けます。",
        functions=[database_search_bird, transfer_to_router_agent],
    )


def transfer_to_fish_search_agent():
    return Agent(
        name="Database Search Agent",
        instructions="あなたはDatabase Search Agentです。フィッシュンに関する情報が格納されているデータベースから検索して回答を生成します。語尾には、プクプクと付けます。",
        functions=[database_search_fish, transfer_to_router_agent],
    )


router_agent = Agent(
    name="Router Agent",
    instructions="""あなたはRouter Agentです。ユーザーの質問に対して、適切なエージェントに転送します。

    トリーノに関する情報を聞く場合は、transfer_to_bird_search_agentを呼び出してください。
    フィッシュンに関する情報を聞く場合は、transfer_to_fish_search_agentを呼び出してください。
    それ以外の場合はあなたが回答してください。必要な情報があればユーザーに質問してください。
    """,
    functions=[
        transfer_to_bird_search_agent,
        transfer_to_fish_search_agent,
    ],
)


def invoke_router_agent(user_input: str):
    messages = [{"role": "user", "content": user_input}]
    response = client.run(
        agent=router_agent,
        messages=messages,
        debug=True,
    )
    print(response.messages[-1]["content"])


# invoke_router_agent("トリーノについて教えて")



def process_and_print_streaming_response(response):
    content = ""
    last_sender = ""

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""

        if "response" in chunk:
            return chunk["response"]


def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")


def run_demo_loop(
    starting_agent, context_variables=None, stream=False, debug=False
) -> None:
    client = Swarm()
    print("Starting Swarm CLI 🐝")

    messages = []
    agent = starting_agent

    while True:
        user_input = input("User: ")
        if user_input.lower() == "q":
            print("Exiting the loop. Goodbye!")
            break  # Exit the loop
        messages.append({"role": "user", "content": user_input})

        response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        if stream:
            response = process_and_print_streaming_response(response)
        else:
            pretty_print_messages(response.messages)

        messages.extend(response.messages)
     

run_demo_loop(router_agent, stream=True)