from typing import TypedDict, List
from typing_extensions import Annotated

# メッセージ追加用の関数
def add_messages(existing_messages: List[str], new_messages: List[str]) -> List[str]:
    return existing_messages + new_messages

# ステートの構築
class State(TypedDict):
    # messages: Annotated[List[str], add_messages]

    messages: List[str]

# サンプルの使用
if __name__ == "__main__":
    # 初期メッセージの作成
    initial_state: State = {"messages": ["こんにちは", "おはようございます"]}
    
    # 新しいメッセージを追加
    new_messages = ["こんばんは", "さようなら"]
    initial_state["messages"] = add_messages(initial_state["messages"], new_messages)

    # initial_state["messages"] = initial_state["messages"] + new_messages

    # 最終的なメッセージを表示
    print(initial_state)
