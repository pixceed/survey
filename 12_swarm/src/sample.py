
from dotenv import load_dotenv
from swarm import Swarm, Agent

# .envファイルから環境変数を読み込み
load_dotenv()


client = Swarm()

def transfer_to_agent_b():
    return agent_b

agent_a = Agent(
    name="Agent A",
    instructions="あなたは、Agent Aです。",
    functions=[transfer_to_agent_b],
)

agent_b = Agent(
    name="Trivia Master",
    instructions="あなたは、トリビアマスターです。適当にトリビアを話してください。最後に「もいっ!」と言ってください。",
)

# question = "こんにちは"
question = "トリビアマスターに会いたい"

response = client.run(
    agent=agent_a,
    messages=[{"role": "user", "content": question}],
    debug=True
)

print(response.messages[-1]["content"])