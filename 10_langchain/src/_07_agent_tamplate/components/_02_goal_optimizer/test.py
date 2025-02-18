import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from orchestration.components._01_goal_creator.agent import Goal, GoalCreator
from orchestration.components._02_goal_optimizer.agent import OptimizedGoal, OptimizedGoalCreator

# .envファイルから環境変数を読み込み
load_dotenv()


def main():

    query = "2019年度~2023年度の売上高の変化は？"

    # LLM
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        temperature=0
    )

    gc = GoalCreator(llm=llm)
    ogc = OptimizedGoalCreator(llm=llm)

    goal: Goal = gc.run(query=query)
    print(f"[目標]\n{goal.description}\n\n")

    optimized_goal: OptimizedGoal = ogc.run(query=goal.description)
    print(f"[最適化された目標]\n{optimized_goal.description}\n\n")


if __name__ == "__main__":
    main()