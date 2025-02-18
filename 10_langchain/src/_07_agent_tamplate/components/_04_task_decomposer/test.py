import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from orchestration.components._01_goal_creator.agent import Goal, GoalCreator
from orchestration.components._02_goal_optimizer.agent import OptimizedGoal, OptimizedGoalCreator
from orchestration.components._03_response_optimizer.agent import ResponseOptimizer
from orchestration.components._04_task_decomposer.agent import DecomposedTasks, TaskDecomposer

# .envファイルから環境変数を読み込み
load_dotenv()


def main():

    query = "2019年度~2023年度の売上高は？"
    print(f"|||||||||||||||||||||||||||||||||||||| [元の質問] ||||||||||||||||||||||||||||||||||||||||||||||\n")
    print(f"{query}\n\n")

    # LLM
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        temperature=0
    )

    gc = GoalCreator(llm=llm)
    ogc = OptimizedGoalCreator(llm=llm)
    ro = ResponseOptimizer(llm=llm)
    td = TaskDecomposer(llm=llm)

    goal: Goal = gc.run(query=query)
    print(f"|||||||||||||||||||||||||||||||||||||| [目標] |||||||||||||||||||||||||||||||||||||||||||||||||\n")
    print(f"{goal.description}\n\n")

    # optimized_goal: OptimizedGoal = ogc.run(query=goal.description)
    # print(f"|||||||||||||||||||||||||||||||||||||| [最適化された目標] |||||||||||||||||||||||||||||||||||||||\n")
    # print(f"{optimized_goal.description}\n\n")

    decomposed_tasks: DecomposedTasks = td.run(query=goal.description)
    print(f"|||||||||||||||||||||||||||||||||||||| [分割されたタスク] |||||||||||||||||||||||||||||||||||||||\n")
    for i, task in enumerate(decomposed_tasks.values):
        print(f"タスク{i+1}. {task}\n")
    print()



if __name__ == "__main__":
    main()