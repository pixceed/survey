import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI

from orchestration.components._01_goal_creator.agent import Goal, GoalCreator
from orchestration.components._02_goal_optimizer.agent import OptimizedGoal, OptimizedGoalCreator
from orchestration.components._03_response_optimizer.agent import ResponseOptimizer
from orchestration.components._04_task_decomposer.agent import DecomposedTasks, TaskDecomposer
from orchestration.components._05_task_executor.agent import TaskExecutor
from orchestration.components._06_result_aggregator.agent import ResultAggregator
from orchestration.tools._01_kessan_rag.tool import get_kessan_rag_tool

# .envファイルから環境変数を読み込み
load_dotenv()


def main():

    query = "2019年~2023年の売上高の変化は？"
    print(f"|||||||||||||||||||||||||||||||||||||| [元の質問] ||||||||||||||||||||||||||||||||||||||||||||||\n")
    print(f"{query}\n\n")


    # LLM
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        temperature=0
    )

    # Tools
    kessan_rag_tool = get_kessan_rag_tool()

    tools = [kessan_rag_tool]

    gc = GoalCreator(llm=llm)
    ogc = OptimizedGoalCreator(llm=llm)
    ro = ResponseOptimizer(llm=llm)
    td = TaskDecomposer(llm=llm)
    te = TaskExecutor(llm=llm, tools=tools)
    ra = ResultAggregator(llm=llm)

    goal: Goal = gc.run(query=query)
    print(f"|||||||||||||||||||||||||||||||||||||| [目標] |||||||||||||||||||||||||||||||||||||||||||||||||\n")
    print(f"{goal.description}\n\n")

    # optimized_goal: OptimizedGoal = ogc.run(query=goal.description)
    # print(f"|||||||||||||||||||||||||||||||||||||| [最適化された目標] |||||||||||||||||||||||||||||||||||||||\n")
    # print(f"{optimized_goal.description}\n\n")

    optimized_response: str = ro.run(query=goal.description)

    print(f"|||||||||||||||||||||||||||||||||||||| [最適化されたレスポンス用プロンプト] ||||||||||||||||||||||\n")
    print(f"{optimized_response}\n\n")

    decomposed_tasks: DecomposedTasks = td.run(query=goal.description)
    print(f"|||||||||||||||||||||||||||||||||||||| [分割されたタスク] |||||||||||||||||||||||||||||||||||||||\n")
    for i, task in enumerate(decomposed_tasks.values):
        print(f"{i+1}. {task}\n")
    print()

    all_result_text = ""
    print(f"|||||||||||||||||||||||||||||||||||||| [各タスクの実行] |||||||||||||||||||||||||||||||||||||||\n")
    for i, task in enumerate(decomposed_tasks.values):

        result_text = ""
        task_result = te.run(task=task, task_exec_history=all_result_text)
        result_text += "--------------------------------------\n"
        result_text += f"タスク{i+1}: {task}\n\n"
        result_text += f"タスク{i+1}実行結果:\n{task_result}\n\n"

        print(result_text)

        all_result_text += result_text

    final_report = ra.run(
        query=goal.description,
        response_definition=optimized_response,
        results=all_result_text
    )

    print(f"|||||||||||||||||||||||||||| [最終レポート] |||||||||||||||||||||||||||||||||\n")
    print(f"{final_report}\n\n")




if __name__ == "__main__":
    main()