
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI



from agent_template.components._07_dynamic_task_controller.agent import DynamicTask, DynamicTaskController
from agent_template.tools.calc_tools import add, multiply, divide
from agent_template.components._05_task_executor.agent import TaskExecutor

# .envファイルから環境変数を読み込み
load_dotenv()


def main():

    # query = "量子で創薬分野をやっていますか？"
    query = "3と4を足してください。その出力を5で割る。次に小数点第一位で四捨五入する。その出力に1を引く。その出力に2を掛ける。 さらに最後にその数を100倍する。"

    print(f"|||||||||||||||||||||||||||||||||||||| [元の質問] ||||||||||||||||||||||||||||||||||||||||||||||\n")
    print(f"{query}\n\n")

    # LLM
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        temperature=0
    )

    tools = [add, multiply, divide]

    dtc = DynamicTaskController(llm=llm)
    te = TaskExecutor(llm=llm, tools=tools)



    task_history = []

    # print(f"|||||||||||||||||||||||||||||||||||||| [最初のタスク] ||||||||||||||||||||||||||||||||||||||||||||||\n")
    # task_content = dtc.run(query=query, task_history=task_history)
    # print(f"{task_content}\n\n")

    # task_result = te.run(task=task_content)
    # task_history.append(DynamicTask(task_content=task_content, task_result=task_result))

    # print(f"|||||||||||||||||||||||||||||||||||||| [2つ目のタスク] ||||||||||||||||||||||||||||||||||||||||||||||\n")
    # task_content = dtc.run(query=query, task_history=task_history)
    # print(f"{task_content}\n\n")

    # task_result = te.run(task=task_content)
    # task_history.append(DynamicTask(task_content=task_content, task_result=task_result))

    # print(f"|||||||||||||||||||||||||||||||||||||| [3つ目のタスク] ||||||||||||||||||||||||||||||||||||||||||||||\n")
    # task_content = dtc.run(query=query, task_history=task_history)
    # print(f"{task_content}\n\n")

    # task_result = te.run(task=task_content)
    # task_history.append(DynamicTask(task_content=task_content, task_result=task_result))



    task_counter = 1
    while True:

        task_content = dtc.run(query=query, task_history=task_history)

        if "finish" in task_content.strip().lower():
            break

        print(f"|||||||||||||||||||||||||||||||||||||| [タスク{task_counter}] ||||||||||||||||||||||||||||||||||||||||||||||\n")
        print(f"[タスク内容]\n{task_content}\n\n")

        task_result = te.run(task=task_content)
        print(f"[タスク実行結果]\n{task_result}\n\n")


        task_history.append(DynamicTask(task_content=task_content, task_result=task_result))


if __name__ == "__main__":
    main()