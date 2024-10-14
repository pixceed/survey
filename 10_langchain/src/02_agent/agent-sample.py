'''
python実行機能があるエージェントを作成
'''


import os
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import chat_agent_executor
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

# チャットエージェントの作成
agent_executor = chat_agent_executor.create_tool_calling_executor(
    chat_model,
    tools=tools
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


question = '''
以下の<入力文章></入力文章>から、「万右衛門」の数をカウントして、その数を教えてください。

<入力文章>
一二月二八日。庭の周囲の高い垣根の向こうには、とても小さな家々のわらぶき屋根があって、もっとも貧しい階層の人たちが暮らしている。これらの小さな住まいの一つから、うなり声が絶えず発せられている――人が苦痛で発するときの深いうめき声である。私は、昼も夜も、もう一週間以上もそれを聞いている。しかし、音はあたかも断末魔のあえぎのように長くなり、また大きくなってきた。「誰かそこで重い病気なのでしょう」と、私の古い通訳者の万右衛門が、ひどく同情して言う。
音はしだいに耳障りで、神経に障るようになってきた。そのせいで、私はかなりぶっきらぼうに言った。「誰か死にかけているなら、いっそそうなった方がましだと思うがね」
万右衛門は、私の意地悪な言葉を払いのけるかのように、両手で突然、慌ただしく三度も身振りをした。このかわいそうな仏教徒は、ぶつぶつ唱え、とがめるような表情をして、立ちさった。そして、いささか良心が咎めたので、使いの者を遣り、病人に医者が必要か、また何か助けが要るか聞きにやらせた。戻った使いの者の話では、医者が病人を診ているし、他は格別必要ないということだった。
けれども、万右衛門の古くさい身振りにもかかわらず、その忍耐強い神経も、この音に煩わせられるようになってきた。彼は、これらの音から少しでも逃れたいとして、通りに近い、正面にある小さい部屋に移りたいと白状した。私も気になって書きものも、読書もできない。私の書斎は、一番奥にあり、病人があたかも同じ部屋にいるかのように、うめき声が間近に聞こえるのである。病気の程度が分かるような、一定の身の毛のよだつ音色を発している。私はつぎのように自問し続けている。私が苦しめられているこれらの音を立てている人間がこれから長く持ちこたえることがどうしてできるのだろうかと。
つぎの朝遅く、病人の部屋で小さな木魚を叩く音と何人かの声で「南無妙法蓮華経」と唱える声で、うめき声がかき消されていたのは救いというか、いくらかほっとした。明らかにその家の中には僧侶や親せきの者たちが集まっている。万右衛門が「死にそうですね」と言う。そして、仏様に捧げる祈りの文句を繰り返した。
木魚の音や読経は数時間続いた。それらが終わったとき、うめき声がまた聞こえた。一呼吸、一呼吸がうめき声だった！　夕方になると。それらはさらにひどくなった――身の毛のよだつほどである。そして、それが突然止んだ。死の沈黙が数秒続いた。そして、ウワーッと泣き出す声――それは女性の泣き声で――そして名を叫ぶ声が聞こえた。「あゝ、亡くなりましたね！」と万右衛門が言った。
私たちは相談した。万右衛門はこの家の者たちがとても貧しいことを知っていた。私は、良心が咎めたので、遺族にわずかな額だが、香典を出そうと言った。万右衛門は、私が全くの好意からそうしようとしているのだと思って、それがいいでしょうと答えた。私は使いの者に、悔やみの言葉と死んだ男のことが分かるなら聞いてくるようにと言った。そこには一種の悲劇があるのではないかと感じていたのだ。そして、日本人の悲劇は一般に興味深いものである。
</入力文章>
'''
# question = "Calculate the value of 2 + 2."

# テンプレートを使ってエージェントに質問する
formatted_question = prompt_template.format(question=question)

# エージェントに数式を実行させる
response = agent_executor.invoke({"messages": [("human", formatted_question)]})

# 結果を出力
print()
print("="*80)
print(response["messages"][-1].content)
print("-"*80)
print(response["messages"][-1])
print("="*80)
print()
