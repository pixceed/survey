from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio


# .envファイルから環境変数を読み込み
load_dotenv()


async def main():
    agent = Agent(
        task="YouTubeで、東海オンエアの最新動画を再生して",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    print(result)


asyncio.run(main())