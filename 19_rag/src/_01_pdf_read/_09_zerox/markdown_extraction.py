import os
import json
import asyncio
from dotenv import load_dotenv
from pyzerox import zerox

# .envファイルから環境変数を読み込み
load_dotenv()


async def main():

    # file_path = "input/2408.14837.pdf"
    file_path = "input/Hada_Labo_Gokujun_Lotion_Overview.pdf"

    # モデルの定義
    model_name = "gpt-4o-mini"

    # モデルによっては追加のキーワード引数が必要な場合がある。その場合はここで指定。
    kwargs = {}

    # システムプロンプト
    custom_system_prompt = None

    # ページの指定
    # None: 全ページ
    # intまたはlist(int): 特定のページ番号を指定、1から始まる
    select_pages = None

    # Markdownファイルの出力先ディレクトリ
    output_dir = "./output/zerox"

    result = await zerox(
        file_path=file_path,
        model=model_name,
        output_dir=output_dir,
        custom_system_prompt=custom_system_prompt,
        select_pages=select_pages,
        **kwargs
    )

    return result

result = asyncio.run(main())