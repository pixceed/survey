"""Tool for Python Exec"""

from typing import Optional

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_experimental.utilities import PythonREPL

class PythonExec(BaseTool):
    name: str = "Python_REPL"
    description: str = "Pythonコードを実行して結果を表示するツール"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:

        try:
            
            # 最終評価を出力するようにする
            last_line = query.split('\n')[-1]
            code = query + f"\nprint({last_line})"

            # コードを実行
            python_repl = PythonREPL()
            result = python_repl.run(code)

            # 実行結果が空の場合
            if not result:
                result = "結果が返されませんでした。"

        except Exception as e:
            # エラーが発生
            result = f"エラーが発生しました: {str(e)}"

        return result