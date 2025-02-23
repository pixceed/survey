import os
import time
import base64
from datetime import datetime
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    ContentFormat,
    AnalyzeResult,
)


# .envファイルから環境変数を読み込み
load_dotenv()



# 設定を読み込み
API_KEY = os.environ["DI_KEY"]
ENDPOINT = os.environ["DI_ENDPOINT"]
MODEL = "prebuilt-layout"

# クライアントを作成
client = DocumentIntelligenceClient(endpoint=ENDPOINT, credential=AzureKeyCredential(API_KEY))

# ファイルを読み込んで
file_path = "input/V_Rohto_Premium_Product_Information.pdf"
# file_path = "input/01_4℃ホールディングス_総合レポート2024.pdf"


print(f"{datetime.now()}: 分析開始")


# with open(file_path, "rb") as file:
#     print(f"{datetime.now()}: アップロード開始")

#     # バイト列で取得
#     pdf_data = file.read()

#     # バイナリデータをBase64エンコードする
#     base64_bytes = base64.b64encode(pdf_data)

#     # bytes型を文字列に変換（UTF-8）
#     base64_source = base64_bytes.decode('utf-8')
    
#     # 分析開始
#     poller = client.begin_analyze_document(
#         MODEL,
#         {"base64Source": base64_source},
#         output_content_format=ContentFormat.MARKDOWN,
#     )
    
#     # ステータスが完了になるまでポーリング
#     while not poller.done():
#         print(f"{datetime.now()}: Waiting...")
#         time.sleep(3)
    
#     # 結果を取得
#     result: AnalyzeResult = poller.result()


# 分析開始
poller = client.begin_analyze_document(
    MODEL,
    AnalyzeDocumentRequest(bytes_source=open(file_path, "rb").read()),
    output_content_format=ContentFormat.MARKDOWN,
)

# ステータスが完了になるまでポーリング
while not poller.done():
    print(f"{datetime.now()}: Waiting...")
    time.sleep(3)

# 結果を取得
result: AnalyzeResult = poller.result()

print(f"{datetime.now()}: 完了！！")

# 結果を出力ディレクトリにPDF名をもとにしたテキストファイルとして保存する
# 入力ファイル名から拡張子を除去してテキストファイル名にする
pdf_name = os.path.basename(file_path)
output_filename = os.path.splitext(pdf_name)[0] + ".txt"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_filename)

# 解析結果（ここでは result を文字列に変換）をファイルに保存
with open(output_path, "w", encoding="utf-8") as out_file:
    out_file.write(result.content)

print(f"{datetime.now()}: 結果を {output_path} に保存しました")