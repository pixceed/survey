from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

# Output Parserの準備
output_parser = StrOutputParser()

# Output Parserの実行
message = AIMessage(content="AIからのメッセージです")
output = output_parser.invoke(message)
print(type(output))
print(output)