from orchestration.tools._01_kessan_rag.tool import get_kessan_rag_retriever


def main():

    # RAGシステム読み込み
    rag_fusion_chain = get_kessan_rag_retriever()


    question = "ギターヒーローの正体は？"
    answer = rag_fusion_chain.invoke(question)

    print("-"*80)
    print(f"\n[質問]\n{question}\n\n[回答]\n{answer}\n")



if __name__=="__main__":
    main()