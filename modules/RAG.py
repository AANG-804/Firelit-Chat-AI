from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import os

# 1. Vectorise the sales response csv data

# 파일 읽어들이고 VDB 구축 <- DB 구축을 매번 하는 문제 있음! 수정 필요
# loader = CSVLoader(file_path="../data/매장요약.csv", encoding="utf-8-sig")
# documents = loader.load()
# embeddings = OpenAIEmbeddings()
# db = FAISS.from_documents(documents, embeddings)


def retrieve_info(query, db):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


# 3. Setup LLMChain & prompts
def make_template(message, db):  # Ensure the database is initialized
    vectorized_data = retrieve_info(message, db)
    template = f"""
    You need to derive business insights based on the summary information in the page_contents_array. You should answer referring to this best_practice or in the same format.
    You will follow ALL of the rules below:

    1/ Response should be based on my page_contents_array database.
    2/ Please compare with two cases of the same industry in a different region.
    3/ Please present business insights based on comparision analysis.

    Below is a message I received from the prospect:
    {message}

    Here is a list of summarization based on page_contents_array:
    {vectorized_data}

    Please write the best response that I should send to this prospect:
    """
    return template
