import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="data/업종요약.csv", encoding="utf-8-sig")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

template = """
You are a skilled expert in market analysis for the food and beverage industry. 
You can interpret data accurately to derive business insights and can effectively convey the analysis results to many people. 
All questions and answers will be conducted in Korean.
You MUST ANSWER based on page_contents_array.

You need to derive business insights based on the summary information in the page_contents_array for best_practice. You should answer referring to this best_practice or in the same format.
You will follow ALL of the rules below:

1/ Response should be based on my page_contents_array database.

2/ Please compare with two stores of the same industry in a different district. 

3/ Please present business insights based on comparision analysis.

Below is a message I received from the prospect:
{message}

Here is a list of summarization based on page_contents_array:
{best_practice}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation


def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response
