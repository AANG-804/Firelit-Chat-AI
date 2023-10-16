import openai
import streamlit as st

import pandas as pd

import os  # Importing the OS library to interact with the operating system
import time
from dotenv import load_dotenv  # Importing the function to load .env variables

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from modules import RAG


def initialize_db():
    if "db" not in st.session_state:
        file_path = "./data/업종요약_new.csv"

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The file {file_path} does not exist. Please check the path.")

        try:
            loader = CSVLoader(file_path=file_path, encoding="utf-8-sig")
            documents = loader.load()
        except Exception as e:
            raise RuntimeError(
                f"Error loading {file_path}. Ensure it is a valid CSV file.") from e

        embeddings = OpenAIEmbeddings()
        st.session_state.db = FAISS.from_documents(documents, embeddings)


def init():
    # .env파일을 환경변수로 읽어드림
    load_dotenv()
    initialize_db()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("API Key is set!")
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = st.session_state.openai_api_key  # Set the OpenAI API key

    st.set_page_config(
        page_title="Firelit Chat AI",
        page_icon="👾"
    )
    st.title("💬 불쏘시개 상권분석 AI")

    # 사용자에게 API 키를 입력받는 경우 사용
    # openai_api_key = st.text_input(
    #    "OpenAI API Key", key="chatbot_api_key", type="password")


# 메세지를 받아 로그에 추가
def add_message_log(role, content):
    pass


def main():

    init()
    # Initializing a message list in the session state if it's not present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {  # 프롬프트 엔지니어링 하는 부분
                "role": "system",
                "content": """
                You are a skilled expert in market analysis for the food and beverage industry. 
                You can interpret data accurately to derive business insights and can effectively convey the analysis results to many people. 
                Please respond to the user's question based on the given information. If you cannot find the answer, kindly state '답변을 찾을 수 없어요'
                All questions and answers will be conducted in Korean.
                """
            },
            {"role": "assistant", "content": """분석을 진행할 상권과 업종을 알려주시면 상권 분석을 도와드리겠습니다.
            먼저 분석을 원하는 행정동을 입력해주세요"""}]  # 유저에게 가장 먼저 던지는 메세지
        st.session_state.first_trial = True

    # Displaying the list of messages on the main page
    for msg in st.session_state.messages[1:]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Sidebar에 사용설명 추가
    with st.sidebar:
        st.header("🤷‍♂️Firelit ChatAI Beta 사용설명", divider='rainbow')
        st.write(
            """
            현재 저희 챗봇은 Beta 버전으로 기능 구현 중에 있습니다.\n
            🚨주의🚨 \n
            1. 답변 생성에는 30초~1분이 소요됩니다.\n
            2. 현재 분석 가능한 행정동, 업종은 다음과 같습니다.
            """
        )
        st.write(pd.DataFrame(
            {"행정동": ['장전동', '청룡동', '노포동', '부곡동', '구서동'],
             "업종": ["한식", "일식", "중식", "카페", "치킨"]}))

    # 폰트 설정
    with open("style.css") as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

    if st.session_state.first_trial:
        # 상권을 선택하게 만들기
        if "locations" not in st.session_state:
            if user_input := st.chat_input():
                st.chat_message("user").write(user_input)
                st.session_state.locations = user_input
                st.session_state.messages.append(
                    {"role": "user", "content": user_input})
                st.session_state.messages.append(
                    {"role": "assistant", "content": "감사합니다! 이번엔 분석할 업종을 입력해주세요!"})
                st.chat_message("assistant").write(
                    "감사합니다! 이번엔 분석할 업종을 입력해주세요!")
        elif "industry" not in st.session_state:
            if user_input := st.chat_input():
                st.chat_message("user").write(user_input)
                st.session_state.industry = user_input
                st.session_state.messages.append(
                    {"role": "user", "content": user_input})
                user_message = f'{st.session_state.locations}의 {st.session_state.industry}에 대해 분석해줘'

                with st.spinner("응답 생성중..."):
                    response = RAG.generate_response(
                        user_message, st.session_state.db, st.session_state.openai_api_key)

                # msg = response.choices[0].message
                # Adding the assistant's response to the list of messages
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})
                # Displaying the assistant's response on the main page
                st.chat_message("assistant").write(response)

                st.session_state.first_trial = False
    else:
        # Taking chat input from the user
        if user_input := st.chat_input():
            # Displaying the user's message on the main page
            st.chat_message("user").write(user_input)

            # 유저 입력(user_input), prompt 처리하기 <- 우리가 원하는 형태를 정제할 수 있음 RAG를 하는 부분
            prompt = user_input

            # Adding the user's message to the list of messages
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            # Making a request to OpenAI's API to get a response based on the list of messages
            with st.spinner("응답 생성중..."):
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613", messages=st.session_state.messages, max_tokens=400, temperature=0)

            # Extracting the assistant's response from the API response
            msg = response.choices[0].message

            # Adding the assistant's response to the list of messages
            st.session_state.messages.append(msg)

            # Displaying the assistant's response on the main page
            st.chat_message("assistant").write(msg.content)


if __name__ == "__main__":
    main()
