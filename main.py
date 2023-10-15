import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("API Key is set!")

    st.set_page_config(
        page_title="Firelit Chat AI",
        page_icon="👾"
    )


def main():

    st.header("상권 분석 AI 👾")

    message("안녕하세요!")

    # 유저 입력
    message("안녕하세요 사용자님, 잘 부탁드려요!", is_user=True)

    with st.sidebar:
        user_input = st.text_input("USER: ", key='user_input')


if __name__ == '__main__':
    main()
