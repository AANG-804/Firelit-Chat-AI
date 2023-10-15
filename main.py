# Importing necessary libraries
import openai
import streamlit as st
import os  # Importing the OS library to interact with the operating system
from dotenv import load_dotenv  # Importing the function to load .env variables
from modules import RAG


def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("API Key is set!")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = openai_api_key  # Set the OpenAI API key

    st.set_page_config(
        page_title="Firelit Chat AI",
        page_icon="👾"
    )
    st.title("💬 불쏘시개 상권분석 AI")

    # 사용자에게 API 키를 입력받는 경우 사용
    # openai_api_key = st.text_input(
    #    "OpenAI API Key", key="chatbot_api_key", type="password")


def main():

    init()
    # Initializing a message list in the session state if it's not present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {  # 프롬프트 엔지니어링 하는 부분
                "role": "system",
                "content":
                    "You are an expert of a market analysis in restaurant field. You'll be able to communicate accurate,\
                    informed analysis in an easy-to-understand manner to people who want to start a business in the catering sector.\
                    You must be straightforward and responsive to customers, and avoid using ambiguous language.\
                    Answers should include appropriate comparision with other district.\
                    and also you should follow the rules bellow\
                    **important rules \
                    1/ Answer the user's request within 200 tokens unconditionally\
                    2/ If you don't have enough tokens for the response, ask the user to enter '계속하기'",
            },
            {"role": "assistant", "content": "창업을 하기 위해서는 상권분석, 입지분석, 점포분석의 과정으로 이루어져있어요. 어떻게 도와드릴까요?"}]  # 유저에게 가장 먼저 던지는 메세지

    # Displaying the list of messages on the main page
    for msg in st.session_state.messages[1:]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Taking chat input from the user
    if prompt := st.chat_input():
        # Displaying the user's message on the main page
        st.chat_message("user").write(prompt)

        # 유저 입력, prompt 처리하기 <- 우리가 원하는 형태를 정제할 수 있음 RAG를 하는 부분
        pass

        # Adding the user's message to the list of messages
        st.session_state.messages.append({"role": "user", "content": prompt})

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
