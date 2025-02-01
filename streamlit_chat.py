# Tutorial available on Streamlit:
# https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from langchain_anthropic import ChatAnthropic

st.title("Helpful Infromation as Aid")

# Review alignment with RedCross tone using Claude
def generate_response(input_text):
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    return llm.invoke(input_text)

if "messages" not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input("How can I help you today?"):

    # Save messages previously sent
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show the user prompt
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve the assistant response
    with st.chat_message("assistant"):
        stream = generate_response(
            input_text=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        )
        response = st.write_stream(stream)

    # Update chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
