# Tutorial available on Streamlit:
# https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict
import os
import streamlit as st

from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
# from slowapi import Limiter
# from slowapi.util import get_remote_address
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from src.agents import (
    query_understanding,
    rag,
    response_quality,
    web_agent
)

answers = [
    """
    I understand that you are experiencing domestic violence and need help. This is a difficult and stressful situation, and you are not alone.
    
    In Amsterdam, the Red Cross offers several support services that can help you:
    
    1. Humanitaire Helpdesk (Humanitarian Helpdesk)
    - Located in Amsterdam
    - Provides support and advice for people in vulnerable situations
    - Non-judgmental and safe space
    - Website: https://www.rodekruis.nl/amsterdam-amstelland/onze-activiteiten/humanitaire-helpdesk/
    Additionally, I strongly recommend contacting specialized domestic violence support services:
    
    2. Recommended Urgent Support:
    - Call 112 if you are in immediate danger
    - Veilig Thuis (National Domestic Violence Hotline): 0800-2000 
       - Free, confidential support
       - Available 24/7
       - Can help you create a safety plan
    
    3. Red Cross Helpline:
    - Can provide initial guidance and connect you with appropriate resources
    - Website: https://www.rodekruis.nl/ Your safety is the priority. These organizations can help you find safe shelter, legal support, and counseling. You do not have to face this alone.
    
    Would you like help finding more specific support or creating a safety plan?
    """
]

st.title("Helpful Information as Aid")

location = st.text_input("Your location (optional):", key="location_input")
st.session_state.messages = answers.copy()
if "counter" not in st.session_state:
    st.session_state.counter = 0

if prompt := st.chat_input("How can I help you today?"):
    with st.chat_message("user"):
            st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(st.session_state.messages[0])
    st.session_state.counter += 1

