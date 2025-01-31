from dotenv import load_dotenv
import os

def get_api_key():
    load_dotenv()
    api_key = os.getenv('CLAUDE_API_KEY')
    return api_key