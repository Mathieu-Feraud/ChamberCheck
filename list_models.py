from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("OPENAI_API_KEY not found in environment")
else:
    client = OpenAI(api_key=api_key)
    models = client.models.list()
    gpt_models = [m.id for m in models.data if 'gpt' in m.id.lower()]
    print("Available GPT models:")
    for model in sorted(gpt_models):
        print(f"  - {model}")
