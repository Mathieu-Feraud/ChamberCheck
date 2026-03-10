from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly this JSON object and no other text: {\"ok\": true}",
        }
    ],
    max_completion_tokens=200,
)

message = response.choices[0].message
print(type(message.content))
print(repr(message.content))
print("finish_reason:", response.choices[0].finish_reason)
print("refusal:", getattr(message, "refusal", None))
