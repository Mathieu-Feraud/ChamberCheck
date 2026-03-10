from dotenv import load_dotenv
from openai import OpenAI

from ChamberCheck.model_analysis.abn_test.abn_test_builder import _build_llm_prompts_from_metadata


load_dotenv()

prompts, _ = _build_llm_prompts_from_metadata(
    metadata_json_path="data/output/scrape_003/abn_test/abn_test_prompts_metadata_024.json",
    top=1,
)

prompt = prompts[0]["prompt"]
print("prompt_length:", len(prompt))

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    max_completion_tokens=500,
    temperature=0.3,
)

choice = response.choices[0]
message = choice.message

print("finish_reason:", choice.finish_reason)
print("usage:", getattr(response, "usage", None))
print("content_type:", type(message.content))
print("content_repr:", repr(message.content))
print("refusal:", getattr(message, "refusal", None))
print("tool_calls:", getattr(message, "tool_calls", None))
