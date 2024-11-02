import json

from openai import OpenAI
from llm_latex_revision.finetune_lf import LFdata
from llm_latex_revision.schema import TexSegment, TargetToSourceStyleTranslation

client = OpenAI(api_key="0",base_url="http://0.0.0.0:8123/v1")

with open("../workplace_tos/segments.json", "r") as f:
    segments = json.load(f)

segments: TexSegment

def get_rewritten(txt: str):
    prompt = LFdata.tex2prompt(tex_str=txt)
    messages = [{"role": "user", "content": prompt}]
    result = client.chat.completions.create(messages=messages, model="Qwen/Qwen2.5-7B-Instruct")
    return result.choices[0].message.content

