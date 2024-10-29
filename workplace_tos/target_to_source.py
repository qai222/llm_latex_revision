import json
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import ollama
import tqdm
from pandas._typing import FilePath

from llm_latex_revision import *


def get_tos_translation(
        prompt_template: PromptTemplate,
        target_segment: TexSegment,
        translation_model_name: str,
        seed: int,
        prompt_including_complete: bool,
        tex_data_path: FilePath = "/mnt/home/qai/llm_latex_revision/data",
):
    assert len(target_segment.tex_file.split("/")) == 2 and target_segment.tex_file.endswith(".tex"), \
        f"`tex_file` field should be <style_name>/<*.tex>, you got {target_segment.tex_file}"
    target_style_name = target_segment.tex_file.split("/")[0]
    document_path = os.path.join(tex_data_path, target_segment.tex_file)
    with open(document_path, "r", encoding="utf-8") as text_file:
        content = text_file.read()

    prompt = prompt_template.get_prompt(segment=target_segment, complete_document=content,
                                        including_complete=prompt_including_complete)
    target_data = TargetStyleData(style_name=target_style_name, segment=target_segment)
    translation_param_set = {
        "seed": seed,
        "num_predict": -1,
        "num_keep": 0,
        "temperature": 0.0,
        # "num_gpu": 3,
        # "num_batch": 1,
        "num_ctx": 1024 * 8,  # 128 if from https://ollama.com/library/llama3.1:70b/blobs/a677b4a4b70c
    }

    translation = TargetToSourceStyleTranslation(
        target_data=target_data,
        prompt_template_name=prompt_template.name,
        translation_params=translation_param_set,
        translation_model_name=translation_model_name,
        prompt=prompt,
        prompt_including_complete=prompt_including_complete
    )
    return translation


def run_tos_translation(translation: TargetToSourceStyleTranslation):
    response_generated = ollama.generate(
        model=translation.translation_model_name,
        prompt=translation.prompt,
        options=translation.translation_params,
        stream=False,
        context=[],
        keep_alive=0.0,
    )
    return response_generated


def main_tos_translation(
        prompt_template: PromptTemplate,
        target_segment: TexSegment,
        translation_model_name: str,
        seed: int,
        prompt_including_complete: bool,
        tex_data_path: FilePath = "/mnt/home/qai/llm_latex_revision/data",
):
    tos = get_tos_translation(prompt_template, target_segment, translation_model_name, seed, prompt_including_complete,
                              tex_data_path)
    res = run_tos_translation(tos)
    source_data = SourceStyleData(text=res['response'], translation_identifier=tos.identifier)
    tos.n_tokens_source = res['eval_count']
    tos.n_tokens_target = res['prompt_eval_count']
    tos.source_data = source_data

    with open(f"responses/{tos.identifier}.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(f"translations/{tos.identifier}.json", "w") as f:
        json.dump(tos.model_dump(), f, indent=2)

    return tos


def main_tos_translation_all(segment_json: FilePath, repeat=1):
    with open(segment_json, "r") as f_segs:
        segments = [TexSegment(**d) for d in json.load(f_segs)]

    for segment in tqdm.tqdm(segments, desc="translate segments"):
        for pt in TOS_PROMPT_TEMPLATES:
            for rng_seed in range(repeat):
                main_tos_translation(
                    prompt_template=pt,
                    target_segment=segment,
                    translation_model_name="llama3.1:70b-instruct-q6_K",
                    seed=rng_seed + 42,
                    prompt_including_complete=False,
                    tex_data_path="/mnt/home/qai/llm_latex_revision/data",
                )


if __name__ == '__main__':
    main_tos_translation_all(
        segment_json="segments.json",
        repeat=1
    )
