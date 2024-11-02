import os
import pathlib

from llm_latex_revision.utils import FilePath

"""
First we need to create a data folder looks like this, `cwc` is the writing style that you want to convert to.
```
- data/         # main data folder
    - raw/
        - cwc/  # contains tex files in the desired writing style
    - tos/      # contains generated data from `LLM-large` to be used in fine tuning
        - cwc/
            - responses/
            - translations/
``` 
"""
TARGET_WRITING_STYLE = "cwc"
MAIN_DATA_FOLDER = os.path.abspath("data")
TEX_FILE_FOLDER = os.path.join(MAIN_DATA_FOLDER, "raw", TARGET_WRITING_STYLE)
TOS_FOLDER = os.path.join(MAIN_DATA_FOLDER, "tos", TARGET_WRITING_STYLE)
TOS_RESPONSES_FOLDER = os.path.join(TOS_FOLDER, "responses")
TOS_TRANSLATIONS_FOLDER = os.path.join(TOS_FOLDER, "translations")


def make_folders(paths: list[FilePath]):
    for p in paths:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)


make_folders([TEX_FILE_FOLDER, TOS_TRANSLATIONS_FOLDER, TOS_RESPONSES_FOLDER])

"""
**step 1**: Split your already published `.tex` files (written in the `target style`) to individual `segments`.
"""
import glob
from tqdm import tqdm
from llm_latex_revision.utils import split_tex, json_dump, json_load
from llm_latex_revision.schema import TargetToSourceStyleTranslation
from llm_latex_revision.templates import TOS_PROMPT_TEMPLATES

TARGET_SEGMENTS = []
for tex_file in tqdm(sorted(glob.glob(os.path.join(TEX_FILE_FOLDER, "*.tex")))):
    TARGET_SEGMENTS += split_tex(tex_file)
json_dump([s.model_dump() for s in TARGET_SEGMENTS], os.path.join(TOS_FOLDER, "segments.json"))

"""
- **step 2**: Use a pretrained LLM (`LLM-large`) to rewrite the `segments` from `step 1` to `segments-rewritten` in different writing styles (`source styles`).
First download and install ollama, then run 
```
ollama pull llama3.1:70b-instruct-q6_K  # this works on 3x24GB gpu node
ollama serve  # opens the ollama API
```
Then use the following script to generate target-to-soruce style translations.
"""
if not TARGET_SEGMENTS:
    TARGET_SEGMENTS = json_load(os.path.join(TOS_FOLDER, "segments.json"))

TOS_REPEAT = 1  # num of repeat for a source writing style

for segment in tqdm(TARGET_SEGMENTS, desc="translate segments"):
    for pt in TOS_PROMPT_TEMPLATES:
        for rng_seed in range(TOS_REPEAT):
            translation = TargetToSourceStyleTranslation.from_target_segment(
                prompt_template=pt,
                target_segment=segment,
                translation_model_name="llama3.1:70b-instruct-q6_K",
                seed=rng_seed + 42,
                prompt_including_complete=False,
                data_path=MAIN_DATA_FOLDER,
            )
            translation.run_translation(dump=True, data_path=MAIN_DATA_FOLDER, )

"""
- **step 3**: Use `segments-rewritten` and `segments` as prompt and completion, respectively, to fine-tune another pretrained LLM (`LLM-small`).
"""
# first prepare finetune data to be used in llama factory
from transformers import AutoTokenizer
from llm_latex_revision.finetune_lf import LFdata

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
                                          trust_remote_code=True)
LFdata.prepare_finetune_data(
    output_folder=MAIN_DATA_FOLDER,
    translation_folder=TOS_TRANSLATIONS_FOLDER,
    target_writing_style=TARGET_WRITING_STYLE,
    tokenizer=tokenizer,
    cutoff=4096
)
# run finetune for qwen2.5
"""
Run finetune for QWen2.5 in LLaMA-factory:
1. install https://github.com/hiyouga/LLaMA-Factory
2. install deepspeed
    - note you need to make sure your torch is compiled against the same CUDA version you have, you can use 
        `conda install -c nvidia/label/cuda-12.4 cuda-toolkit` to override existing CUDA installation
3. run lora finetune using 
    `FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=1,2,3 llamafactory-cli train qwen2-7b-lora-sft-deepspeed.yaml`
4. merge lora with the base model `llamafactory-cli export merge.yaml`, this will by default save the merged model at
    `models/qwen2.5_lora_sft` 
"""

"""
- **step 4**: Use the fine-tuned `LLM-small` to rewrite `.tex` file.
1. Start an OPENAI API instance thru llama factory using
    `API_PORT=8123 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api inference.yaml`
2. Using the following code to rewrite a tex file 
"""
