import glob
import json
import os.path
from typing import Optional
from scipy import stats
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, FilePath

from .schema import TargetToSourceStyleTranslation
from .templates import ADDITIONAL_INSTRUCTION


class LFdata(BaseModel):
    """ training data of llama factory in alpaca format """

    instruction: str
    """user instruction"""

    input: Optional[str] = ""
    """ user input """

    output: str
    """model response"""

    system: Optional[str] = ""
    """ system prompt """

    history: list = Field(default_factory=list)

    @staticmethod
    def tex2prompt(tex_str: str):
        ins = "You are a senior researcher in chemistry, chemical engineering, and computer science."
        ins += "\nNow, rewrite the following LaTeX segment so that it suits a research article to be published in an academic journal."
        ins += "\n[START OF THE ORIGINAL LATEX SEGMENT]\n"
        ins += tex_str
        ins += "\n[END OF THE ORIGINAL LATEX SEGMENT]\n"
        ins += ADDITIONAL_INSTRUCTION
        return ins

    @classmethod
    def from_tos_translation(cls, translation: TargetToSourceStyleTranslation):
        ins = LFdata.tex2prompt(translation.source_data.text)
        return cls(
            instruction=ins,
            output=translation.target_data.segment.text
        )

    def get_n_total_tokens(self, tokenizer):
        return count_token(self.instruction, self.output, tokenizer)[-1]

    @staticmethod
    def prepare_finetune_data(
            output_folder: FilePath, translation_folder: FilePath, target_writing_style: str, tokenizer,
            cutoff: int = 4096, plot_token_ecdf:bool = False
    ):
        lf_data_list = []
        token_counts = []
        for translation_json in glob.glob(os.path.join(translation_folder, "*.json")):
            with open(translation_json, "r") as f:
                trans = TargetToSourceStyleTranslation(**json.load(f))
            lf_data = LFdata.from_tos_translation(trans)
            n_token = lf_data.get_n_total_tokens(tokenizer)
            if n_token > cutoff:
                token_counts.append(n_token)
                continue
            lf_data_list.append(lf_data.model_dump())
        with open(os.path.join(output_folder, f"latex_revision_{target_writing_style}.json"), "w") as f:
            json.dump(lf_data_list, f, indent=2)

        data_info = {f"latex_revision_{target_writing_style}": {
            "file_name": f"latex_revision_{target_writing_style}.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history"
            }
        }}
        with open(os.path.join(output_folder, "dataset_info.json"), "w") as f:
            json.dump(data_info, f, indent=2)

        if plot_token_ecdf:
            res = stats.ecdf(token_counts)
            ax = plt.subplot()
            res.cdf.plot(ax)
            plt.savefig(os.path.join(output_folder, f"token_ecdf_{target_writing_style}.png"))


def count_token(prompt, completion, tokenizer):
    prompt_token = tokenizer(text=prompt)
    completion_token = tokenizer(text=completion)
    n_p = len(prompt_token)
    n_c = len(completion_token)
    n_total = n_p + n_c
    return n_p, n_c, n_total
