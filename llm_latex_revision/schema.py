from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TexSegment(BaseModel):
    """ A segment from a tex file. """

    type: str
    """ The segment type of this text in the original tex file. """

    index: int
    """ The segment index from the original tex file. """

    text: str
    """ The raw latex text. """

    tex_file: str
    """ The name of the original tex file. """

    beginning_char_index: int
    """ The beginning character index in the original tex file. """

    ending_char_index: int
    """ The ending character index in the original tex file. """


class TargetStyleData(BaseModel):
    """ Text written in the target style.  """

    segment: TexSegment
    """ Corresponding segment. """

    style_name: str
    """ The style name, usually the author's name. """


class SourceStyleData(BaseModel):
    """ Text written in a style that is different from the target style. """

    text: str
    """ The raw latex text. """

    translation_identifier: Optional[str] = None
    """ From which translation this data is generated. """


class TranslationParameterSet(BaseModel):
    """
    - Field names came from https://github.com/ollama/ollama/blob/main/docs/api.md#generate-request-with-options

    - Field default and docstring from 
        - https://github.com/ggerganov/llama.cpp/tree/master/examples/main or
        - https://github.com/abetlen/llama-cpp-python/blob/7403e002b8e033c0a34e93fba2b311e2118487fe/llama_cpp/llama.py#L146
        unless otherwise stated
    """

    num_keep: int = -1
    # llama.cpp default: 0 no tokens are kept. we set to -1: retain all tokens from the initial prompt.
    """ https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#keep-prompt """

    seed: int = 42
    """ https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#rng-seed """

    num_predict: int = -2
    # llama.cpp default: -1 = infinity, we set to: -2 = until context filled
    """ https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#number-of-tokens-to-predict """

    top_k: int = 40
    """ https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#top-k-sampling """

    top_p: float = 0.9
    """ https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#top-p-sampling """

    min_p: float = 0.1
    """ https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#min-p-sampling """

    temperature: float = 0.8
    """ https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#temperature """

    num_ctx: int = 0
    """ https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#context-size """

    # better use system defaults for the following

    # repeat_penalty: float = 1.0
    # mirostat: int = 0
    # mirostat_tau: float = 5.0
    # mirostat_eta: float = 0.1
    # penalize_newline: bool = True
    # numa: bool = False
    # tfs_z: float = 1.0
    # typical_p: float = 1.0
    # repeat_last_n: int = 64
    # num_batch: int
    # num_gpu: int
    # main_gpu: int
    # low_vram: bool
    # f16_kv: bool
    # vocab_only: bool
    # use_mmap: bool
    # use_mlock: bool
    # num_thread: bool
    # stop: list = []
    # presence_penalty: float
    # frequency_penalty:float


class PromptTemplate(BaseModel):
    name: str
    """ A unique name of this prompt template """

    instruction_role: str

    instruction_comprehend: str
    """ instruction to comprehend the original article """

    start_original: str = "[START OF THE COMPLETE LATEX DOCUMENT]"

    end_original: str = "[END OF THE COMPLETE LATEX DOCUMENT]"

    start_segment: str = "[START OF THE ORIGINAL LATEX SEGMENT]"

    end_segment: str = "[END OF THE ORIGINAL LATEX SEGMENT]"

    instruction_rewrite: str
    """ instruction to rewrite specific segment """

    addition_instruction: str

    def get_prompt(self, segment: TexSegment, complete_document: str, including_complete: bool = False):
        if including_complete:
            parts = [
                self.instruction_role,
                self.instruction_comprehend,
                self.start_original,
                complete_document,
                self.end_original,
                self.instruction_rewrite,
                self.start_segment,
                segment.text,
                self.end_segment,
                self.addition_instruction
            ]

        else:
            parts = [
                self.instruction_role,
                self.instruction_rewrite,
                self.start_segment,
                segment.text,
                self.end_segment,
                self.addition_instruction
            ]
        return "\n".join(parts)


class TargetToSourceStyleTranslation(BaseModel):  # TODO this is not used as it causes error if all fields are set
    """ A translation instance from target style to source style, used in generating data pairs for fine-tuning. """

    identifier: str = Field(default_factory=lambda: f"translation-{uuid4()}")
    """ The identifier of this translation. """

    target_data: TargetStyleData
    """ Target data in this translation. """

    source_data: Optional[SourceStyleData] = None
    """ Source data in this translation. """

    tokenizer: Optional[str] = None
    """ Tokenizer used in this translation. """

    n_tokens_target: Optional[int] = None
    """ The number of tokens of the target text. """

    n_tokens_source: Optional[int] = None
    """ The number of tokens of the source text. """

    translation_model_name: str
    """ The name of the translation model. """

    translation_params: dict = dict()
    """ The parameter set used in this translation. """

    prompt_template_name: str
    """ The prompt template used to generate input prompt. """

    prompt: str
    """ The actual prompt. """

    prompt_including_complete: bool
