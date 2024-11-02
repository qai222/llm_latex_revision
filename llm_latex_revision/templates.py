from .schema import PromptTemplate

INSTRUCTION_COMPREHEND_BASE = """Please read the following complete LaTeX file to understand the full context of the research article:"""

ADDITIONAL_INSTRUCTION = """
After rewriting, modify the rewritten segment according to the following requirements step by step. 
1. Ensure that no new information (concepts, statements, etc.) is found in the rewritten segment compared to the original segment.
2. Ensure that no information (concepts, statements, etc.) is missing in the rewritten segment compared to the original segment.
3. Ensure that numerical descriptions are kept exactly as in the original segment (e.g., 'A has 100 apples' cannot be rewritten as 'A has many apples').
4. Ensure that latex macro stay intact (e.g., '\\begin{abstract}' will still be '\\begin{abstract}' in the rewritten text).
5. Ensure that inline latex citations stay intact (e.g., '\\cite{abcd1234}' will still be '\\cite{abcd1234}' in the rewritten segment). 
6. Ensure that the rewritten segment is not shorter than the original segment.
After modifying based on these requirements, provide the final rewritten segment in one LaTeX code block enclosed by three backticks.
Your response should be this LaTeX code block and nothing else.
"""

"""
6. Ensure that the rewritten segment is not significantly longer than the original segment (rewritten segment length <= 130% original segment length).
7. Ensure that the rewritten segment is not significantly shorter than the original segment (rewritten segment length >= 70% original segment length).
"""


TOS_TECHNICAL_REPORT = PromptTemplate(
    name="TOS_TECHNICAL_REPORT",
    instruction_role="You are a technical writer tasked with adapting a segment of a research article to fit into a technical report.",
    instruction_comprehend=INSTRUCTION_COMPREHEND_BASE,
    instruction_rewrite="Now, rewrite the following LaTeX segment so that it suits a technical report, adjusting the style and tone accordingly.",
    addition_instruction=ADDITIONAL_INSTRUCTION,
)

TOS_PATENT = PromptTemplate(
    name="TOS_PATENT",
    instruction_role="You are a senior researcher working with a patent attorney to describe findings from a research article.",
    instruction_comprehend=INSTRUCTION_COMPREHEND_BASE,
    instruction_rewrite="Now, rewrite the following LaTeX segment so that it suits a patent application, using formal and precise language.",
    addition_instruction=ADDITIONAL_INSTRUCTION,
)

TOS_LECTURE_NOTES = PromptTemplate(
    name="TOS_LECTURE_NOTES",
    instruction_role="You are creating lecture notes derived from a research article for an academic course.",
    instruction_comprehend=INSTRUCTION_COMPREHEND_BASE,
    instruction_rewrite="Now, rewrite the following LaTeX segment so that it suits lecture notes, explaining concepts in a clear and educational manner.",
    addition_instruction=ADDITIONAL_INSTRUCTION,
)

TOS_WHITE_PAPER = PromptTemplate(
    name="TOS_WHITE_PAPER",
    instruction_role="You are drafting a white paper to inform decision makers based on a research article.",
    instruction_comprehend=INSTRUCTION_COMPREHEND_BASE,
    instruction_rewrite="Now, rewrite the following LaTeX segment so that it suits a white paper in an objective, passive tone.",
    addition_instruction=ADDITIONAL_INSTRUCTION,
)


TOS_PROMPT_TEMPLATES = [
    TOS_WHITE_PAPER,
    TOS_LECTURE_NOTES,
    TOS_PATENT,
    TOS_TECHNICAL_REPORT,
]
