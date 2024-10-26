import glob
import json
import os.path
from collections import Counter

from pandas._typing import FilePath
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode

from llm_latex_revision import TexSegment

"""
This script split a tex file into segments such that the union of the segments is identical to the original tex file.
"""


def count_environments(nodes, env_counts):
    for node in nodes:
        if isinstance(node, LatexEnvironmentNode):
            env_counts[node.environmentname] += 1
            count_environments(node.nodelist, env_counts)
        elif hasattr(node, 'nodelist'):
            count_environments(node.nodelist, env_counts)


def collect_unique_names(nodes, unique_macros, unique_environments):
    for node in nodes:
        if isinstance(node, LatexMacroNode):
            unique_macros.add(node.macroname)
        elif isinstance(node, LatexEnvironmentNode):
            unique_environments.add(node.environmentname)
            collect_unique_names(node.nodelist, unique_macros, unique_environments)
        elif hasattr(node, 'nodelist'):
            collect_unique_names(node.nodelist, unique_macros, unique_environments)


def find_segments(nodes, segments, known_segment_commands, known_segment_environments, parent_pos=0):
    for node in nodes:
        if isinstance(node, LatexMacroNode):
            # Check if the macro is a segment marker
            if node.macroname in known_segment_commands:
                segments.append((node.pos, node))
        elif isinstance(node, LatexEnvironmentNode):
            # Check if the environment is a segment marker
            if node.environmentname in known_segment_environments:
                segments.append((node.pos, node))
            # Recursively process child nodes
            find_segments(node.nodelist, segments, known_segment_commands, known_segment_environments, node.pos)
        elif hasattr(node, 'nodelist'):
            # Recursively process child nodes
            find_segments(node.nodelist, segments, known_segment_commands, known_segment_environments, parent_pos)
        else:
            continue


def split_tex(
        tex_file: FilePath,
) -> list[TexSegment]:
    with open(tex_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse the LaTeX content
    lw = LatexWalker(content)
    nodes, _, _ = lw.get_latex_nodes()

    # Get macros
    unique_macros = set()
    unique_environments = set()
    env_counts = Counter()
    collect_unique_names(nodes, unique_macros, unique_environments)
    count_environments(nodes, env_counts)

    # Treat all macros starting with 'sub' as segments
    known_segment_commands = {
        name for name in unique_macros if
        name.startswith('sub') or name in ['section', 'chapter', 'part', 'paragraph']
    }

    # Treat environments that occur more than once as segments
    known_segment_environments = {name for name, count in env_counts.items() if count > 1}
    # known_segment_environments = {'abstract', 'figure', 'table', 'environment'}  # Add your environment names here

    segments = []
    find_segments(nodes, segments, known_segment_commands, known_segment_environments)

    # Sort segments by position
    segments.sort(key=lambda x: x[0])

    # Ensure start at position 0
    if segments and segments[0][0] != 0:
        segments.insert(0, (0, None))
    elif not segments:
        segments.append((0, None))  # Handle case with no segments

    # Add end of content to segments
    segments.append((len(content), None))

    # Split the content into segments
    split_contents = []
    for i in range(len(segments) - 1):
        start_pos = segments[i][0]
        end_pos = segments[i + 1][0]
        segment_node = segments[i][1]
        segment_content = content[start_pos:end_pos]

        # Determine segment type
        if isinstance(segment_node, LatexMacroNode):
            marker = segment_node.macroname
        elif isinstance(segment_node, LatexEnvironmentNode):
            marker = segment_node.environmentname
        elif segment_node is None:
            marker = 'non-segmented'
        else:
            marker = 'unknown'
        tex_segment = TexSegment(
            type=marker,
            index=i,
            text=segment_content,
            tex_file=os.path.basename(os.path.dirname(tex_file)) + "/" + os.path.basename(tex_file),
            beginning_char_index=start_pos,
            ending_char_index=end_pos,
        )
        split_contents.append(tex_segment)
    # for idx, segment in enumerate(split_contents):
    #     print(f"Segment {idx + 1} ({segment.type}):\n{len(segment.text)}\n{segment.text}\n{'-' * 40}")
    assert "".join([seg.text for seg in split_contents]) == content
    return split_contents


if __name__ == '__main__':
    TEX_FILES = [*glob.glob("/mnt/home/qai/llm_latex_revision/data/cwc/*.tex")]
    SEGMENTS = []
    for TEX_FILE in TEX_FILES:
        SEGMENTS += split_tex(tex_file=TEX_FILE)
    with open("segments.json", "w") as f:
        json.dump([SEG.model_dump() for SEG in SEGMENTS], f, indent=2)

