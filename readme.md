LLM LaTeX Reviewer
---
Revise `.tex` file to fit your personal writing style.

### Workflow
- **step 1**: Split your already published `.tex` files (written in `target style`) to individual `segments`.
- **step 2**: Use a pretrained LLM (`LLM-large`) to rewrite the `segments` from `step 1` to `segments-rewritten` in different writing styles (`source styles`).
- **step 3**: Use `segments-rewritten` and `segments` as prompt and completion, respectively, to fine-tune another pretrained LLM (`LLM-small`).
- **step 4**: Use the fine-tuned `LLM-small` to rewrite `.tex` file in `target style`.
