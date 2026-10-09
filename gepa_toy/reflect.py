"""Reflector: reflection prompt plus extraction of the proposed instruction.

See docs/gepa_design.md sections 4.3 and 4.4 ("Reflection").
"""

import re
from collections.abc import Callable

from gepa_toy.gepa import Propose, Rollout

REFLECTION_PROMPT = """\
The instruction below is given to a model that classifies customer support \
tickets. The examples after it show tickets, the model's output under this \
instruction, and feedback on each output.

Current instruction:
```
{instruction}
```

Examples:

{rollouts}

Write an improved instruction that fixes the problems the feedback shows. Keep \
the rules general: they must hold for tickets that are not in the examples. \
Reply with the complete replacement instruction in one fenced code block and \
nothing else.
"""

# Default instruction-proposal prompt from the reference implementation, with
# <curr_param> and <side_info> renamed to our placeholders. Copied from
# https://github.com/gepa-ai/gepa/blob/462e437a09be67d2acb59564cc0ea59132f5777c/src/gepa/strategies/instruction_proposal.py
DSPY_REFLECTION_PROMPT = """I provided an assistant with the following instructions to perform a task for me:
```
{instruction}
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
{rollouts}
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

# Both fences must start a line: a ``` mentioned mid-sentence before the real
# block would otherwise pair with its opening fence and the block would be lost.
FENCED_BLOCK = re.compile(
    r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL
)


def build_reflection_prompt(
    template: str, instruction: str, rollouts: list[Rollout]
) -> str:
    examples = "\n\n".join(
        f"Example {i}\n{r.trace}\n{r.feedback}" for i, r in enumerate(rollouts, 1)
    )
    values = {"instruction": instruction, "rollouts": examples}
    # One pass over the template, so a placeholder inside the instruction or a
    # trace is never substituted (sequential str.replace would do that).
    return re.sub(r"\{(instruction|rollouts)\}", lambda m: values[m[1]], template)


def extract_instruction(reply: str) -> str:
    blocks = FENCED_BLOCK.findall(reply)
    if not blocks:
        raise ValueError(f"No fenced code block in reflection reply: {reply[:80]!r}")
    return blocks[-1].removesuffix("\n")


def make_reflector(
    complete: Callable[[str], str], template: str = REFLECTION_PROMPT
) -> Propose:
    def propose(parent: str, rollouts: list[Rollout]) -> str:
        # Cleared first and recorded step by step so the core can log exactly
        # what a failed proposal sent and got back.
        propose.last_prompt = propose.last_reply = None
        propose.last_prompt = build_reflection_prompt(template, parent, rollouts)
        propose.last_reply = complete(propose.last_prompt)
        return extract_instruction(propose.last_reply)

    propose.last_prompt = None
    propose.last_reply = None
    return propose
