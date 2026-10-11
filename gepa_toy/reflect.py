"""Reflection prompt templates for the support-ticket task."""

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
