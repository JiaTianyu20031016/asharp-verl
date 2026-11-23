"""Add Vine step separators to an existing processed MATH parquet dataset."""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STEP_INSTRUCTION = (
    "Separate each reasoning step from the next with exactly one blank line (two newline characters, \\n\\n)."
)


def add_step_instruction(table: pa.Table) -> pa.Table:
    """Return a copy with the instruction appended to each user prompt."""
    prompt_index = table.schema.get_field_index("prompt")
    if prompt_index < 0:
        raise ValueError("MATH parquet must contain a prompt column")
    prompts = table.column(prompt_index).to_pylist()
    for messages in prompts:
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("Expected one user message per MATH prompt")
        content = messages[0]["content"]
        if STEP_INSTRUCTION in content:
            raise ValueError("MATH prompt already contains the Vine step instruction")
        messages[0]["content"] = f"{content} {STEP_INSTRUCTION}"
    field = table.schema.field(prompt_index)
    return table.set_column(prompt_index, field, pa.array(prompts, type=field.type))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        source = args.source_dir / f"{split}.parquet"
        destination = args.output_dir / f"{split}.parquet"
        if destination.exists():
            raise FileExistsError(destination)
        pq.write_table(add_step_instruction(pq.read_table(source)), destination)
        print(destination)


if __name__ == "__main__":
    main()
