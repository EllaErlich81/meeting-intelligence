"""JSON read/write helpers for persisting stage artifacts to disk.

Every stage's output is written as a JSON file in the run's output
directory. This is what makes "run each stage separately" possible: a
later stage (or a later CLI invocation, possibly on a different day) reads
the JSON artifact instead of requiring the whole pipeline to be re-run in
one process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

ModelT = TypeVar("ModelT", bound=BaseModel)


def write_model(path: str | Path, model: BaseModel) -> None:
    Path(path).write_text(model.model_dump_json(indent=2), encoding="utf-8")


def read_model(path: str | Path, model_cls: type[ModelT]) -> ModelT:
    return model_cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_model_list(path: str | Path, models: list[BaseModel]) -> None:
    Path(path).write_text(json.dumps([m.model_dump(mode="json") for m in models], indent=2), encoding="utf-8")


def read_model_list(path: str | Path, model_cls: type[ModelT]) -> list[ModelT]:
    adapter: TypeAdapter[list[ModelT]] = TypeAdapter(list[model_cls])
    return adapter.validate_json(Path(path).read_text(encoding="utf-8"))
