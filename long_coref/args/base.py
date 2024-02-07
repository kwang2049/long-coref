from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import logging
import os
from typing import Optional, Set, Type
from long_coref.utils import get_commit_hash, parse_cli
from abc import ABC, abstractmethod


@dataclass
class AutoNameArguments:
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def escaped(self) -> Set[str]:
        """Escaped items for building the run_name."""
        return set()

    @property
    def run_name(self) -> str:
        items = [
            f"{k}_{getattr(self, k)}"
            for k in self.__annotations__.keys()
            if k not in self.escaped
        ]
        return "/".join([self.name] + items + [f"git_hash_{get_commit_hash()}"])

    @classmethod
    def parse_and_print(cls: Type[AutoNameArguments]) -> None:
        run_name = parse_cli(cls).run_name
        print(run_name)

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # Needed as a MixIn


@dataclass
class DumpableArgumentsMixIn(ABC):
    output_dir: Optional[str] = None

    @abstractmethod
    def build_output_dir(self) -> str:
        pass

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.output_dir = self.build_output_dir()
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # Needed as a MixIn

    def dump_arguments(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        fargs = os.path.join(self.output_dir, "config.json")
        with open(fargs, "w") as f:
            json.dump(asdict(self), f, indent=4)
        logging.info(f"Dumped config to {fargs}")
