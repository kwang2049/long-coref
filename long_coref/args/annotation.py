from dataclasses import dataclass
from enum import Enum
import os
from typing import Optional, Set

from long_coref.utils import parse_cli
from long_coref.args.base import AutoNameArguments, DumpableArgumentsMixIn


class AnnotationMethod(str, Enum):
    parenthesis = "parenthesis"
    replacement = "replacement"
    sentinel = "sentinel"


class TFIDFType(str, Enum):
    tfidf = "tfidf"
    idf = "idf"


@dataclass
class AnnotationArguments(AutoNameArguments, DumpableArgumentsMixIn):
    data_dir: Optional[str] = None
    allennlp_model_path: str = (
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    )
    annotating_labeled_queries: bool = False
    root_output_dir: str = "annotated"

    # Model related:
    spans_per_word: float = 0.2  # 0.4
    max_antecedents: int = 50  # 50
    max_span_width: int = 10  # 30
    coarse_to_fine: bool = False  # False
    max_antecedents_further: int = 1  # 50
    span_batch_size: int = 4096

    # Annotation related:
    annotation_method: AnnotationMethod = AnnotationMethod.parenthesis
    tfidf: Optional[TFIDFType] = None
    across_paragraphs_only: bool = True
    first_paragraph_mention_only: bool = True
    different_spans_only: bool = True
    ignore_stopwords: bool = True

    def __post_init__(self) -> None:
        self.annotation_method = AnnotationMethod(self.annotation_method)
        if self.tfidf:
            self.tfidf = TFIDFType(self.tfidf)
        super().__post_init__()  # Needed as a MixIn

    @property
    def escaped(self) -> Set[str]:
        return {"root_output_dir"}

    def build_output_dir(self) -> str:
        return os.path.join("annotated", self.run_name)


if __name__ == "__main__":
    args = parse_cli(AnnotationArguments)
    print(args.run_name)
    args.dump_arguments()
