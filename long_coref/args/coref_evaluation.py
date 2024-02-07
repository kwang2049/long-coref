from dataclasses import dataclass
from long_coref.args.base import AutoNameArguments


@dataclass
class CorefEvaluationArguments(AutoNameArguments):
    allennlp_model_path: str = (
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    )
    ontonotes_eval_path: str = "data/OntoNotes/test.english.v4_gold_conll"
    output_dir: str = "resolutions"
    spans_per_word: float = 0.4
    max_antecedents: int = 50
    max_span_width: int = 30
    coarse_to_fine: bool = True
    max_antecedents_further: int = 50
    span_batch_size: int = 4096
    long_doc: bool = False


if __name__ == "__main__":
    CorefEvaluationArguments.parse_and_print()
