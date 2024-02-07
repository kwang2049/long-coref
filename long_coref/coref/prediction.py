from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import logging
import os
import tempfile
from typing import Dict, List, Optional, Set, Tuple, Type, TypedDict, Union
from allennlp.common.util import get_spacy_model
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp_models.coref.predictors import coref
from allennlp_models.coref.dataset_readers.conll import ConllCorefReader
import torch
from long_coref.coref.utils import ArchiveContent, clone_and_extract, pop_and_return
from long_coref.coref.modeling import CoreferenceResolver
from allennlp.data.token_indexers.pretrained_transformer_mismatched_indexer import (
    PretrainedTransformerMismatchedIndexer,
)
from long_coref.utils import color_generator
import networkx as nx


@dataclass
class Token:
    text: str
    text_with_ws: Optional[str] = None

    def __str__(self) -> str:
        return self.text

    @classmethod
    def decode(cls: Type[Token], tokens: List[Token]) -> str:
        decoded: str = "".join(token.text_with_ws for token in tokens)
        return decoded.strip()

    def prepend(self, new_text: str) -> None:
        self.text = new_text + self.text
        if self.text_with_ws is not None:
            self.text_with_ws = new_text + self.text_with_ws

    def append(self, new_text: str) -> None:
        self.text = self.text + new_text
        if self.text_with_ws is not None:
            self.text_with_ws = self.text_with_ws + new_text


class TokenJson(TypedDict):
    text: str
    text_with_ws: Optional[str]


class ResolutionJson(TypedDict):
    pargraph_sentences: List[List[List[TokenJson]]]
    top_spans: List[List[int]]
    antecedents: List[Optional[List[int]]]


@dataclass
class Resolution:
    document: List[Token]  # List of words
    pargraph_sentences: List[List[List[Token]]]
    top_spans: Set[Tuple[int, int]]
    mention2antecedent: Dict[
        Tuple[int, int], Tuple[int, int]
    ]  #  Span -> antecedent. Only spans that have an antecedent are included

    def build_clusters(self) -> List[Set[Tuple[int, int]]]:
        G: nx.classes.graph.Graph = nx.path_graph(0)
        spans = list(self.top_spans)
        span2index = {span: i for i, span in enumerate(spans)}
        for mention, antecedent in self.mention2antecedent.items():
            nx.add_path(G, (span2index[antecedent], span2index[mention]))
        clusters: List[Set[Tuple[int, int]]] = []
        for cluster in nx.connected_components(G):
            clusters.append({spans[span_id] for span_id in cluster})
        return clusters

    def get_span_tokens(self, span: Tuple[int, int]) -> List[Token]:
        return self.document[span[0] : span[1] + 1]

    def shift_all_values(self, offset: int) -> None:
        """Shift all values by offset."""
        self.top_spans = {
            (span[0] + offset, span[1] + offset) for span in self.top_spans
        }
        self.mention2antecedent = {
            (mention[0] + offset, mention[1] + offset): (
                antecedent[0] + offset,
                antecedent[1] + offset,
            )
            for mention, antecedent in self.mention2antecedent.items()
        }

    def merge(self, next_resolution: Resolution, new_paragraph_start: int) -> None:
        """Merge the next resolution into the current one, where only the new paragraphs will be considered."""
        new_paragraph_sentences = next_resolution.pargraph_sentences[
            new_paragraph_start:
        ]
        new_text = sum(sum(new_paragraph_sentences, []), [])
        self.document += new_text
        self.pargraph_sentences += new_paragraph_sentences
        self.top_spans.update(next_resolution.top_spans)
        self.mention2antecedent.update(next_resolution.mention2antecedent)

    def to_dict(self) -> ResolutionJson:
        top_spans = [list(span) for span in sorted(self.top_spans)]
        antecedents = [
            (
                list(self.mention2antecedent[tuple(span)])
                if tuple(span) in self.mention2antecedent
                else None
            )
            for span in top_spans
        ]
        return {
            "pargraph_sentences": [
                [[asdict(token) for token in sent] for sent in paragraph]
                for paragraph in self.pargraph_sentences
            ],
            "top_spans": top_spans,
            "antecedents": antecedents,
        }

    @classmethod
    def from_dict(cls: Type[Resolution], resolution_json: ResolutionJson) -> Resolution:
        mention2antecedent = {
            tuple(mention): tuple(antecedent)
            for mention, antecedent in zip(
                resolution_json["top_spans"], resolution_json["antecedents"]
            )
            if antecedent is not None
        }
        pargraph_sentences = [
            [[Token(**token) for token in sent] for sent in paragraph]
            for paragraph in resolution_json["pargraph_sentences"]
        ]
        document = sum(sum(pargraph_sentences, []), [])
        return Resolution(
            document=document,
            pargraph_sentences=pargraph_sentences,
            top_spans=set(tuple(span) for span in resolution_json["top_spans"]),
            mention2antecedent=mention2antecedent,
        )

    def to_html(self) -> str:
        clusters = self.build_clusters()
        start2cluster_ids: Dict[int, List[int]] = {}
        end2cluster_ids: Dict[int, List[int]] = {}
        for cluster_id, cluster in enumerate(clusters):
            for start, end in cluster:
                start2cluster_ids.setdefault(start, [])
                start2cluster_ids[start].append(cluster_id)
                end2cluster_ids.setdefault(end, [])
                end2cluster_ids[end].append(cluster_id)
        colors = [color for color, _ in zip(color_generator(), clusters)]
        word_position = 0
        html = ""
        for paragraph in self.pargraph_sentences:
            tagged_words: List[Token] = []
            for sentence in paragraph:
                for word in sentence:
                    word = replace(word)  # Clone
                    if word_position in start2cluster_ids:
                        word.prepend(
                            "".join(
                                [
                                    f'<span style="border:2px solid {colors[cluster_id]};">'
                                    for cluster_id in start2cluster_ids[word_position]
                                ]
                            )
                        )
                    if word_position in end2cluster_ids:
                        word.append(
                            "".join(
                                [f"</span>" for _ in end2cluster_ids[word_position]]
                            )
                        )
                    tagged_words.append(word)
                    word_position += 1
            tagged_paragraph = Token.decode(tagged_words)
            html += f"<p>{tagged_paragraph}</p>\n"
        return html

    def dump_html(self, fto: Optional[str] = None) -> None:
        if fto is None:
            fto = tempfile.mkstemp(suffix=".html")[1]
        with open(fto, "w") as f:
            f.write(self.to_html())
        logging.info(f"Dumped html to {fto}")

    def to_allennlp_format(self) -> PredictionDict:
        top_spans = list(sorted(self.top_spans))
        predicted_antecedents = [
            0 if top_span in self.mention2antecedent else -1 for top_span in top_spans
        ]
        span2span_index = {span: i for i, span in enumerate(top_spans)}
        antecedent_indices: List[List[int]] = []
        for top_span in top_spans:
            if top_span in self.mention2antecedent:
                antecedent = self.mention2antecedent[top_span]
                antecedent_index = span2span_index[antecedent]
            else:
                antecedent_index = len(top_spans)  # So that it would not be indexed
            antecedent_indices.append(
                [antecedent_index]
            )  # So this aligns with predicted_antecedents = 0's
        clusters = self.build_clusters()
        return PredictionDict(
            document=self.document,
            top_spans=[list(span) for span in top_spans],
            predicted_antecedents=predicted_antecedents,
            antecedent_indices=antecedent_indices,
            clusters=[[list(span) for span in cluster] for cluster in clusters],
        )

    @classmethod
    def from_allennlp_format(
        cls: Type[Resolution],
        prediciton: PredictionDict,
        pargraph_sents: List[List[List[Token]]],
    ) -> Resolution:
        top_spans = {tuple(span) for span in prediciton["top_spans"]}
        # Note that predicted_antecedents do not directly correspond to the indices of top_spans!!!
        # One needs to go to antecedent_indices for the actual indices of top_spans.
        mention2antecedent = {
            tuple(mention): tuple(
                prediciton["top_spans"][prediciton["antecedent_indices"][i][antecedent]]
            )
            for i, (mention, antecedent) in enumerate(
                zip(
                    prediciton["top_spans"],
                    prediciton["predicted_antecedents"],
                )
            )
            if antecedent != CorefPredictor.NO_ANTECEDENT
        }
        return Resolution(
            document=sum(sum(pargraph_sents, []), []),
            pargraph_sentences=pargraph_sents,
            top_spans=top_spans,
            mention2antecedent=mention2antecedent,
        )


class PredictionDict(TypedDict):
    document: List[str]
    top_spans: List[List[int]]
    predicted_antecedents: List[int]
    clusters: List[List[List[int]]]
    antecedent_indices: List[List[int]]


class CorefPredictor(coref.CorefPredictor):
    NO_ANTECEDENT = -1

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(coref.CorefPredictor, self).__init__(model, dataset_reader)
        self._spacy = get_spacy_model(
            "en_core_web_sm", pos_tags=False, parse=False, ner=False
        )
        self._spacy.add_pipe("sentencizer")
        self._spacy.max_length = 5000000
        self._dataset_reader: ConllCorefReader
        self.set_device()

    def set_device(self, device: Optional[Union[int, str]] = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        logging.info(f"Using device {device}")
        self.cuda_device = device
        self._model.to(device)

    def alter_hyperparameters(
        self,
        spans_per_word: float = 0.4,
        max_antecedents: int = 50,
        max_span_width: int = 30,
        coarse_to_fine: bool = True,
        max_antecedents_further: int = 50,
        span_batch_size: int = 4096,
    ):
        dataset_reader: ConllCorefReader = self._dataset_reader
        model: CoreferenceResolver = self._model
        dataset_reader._max_span_width = max_span_width
        model._spans_per_word = spans_per_word
        model._max_antecedents = max_antecedents
        model._max_span_width = max_span_width
        model._coarse_to_fine = coarse_to_fine
        model.max_antecedents_further = max_antecedents_further
        model.span_batch_size = span_batch_size

    @classmethod
    def from_extracted_archive(
        cls: Type[CorefPredictor], archive_content: ArchiveContent
    ) -> CorefPredictor:
        config = archive_content.config
        dataset_reader_params = config.get("validation_dataset_reader")
        kwargs: dict = dataset_reader_params.as_dict()
        token_indexers = {
            "tokens": PretrainedTransformerMismatchedIndexer(
                **pop_and_return(kwargs["token_indexers"]["tokens"], to_pop="type")
            )
        }
        dataset_reader = ConllCorefReader(
            max_span_width=kwargs["max_span_width"],
            token_indexers=token_indexers,
            max_sentences=kwargs.get("max_sentences"),
        )
        model = CoreferenceResolver.from_extracted_archive(archive_content)
        return CorefPredictor(model=model, dataset_reader=dataset_reader)

    @classmethod
    def from_archive(cls: Type[CorefPredictor], archive_path: str) -> CorefPredictor:
        archive_content = clone_and_extract(archive_path)
        return cls.from_extracted_archive(archive_content)

    def resolve_paragraphs(self, paragraphs: List[str]) -> Resolution:
        if all(len(p.strip()) == 0 for p in paragraphs):
            return Resolution(
                document=[Token(text=p, text_with_ws=p) for p in paragraphs],
                pargraph_sentences=[
                    [[Token(text=p, text_with_ws=p)]] for p in paragraphs
                ],
                top_spans=set(),
                mention2antecedent={},
            )

        self._model.eval()
        pargraph_sents = [
            [
                [
                    Token(text=token.text, text_with_ws=token.text_with_ws)
                    for token in sent
                ]
                for sent in self._spacy(paragraph).sents
            ]
            for paragraph in paragraphs
        ]
        sentences = sum(pargraph_sents, [])
        instance = self._dataset_reader.text_to_instance(
            [[token.text for token in sent] for sent in sentences]
        )
        predicted: PredictionDict = self.predict_instance(instance)
        return Resolution.from_allennlp_format(
            prediciton=predicted, pargraph_sents=pargraph_sents
        )

    def resolve_iterative(
        self, paragraphs: List[str], context_paragraph_num: int = 2
    ) -> Resolution:
        resolution: Resolution = Resolution(
            document=[],
            top_spans=set(),
            mention2antecedent=dict(),
            pargraph_sentences=[],
        )
        paragraph_nwords = []
        for i, paragraph in enumerate(paragraphs):
            context_start = max(i - context_paragraph_num, 0)
            context_end = i
            context_paragraphs = paragraphs[context_start:context_end]
            nwords_before_context_start = sum(paragraph_nwords[:context_start])
            resolution_i = self.resolve_paragraphs(context_paragraphs + [paragraph])
            resolution_i.shift_all_values(nwords_before_context_start)
            # resolution_i.dump_html(f"resolutions/{i}.html")
            resolution.merge(next_resolution=resolution_i, new_paragraph_start=-1)
            # resolution.dump_html(f"resolutions/{i}-merged.html")
            paragraph_nwords.append(
                sum(len(sent) for sent in resolution_i.pargraph_sentences[-1])
            )
        return resolution


if __name__ == "__main__":
    from long_coref.utils import get_wiki_pages, set_logger_format
    import tqdm
    import time

    set_logger_format()
    model_name = "spanbert-large"
    # model_name = "longformer-base"
    predictor = CorefPredictor.from_archive(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
        # "/home/fb20user07/research/allennlp-coref/output_longformer-base/model.tar.gz"
    )
    wiki_pages = get_wiki_pages(
        [
            "The Half Moon, Putney",
            "Mahatma Gandhi",
            "Mansa Musa",
            # "Bill Clinton",
            # "J. K. Rowling",
            # "Bono",
            # "Mark Antony",
            # "Germany",
            # "Great Pyramid of Giza",
            # "North Korea",
            # "Westminster Abbey",
            # "Aspirin",
            # "Diesel engine",
            # "Guinea pig",
            # "Polio",
            # "Spanish flu",
            # "Windows XP",
            # "King K. Rool",
            # "Sailor Moon (character)",
            # "Cello",
            # "FIFA",
            # "Mueller report",
        ]
    )
    resolutions: List[Resolution] = []
    start = time.time()
    for page in tqdm.tqdm(wiki_pages, desc="Resolving documents"):
        resolution = predictor.resolve_iterative(
            page.paragraphs, context_paragraph_num=2
        )
        resolutions.append(resolution)
    duration = time.time() - start
    output_dir = os.path.join("resolutions", model_name)
    os.makedirs(output_dir, exist_ok=True)
    for page, resolution in zip(wiki_pages, resolutions):
        resolution.dump_html(os.path.join(output_dir, f"{page.title}.html"))
    nwords = sum(len(resolution.document) for resolution in resolutions)
    print(f"Speed: {nwords/duration:.1f} w/s")
    # SpaCy speed: 600+ w/s
