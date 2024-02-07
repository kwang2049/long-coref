from __future__ import annotations
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Type
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.pass_through_encoder import PassThroughEncoder
from allennlp.nn import InitializerApplicator
from allennlp_models.coref import models
from allennlp.common.params import Params
from allennlp.modules.text_field_embedders.basic_text_field_embedder import (
    BasicTextFieldEmbedder,
)
from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import (
    PretrainedTransformerMismatchedEmbedder,
)
from long_coref.coref.utils import clone_and_extract, ArchiveContent, pop_and_return
import torch.nn.functional as F
from allennlp.nn import util


logger = logging.getLogger(__name__)


class CoreferenceResolver(models.CoreferenceResolver):
    """
    This `Model` implements the coreference resolution model described in
    [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/pdf/1804.05392.pdf)
    by Lee et al., 2018.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representations are scored and used to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the `text` `TextField` we get as input to the model.
    context_layer : `Seq2SeqEncoder`
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : `FeedForward`
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward : `FeedForward`
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size : `int`
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width : `int`
        The maximum width of candidate spans.
    spans_per_word: `float`, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: `int`, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    coarse_to_fine: `bool`, optional (default = `False`)
        Whether or not to apply the coarse-to-fine filtering.
    inference_order: `int`, optional (default = `1`)
        The number of inference orders. When greater than 1, the span representations are
        updated and coreference scores re-computed.
    lexical_dropout : `int`
        The probability of dropping out dimensions of the embedded text.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        mention_feedforward: FeedForward,
        antecedent_feedforward: FeedForward,
        feature_size: int,
        max_span_width: int,
        spans_per_word: float,
        max_antecedents: int,
        coarse_to_fine: bool = False,
        inference_order: int = 1,
        lexical_dropout: float = 0.2,
        initializer: InitializerApplicator = InitializerApplicator(),
        trained_with_coarse_to_fine: Optional[bool] = None,
        max_antecedents_further: Optional[int] = None,
        span_batch_size: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab,
            text_field_embedder,
            context_layer,
            mention_feedforward,
            antecedent_feedforward,
            feature_size,
            max_span_width,
            spans_per_word,
            max_antecedents,
            coarse_to_fine,
            inference_order,
            lexical_dropout,
            initializer,
            **kwargs,
        )
        self.span_batch_size = span_batch_size
        assert span_batch_size > max_antecedents
        self.trained_with_coarse_to_fine = (
            coarse_to_fine
            if trained_with_coarse_to_fine is None
            else trained_with_coarse_to_fine
        )  # Affects how we do in _distance_pruning
        self.max_antecedents_further = (
            max_antecedents
            if max_antecedents_further is None
            else max_antecedents_further
        )

    def _bilinear_weighting(
        self,
        top_span_embeddings: torch.FloatTensor,
        top_antecedent_indices: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Calculate the bilinear weighting between the span embeddings and their antecedents.
        This was missed for distance prunning in the orginal implementation.

        :param top_span_embeddings: (batch_size, num_spans_to_keep, embedding_size)
        :param top_antecedent_indices: (num_spans_to_keep, max_antecedents)
        :return: (batch_size, num_spans_to_keep, max_antecedents)
        """
        # (batch_size, num_spans_to_keep, embedding_size)
        num_spans_to_keep = top_span_embeddings.shape[1]
        expanded_shape = list(top_span_embeddings.shape)
        max_antecedents = top_antecedent_indices.shape[-1]
        expanded_shape.insert(2, max_antecedents)

        # (batch_size, num_spans_to_keep, embedding_size)
        bilinear_weights: torch.Tensor = self._coarse2fine_scorer(top_span_embeddings)

        # (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        expanded_bilinear_weights = bilinear_weights.unsqueeze(2).expand(expanded_shape)
        antecedent_bilinear_weights = expanded_bilinear_weights.gather(
            dim=1,
            index=top_antecedent_indices.unsqueeze(-1).expand_as(
                expanded_bilinear_weights
            ),
        )

        # (batch_size, num_spans_to_keep, max_antecedents)
        partial_antecedent_scores = (
            top_span_embeddings.unsqueeze(2) * antecedent_bilinear_weights
        ).sum(dim=-1)
        return partial_antecedent_scores

    def _distance_pruning(
        self,
        top_span_embeddings: torch.FloatTensor,
        top_span_mention_scores: torch.FloatTensor,
        max_antecedents: int,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.LongTensor]:
        """
        Generates antecedents for each span and prunes down to `max_antecedents`. This method
        prunes antecedents only based on distance (i.e. number of intervening spans). The closest
        antecedents are kept.

        # Parameters

        top_span_embeddings: `torch.FloatTensor`, required.
            The embeddings of the top spans.
            (batch_size, num_spans_to_keep, embedding_size).
        top_span_mention_scores: `torch.FloatTensor`, required.
            The mention scores of the top spans.
            (batch_size, num_spans_to_keep).
        max_antecedents: `int`, required.
            The maximum number of antecedents to keep for each span.

        # Returns

        top_partial_coreference_scores: `torch.FloatTensor`
            The partial antecedent scores for each span-antecedent pair. Computed by summing
            the span mentions scores of the span and the antecedent. This score is partial because
            compared to the full coreference scores, it lacks the interaction term
            w * FFNN([g_i, g_j, g_i * g_j, features]).
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_mask: `torch.BoolTensor`
            The mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_offsets: `torch.LongTensor`
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_indices: `torch.LongTensor`
            The indices of every antecedent to consider with respect to the top k spans.
            (batch_size, num_spans_to_keep, max_antecedents)
        """
        # These antecedent matrices are independent of the batch dimension - they're just a function
        # of the span's position in top_spans.
        # The spans are in document order, so we can just use the relative
        # index of the spans to know which other spans are allowed antecedents.

        num_spans_to_keep = top_span_embeddings.size(1)
        device = util.get_device_of(top_span_embeddings)

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (max_antecedents,),
        # (num_spans_to_keep, max_antecedents)
        (
            top_antecedent_indices,
            top_antecedent_offsets,
            top_antecedent_mask,
        ) = self._generate_valid_antecedents(  # noqa
            num_spans_to_keep, max_antecedents, device
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        top_antecedent_mention_scores = util.flattened_index_select(
            top_span_mention_scores.unsqueeze(-1), top_antecedent_indices
        ).squeeze(-1)

        partial_antecedent_scores = 0
        if self.trained_with_coarse_to_fine:
            partial_antecedent_scores_list = []
            for b in range(0, num_spans_to_keep, self.span_batch_size):
                e = min(b + self.span_batch_size, num_spans_to_keep)
                antecedent_indices = top_antecedent_indices[b:e] - b
                look_ahead = max_antecedents if b > 0 else 0
                span_embeddings = top_span_embeddings[:, b - look_ahead : e]
                antecedent_indices = (
                    top_antecedent_indices[b - look_ahead : e] - b + look_ahead
                )
                antecedent_indices[:look_ahead] = 0
                partial_antecedent_scores = self._bilinear_weighting(
                    top_span_embeddings=span_embeddings,
                    top_antecedent_indices=antecedent_indices,
                )  # (batch_size, (max_antecedents+)span_batch_size, max_antecedents)
                partial_antecedent_scores_list.append(
                    partial_antecedent_scores[:, -(e - b) :]
                )
            partial_antecedent_scores = torch.cat(partial_antecedent_scores_list, dim=1)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents) * 4
        top_partial_coreference_scores = (
            top_span_mention_scores.unsqueeze(-1) + top_antecedent_mention_scores
        )
        top_antecedent_indices = top_antecedent_indices.unsqueeze(0).expand_as(
            top_partial_coreference_scores
        )
        top_antecedent_offsets = top_antecedent_offsets.unsqueeze(0).expand_as(
            top_partial_coreference_scores
        )
        top_antecedent_mask = top_antecedent_mask.expand_as(
            top_partial_coreference_scores
        )

        return (
            top_partial_coreference_scores + partial_antecedent_scores,
            top_antecedent_mask,
            top_antecedent_offsets,
            top_antecedent_indices,
        )

    def keep_top_antecedents(
        self,
        top_partial_coreference_scores: torch.FloatTensor,
        top_antecedent_mask: torch.BoolTensor,
        top_antecedent_offsets: torch.LongTensor,
        top_antecedent_indices: torch.LongTensor,
        top: int,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.LongTensor]:
        """ "Keep top-k of the results."""
        (
            batch_size,
            num_spans_to_keep,
            nantecedents,
        ) = top_partial_coreference_scores.shape
        if nantecedents == top:
            return (
                top_partial_coreference_scores,
                top_antecedent_mask,
                top_antecedent_offsets,
                top_antecedent_indices,
            )

        # (batch_size, num_spans_to_keep, top)
        (
            kept_partial_coreference_scores,
            kept_antecedent_mask,
            kept_antecedent_indices,
        ) = util.masked_topk(top_partial_coreference_scores, top_antecedent_mask, top)
        selected_antecedent_indices = util.batched_index_select(
            top_antecedent_indices.reshape(
                batch_size * num_spans_to_keep, nantecedents, 1
            ).clone(),
            kept_antecedent_indices.view(-1, top),
        ).view(batch_size, num_spans_to_keep, top)
        selected_antecedent_offsets = util.batched_index_select(
            top_antecedent_offsets.reshape(
                batch_size * num_spans_to_keep, nantecedents, 1
            ).clone(),
            kept_antecedent_indices.view(-1, top),
        ).view(batch_size, num_spans_to_keep, top)
        return (
            kept_partial_coreference_scores,
            kept_antecedent_mask,
            selected_antecedent_offsets,
            selected_antecedent_indices,
        )

    def forward(
        self,  # type: ignore
        text: TextFieldTensors,
        spans: torch.IntTensor,
        span_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Copied and modifed from the original code. Changes: math.floor -> math.ceil; removed redundant squeeze
        # Parameters

        text : `TextFieldTensors`, required.
            The output of a `TextField` representing the text of
            the document.
        spans : `torch.IntTensor`, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a `ListField[SpanField]` of
            indices into the text of the document.
        span_labels : `torch.IntTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans), representing the cluster ids
            of each span, or -1 for those which do not appear in any clusters.
        metadata : `List[Dict[str, Any]]`, optional (default = `None`).
            A metadata dictionary for each instance in the batch. We use the "original_text" and "clusters" keys
            from this dictionary, which respectively have the original text and the annotated gold coreference
            clusters for that instance.

        # Returns

        An output dictionary consisting of:

        top_spans : `torch.IntTensor`
            A tensor of shape `(batch_size, num_spans_to_keep, 2)` representing
            the start and end word indices of the top spans that survived the pruning stage.
        antecedent_indices : `torch.IntTensor`
            A tensor of shape `(num_spans_to_keep, max_antecedents)` representing for each top span
            the index (with respect to top_spans) of the possible antecedents the model considered.
        predicted_antecedents : `torch.IntTensor`
            A tensor of shape `(batch_size, num_spans_to_keep)` representing, for each top span, the
            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
            was no predicted link.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        batch_size = spans.size(0)
        document_length = text_embeddings.size(1)
        num_spans = spans.size(1)

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text)

        # Shape: (batch_size, num_spans)
        span_mask = spans[:, :, 0] >= 0
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, document_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(
            contextualized_embeddings, spans
        )
        # Shape: (batch_size, num_spans, emebedding_size)
        attended_span_embeddings_list = []
        for b in range(0, num_spans, self.span_batch_size):
            e = min(b + self.span_batch_size, num_spans)
            attended_span_embeddings = self._attentive_span_extractor(
                text_embeddings, spans[:, b:e]
            )
            attended_span_embeddings_list.append(attended_span_embeddings)
        attended_span_embeddings = torch.cat(attended_span_embeddings_list, dim=1)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat(
            [endpoint_span_embeddings, attended_span_embeddings], -1
        )

        # Prune based on mention scores.
        num_spans_to_keep = int(math.ceil(self._spans_per_word * document_length))
        num_spans_to_keep = min(num_spans_to_keep, num_spans)

        # Shape: (batch_size, num_spans)
        span_mention_scores_list = []
        for b in range(0, num_spans, self.span_batch_size):
            e = min(b + self.span_batch_size, num_spans)
            span_mention_scores = self._mention_scorer(
                self._mention_feedforward(span_embeddings[:, b:e])
            ).squeeze(-1)
            span_mention_scores_list.append(span_mention_scores)
        span_mention_scores = torch.cat(span_mention_scores_list, dim=-1)

        # Shape: (batch_size, num_spans) for all 3 tensors
        top_span_mention_scores, top_span_mask, top_span_indices = util.masked_topk(
            span_mention_scores, span_mask, num_spans_to_keep
        )

        # Shape: (batch_size * num_spans_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select spans for each element in the batch.
        # This reformats the indices to take into account their
        # index into the batch. We precompute this here to make
        # the multiple calls to util.batched_index_select below more efficient.
        flat_top_span_indices = util.flatten_and_batch_shift_indices(
            top_span_indices, num_spans
        )

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(
            spans, top_span_indices, flat_top_span_indices
        )
        # Shape: (batch_size, num_spans_to_keep, embedding_size)
        top_span_embeddings = util.batched_index_select(
            span_embeddings, top_span_indices, flat_top_span_indices
        )

        # Compute indices for antecedent spans to consider.
        max_antecedents = min(self._max_antecedents, num_spans_to_keep)

        # Now that we have our variables in terms of num_spans_to_keep, we need to
        # compare span pairs to decide each span's antecedent. Each span can only
        # have prior spans as antecedents, and we only consider up to max_antecedents
        # prior spans. So the first thing we do is construct a matrix mapping a span's
        # index to the indices of its allowed antecedents.

        # Once we have this matrix, we reformat our variables again to get embeddings
        # for all valid antecedents for each span. This gives us variables with shapes
        # like (batch_size, num_spans_to_keep, max_antecedents, embedding_size), which
        # we can use to make coreference decisions between valid span pairs.

        # Shape: (batch_size, num_spans_to_keep, max_antecedents) for all 4 tensors
        if self._coarse_to_fine:
            pruned_antecedents = self._coarse_to_fine_pruning(
                top_span_embeddings,
                top_span_mention_scores,
                top_span_mask,
                max_antecedents,
            )
        else:
            pruned_antecedents = self._distance_pruning(
                top_span_embeddings, top_span_mention_scores, max_antecedents
            )

        # Further prune the results as we found this does not hurt the performance much
        # Shape: (batch_size, num_spans_to_keep, top) for all 4 tensors
        kept_antecedents = self.keep_top_antecedents(
            *pruned_antecedents, top=min(max_antecedents, self.max_antecedents_further)
        )
        (
            top_partial_coreference_scores,
            top_antecedent_mask,
            top_antecedent_offsets,
            top_antecedent_indices,
        ) = kept_antecedents

        flat_top_antecedent_indices = util.flatten_and_batch_shift_indices(
            top_antecedent_indices, num_spans_to_keep
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        top_antecedent_embeddings = util.batched_index_select(
            top_span_embeddings, top_antecedent_indices, flat_top_antecedent_indices
        )
        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        coreference_scores = self._compute_coreference_scores(
            top_span_embeddings,
            top_antecedent_embeddings,
            top_partial_coreference_scores,
            top_antecedent_mask,
            top_antecedent_offsets,
        )

        for _ in range(self._inference_order - 1):
            dummy_mask = top_antecedent_mask.new_ones(batch_size, num_spans_to_keep, 1)
            # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents,)
            top_antecedent_with_dummy_mask = torch.cat(
                [dummy_mask, top_antecedent_mask], -1
            )
            # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
            attention_weight = util.masked_softmax(
                coreference_scores,
                top_antecedent_with_dummy_mask,
                memory_efficient=True,
            )
            # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents, embedding_size)
            top_antecedent_with_dummy_embeddings = torch.cat(
                [top_span_embeddings.unsqueeze(2), top_antecedent_embeddings], 2
            )
            # Shape: (batch_size, num_spans_to_keep, embedding_size)
            attended_embeddings = util.weighted_sum(
                top_antecedent_with_dummy_embeddings, attention_weight
            )
            # Shape: (batch_size, num_spans_to_keep, embedding_size)
            top_span_embeddings = self._span_updating_gated_sum(
                top_span_embeddings, attended_embeddings
            )

            # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
            top_antecedent_embeddings = util.batched_index_select(
                top_span_embeddings, top_antecedent_indices, flat_top_antecedent_indices
            )
            # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
            coreference_scores = self._compute_coreference_scores(
                top_span_embeddings,
                top_antecedent_embeddings,
                top_partial_coreference_scores,
                top_antecedent_mask,
                top_antecedent_offsets,
            )

        # We now have, for each span which survived the pruning stage,
        # a predicted antecedent. This implies a clustering if we group
        # mentions which refer to each other in a chain.
        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_antecedents = coreference_scores.max(2)
        # Subtract one here because index 0 is the "no antecedent" class,
        # so this makes the indices line up with actual spans if the prediction
        # is greater than -1.
        predicted_antecedents -= 1

        output_dict = {
            "top_spans": top_spans,
            "antecedent_indices": top_antecedent_indices,
            "predicted_antecedents": predicted_antecedents,
        }
        if span_labels is not None:
            # Find the gold labels for the spans which we kept.
            # Shape: (batch_size, num_spans_to_keep, 1)
            pruned_gold_labels = util.batched_index_select(
                span_labels.unsqueeze(-1), top_span_indices, flat_top_span_indices
            )

            # Shape: (batch_size, num_spans_to_keep, max_antecedents)
            antecedent_labels = util.batched_index_select(
                pruned_gold_labels, top_antecedent_indices, flat_top_antecedent_indices
            ).squeeze(-1)
            antecedent_labels = util.replace_masked_values(
                antecedent_labels, top_antecedent_mask, -100
            )

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            gold_antecedent_labels = self._compute_antecedent_gold_labels(
                pruned_gold_labels, antecedent_labels
            )
            # Now, compute the loss using the negative marginal log-likelihood.
            # This is equal to the log of the sum of the probabilities of all antecedent predictions
            # that would be consistent with the data, in the sense that we are minimising, for a
            # given span, the negative marginal log likelihood of all antecedents which are in the
            # same gold cluster as the span we are currently considering. Each span i predicts a
            # single antecedent j, but there might be several prior mentions k in the same
            # coreference cluster that would be valid antecedents. Our loss is the sum of the
            # probability assigned to all valid antecedents. This is a valid objective for
            # clustering as we don't mind which antecedent is predicted, so long as they are in
            #  the same coreference cluster.
            coreference_log_probs = util.masked_log_softmax(
                coreference_scores, top_span_mask.unsqueeze(-1)
            )
            correct_antecedent_log_probs = (
                coreference_log_probs + gold_antecedent_labels.log()
            )
            negative_marginal_log_likelihood = -util.logsumexp(
                correct_antecedent_log_probs
            ).sum()

            self._mention_recall(top_spans, metadata)
            self._conll_coref_scores(
                top_spans, top_antecedent_indices, predicted_antecedents, metadata
            )

            output_dict["loss"] = negative_marginal_log_likelihood

        if metadata is not None:
            output_dict["document"] = [x["original_text"] for x in metadata]
        return output_dict

    @classmethod
    def from_extracted_archive(
        cls: Type[CoreferenceResolver], archive_content: ArchiveContent
    ) -> CoreferenceResolver:
        # Clone model file and extract:
        weight_path = archive_content.weight_path
        config = archive_content.config
        archive_dir = archive_content.archive_dir

        # Build vocab:
        vocab_dir = os.path.join(archive_dir, "vocabulary")
        vocab_params = config.get("vocabulary", Params({}))
        vocab_choice = vocab_params.pop_choice(
            "type", Vocabulary.list_available(), True
        )
        vocab_class, _ = Vocabulary.resolve_class_name(vocab_choice)
        vocab = vocab_class.from_files(
            vocab_dir, vocab_params.get("padding_token"), vocab_params.get("oov_token")
        )

        # Build model:
        model_params = config.get("model")
        kwargs = model_params.as_dict()
        ## Build mention_ffd:
        mention_ffd_kwargs = dict(kwargs["mention_feedforward"])
        mention_ffd_activations: List[str] = mention_ffd_kwargs.pop("activations")
        assert mention_ffd_activations == "relu", NotImplemented
        mention_feedforward = FeedForward(
            activations=[
                torch.nn.ReLU() for _ in range(mention_ffd_kwargs["num_layers"])
            ],
            **mention_ffd_kwargs,
        )
        ## Build antecedent_ffd:
        antecedent_ffd_kwargs = dict(kwargs["antecedent_feedforward"])
        antecedent_ffd_activations: List[str] = antecedent_ffd_kwargs.pop("activations")
        assert antecedent_ffd_activations == "relu", NotImplemented
        antecedent_feedforward = FeedForward(
            activations=[
                torch.nn.ReLU() for _ in range(antecedent_ffd_kwargs["num_layers"])
            ],
            **antecedent_ffd_kwargs,
        )
        ## Instantiate the model:
        model = CoreferenceResolver(
            vocab=vocab,
            text_field_embedder=BasicTextFieldEmbedder(
                {
                    "tokens": PretrainedTransformerMismatchedEmbedder(
                        **pop_and_return(
                            kwargs["text_field_embedder"]["token_embedders"]["tokens"],
                            to_pop="type",
                        )
                    )
                }
            ),
            context_layer=PassThroughEncoder(
                **pop_and_return(kwargs["context_layer"], to_pop="type")
            ),
            mention_feedforward=mention_feedforward,
            antecedent_feedforward=antecedent_feedforward,
            feature_size=kwargs["feature_size"],
            max_span_width=kwargs["max_span_width"],
            spans_per_word=kwargs["spans_per_word"],
            max_antecedents=kwargs["max_antecedents"],
            coarse_to_fine=kwargs["coarse_to_fine"],
            inference_order=kwargs["inference_order"],
            trained_with_coarse_to_fine=kwargs["coarse_to_fine"],
            # lexical_dropout=kwargs["lexical_dropout"],
            # initializer=InitializerApplicator(
            #     **pop_and_return(kwargs["initializer"], to_pop="type")
            # ),
        )
        model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=False)
        return model

    @classmethod
    def from_archive(
        cls: Type[CoreferenceResolver], archive_path: str
    ) -> CoreferenceResolver:
        archive_content = clone_and_extract(archive_path)
        return cls.from_extracted_archive(archive_content)


if __name__ == "__main__":
    from long_coref.utils import set_logger_format

    set_logger_format()

    CoreferenceResolver.from_archive(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    )
