from dataclasses import asdict
import json
import logging
from os import PathLike
import os
import time
from typing import List, Union
from allennlp.evaluation.evaluator import SimpleEvaluator as AllenNLPSimpleEvaluator
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler
from long_coref.utils import get_wiki_pages, parse_cli
from long_coref.args.coref_evaluation import CorefEvaluationArguments
from long_coref.coref.utils import clone_and_extract
from long_coref.coref.prediction import CorefPredictor, Resolution
from long_coref.coref.modeling import CoreferenceResolver
from os import PathLike
from pathlib import Path
import torch
import tqdm
from allennlp.common.util import dump_metrics, int_to_device
from allennlp.nn import util as nn_util


class SimpleEvaluator(AllenNLPSimpleEvaluator):
    def __call__(
        self,
        model: CoreferenceResolver,
        data_loader: MultiProcessDataLoader,
        metrics_output_file: Union[str, PathLike] = None,
        predictions_output_file: Union[str, PathLike] = None,
    ):
        """
        Evaluate a single data source.

        # Parameters

        model : `Model`
            The model to evaluate
        data_loader : `DataLoader`
            The `DataLoader` that will iterate over the evaluation data (data loaders already contain
            their data).
        batch_weight_key : `str`, optional (default=`None`)
            If given, this is a key in the output dictionary for each batch that specifies how to weight
            the loss for that batch.  If this is not given, we use a weight of 1 for every batch.
        metrics_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the final metrics to.
        predictions_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the predictions to.

        # Returns

        metrics: `Dict[str, Any]`
            The metrics from evaluating the file.
        """
        data_loader.set_target_device(int_to_device(self.cuda_device))
        metrics_output_file = (
            Path(metrics_output_file) if metrics_output_file is not None else None
        )
        if predictions_output_file is not None:
            predictions_file = Path(predictions_output_file).open("w", encoding="utf-8")
        else:
            predictions_file = None  # type: ignore

        model_postprocess_function = getattr(model, self.postprocessor_fn_name, None)

        with torch.no_grad():
            model.eval()
            iterator = iter(data_loader)
            logging.info("Iterating over dataset")
            generator_tqdm = tqdm.tqdm(iterator, total=len(data_loader))
            # Number of batches in instances.
            batch_count = 0
            # Number of batches where the model produces a loss.
            loss_count = 0
            # Cumulative weighted loss
            total_loss = 0.0

            for batch in generator_tqdm:
                batch_count += 1
                batch = nn_util.move_to_device(batch, self.cuda_device)
                output_dict: dict = model(**batch)
                loss: torch.Tensor = output_dict.get("loss")

                metrics = model.get_metrics()

                if loss is not None:
                    loss_count += 1
                    total_loss += loss.item()
                    # Report the average loss so far.
                    metrics["loss"] = total_loss / loss_count

                description = (
                    ", ".join(
                        [
                            "%s: %.2f" % (name, value)
                            for name, value in metrics.items()
                            if not name.startswith("_")
                        ]
                    )
                    + " ||"
                )
                generator_tqdm.set_description(description, refresh=False)

                # TODO(gabeorlanski): Add in postprocessing the batch for token
                #  metrics
                if predictions_file is not None:
                    predictions_file.write(
                        self.batch_serializer(
                            batch,
                            output_dict,
                            data_loader,
                            output_postprocess_function=model_postprocess_function,
                        )
                        + "\n"
                    )

            if predictions_file is not None:
                predictions_file.close()

            final_metrics = model.get_metrics(reset=True)
            if loss_count > 0:
                # Sanity check
                if loss_count != batch_count:
                    raise RuntimeError(
                        "The model you are trying to evaluate only sometimes produced a loss!"
                    )
                final_metrics["loss"] = total_loss / loss_count

            if metrics_output_file is not None:
                dump_metrics(str(metrics_output_file), final_metrics, log=True)

            return final_metrics


def long_document_prediction(predictor: CorefPredictor, output_dir: str) -> None:
    wiki_pages = get_wiki_pages(
        [
            "The Half Moon, Putney",
            "Mahatma Gandhi",
            "Mansa Musa",
            "Bill Clinton",
            "J. K. Rowling",
            "Bono",
            "Mark Antony",
            "Germany",
            "Great Pyramid of Giza",
            "North Korea",
            "Westminster Abbey",
            "Aspirin",
            "Diesel engine",
            "Guinea pig",
            "Polio",
            "Spanish flu",
            "Windows XP",
            "King K. Rool",
            "Sailor Moon (character)",
            "Cello",
            "FIFA",
            "Mueller report",
        ]
    )
    resolutions: List[Resolution] = []
    start = time.time()
    for page in tqdm.tqdm(wiki_pages, desc="Resolving documents"):
        resolution = predictor.resolve_paragraphs(page.paragraphs)
        resolutions.append(resolution)
    duration = time.time() - start
    os.makedirs(output_dir, exist_ok=True)
    for page, resolution in zip(wiki_pages, resolutions):
        resolution.dump_html(os.path.join(output_dir, f"{page.title}.html"))
    nwords = sum(len(resolution.document) for resolution in resolutions)
    speed_str = f"{nwords/duration:.1f} w/s"
    logging.info(f"Speed: {speed_str}")
    with open(os.path.join(output_dir, "long-document-speed.json"), "w") as f:
        json.dump(
            {
                "nwords": nwords,
                "duration": duration,
                "ndocs": len(wiki_pages),
                "speed": speed_str,
            },
            f,
            indent=4,
        )


def run_evaluation() -> None:
    args = parse_cli(CorefEvaluationArguments)
    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output folder: {output_dir}")
    archive_content = clone_and_extract(args.allennlp_model_path)
    predictor = CorefPredictor.from_extracted_archive(archive_content)
    predictor.alter_hyperparameters(
        spans_per_word=args.spans_per_word,
        max_antecedents=args.max_antecedents,
        max_span_width=args.max_span_width,
        coarse_to_fine=args.coarse_to_fine,
        max_antecedents_further=args.max_antecedents_further,
        span_batch_size=args.span_batch_size,
    )
    model: CoreferenceResolver = predictor._model
    dataset_reader = predictor._dataset_reader
    evaluator = SimpleEvaluator(cuda_device=predictor.cuda_device)
    batch_sampler = BucketBatchSampler(
        batch_size=1, sorting_keys=["text"], shuffle=False
    )
    data_loader = MultiProcessDataLoader(
        reader=dataset_reader,
        data_path=args.ontonotes_eval_path,
        batch_sampler=batch_sampler,
    )
    data_loader.index_with(model.vocab)
    metrics = evaluator(model, data_loader)
    logging.info(json.dumps(metrics, indent=4))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(asdict(args), f, indent=4)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    if args.long_doc:
        long_document_prediction(predictor=predictor, output_dir=output_dir)


if __name__ == "__main__":
    from long_coref.utils import set_logger_format

    set_logger_format()
    run_evaluation()


#### span-large _coarse_to_fine == True ####
# [2023-10-10 15:36:25] INFO [root.set_device:178] Using device cuda
# loading instances: 348it [00:10, 34.11it/s]
# [2023-10-10 15:36:43] INFO [root.__call__:66] Iterating over dataset
# 0it [00:00, ?it/s]/home/fb20user07/miniconda3/envs/lcfr/lib/python3.8/site-packages/allennlp/modules/token_embedders/pretrained_transformer_embedder.py:385: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
#   num_effective_segments = (seq_lengths + self._max_length - 1) // self._max_length
# coref_precision: 0.79, coref_recall: 0.78, coref_f1: 0.79, mention_recall: 0.97, loss: 99.68 ||: : 348it [00:50,  6.93it/s]
# [2023-10-10 15:37:33] INFO [root.run_evaluation:152] {
#     "coref_precision": 0.7920223281022736,
#     "coref_recall": 0.7836041369661952,
#     "coref_f1": 0.7877344649263414,
#     "mention_recall": 0.9678661936137861,
#     "loss": 99.68152958758388
# }

#### span-large _coarse_to_fine == False ####
# coref_precision: 0.78, coref_recall: 0.70, coref_f1: 0.74, mention_recall: 0.97, loss: 94.25 ||: 100%|█████████████████████| 348/348 [00:54<00:00,  6.35it/s]
# [2023-10-10 23:42:04] INFO [root.run_evaluation:143] {
#     "coref_precision": 0.7763389531660714,
#     "coref_recall": 0.7044847962905217,
#     "coref_f1": 0.7380218176446167,
#     "mention_recall": 0.9678661936137861,
#     "loss": 94.25351154385454
# }

#### span-large _coarse_to_fine == False, top == 1 ####
# coref_precision: 0.78, coref_recall: 0.69, coref_f1: 0.73, mention_recall: 0.97, loss: 77.54 ||: 100%|████████████████████████████████████████████████| 348/348 [00:36<00:00,  9.55it/s]
# [2023-10-11 20:51:09] INFO [root.run_evaluation:146] {
#     "coref_precision": 0.7787969173250571,
#     "coref_recall": 0.6938900781392183,
#     "coref_f1": 0.733175862070316,
#     "mention_recall": 0.9679214733859542,
#     "loss": 77.54290693808416
# }

#### span-large _coarse_to_fine == True _max_antecedents == 1 ####
# coref_precision: 0.80, coref_recall: 0.76, coref_f1: 0.78, mention_recall: 0.97, loss: 87.25 ||: 100%|█████████████████| 348/348 [00:33<00:00, 10.49it/s]
# [2023-10-11 18:14:06] INFO [root.run_evaluation:144] {
#     "coref_precision": 0.7954121789256522,
#     "coref_recall": 0.7630793831836146,
#     "coref_f1": 0.7788768111717247,
#     "mention_recall": 0.9679214733859542,
#     "loss": 87.24771279292345
# }

#### span-base trained with _coarse_to_fine == False ####
# coref_precision: 0.75, coref_recall: 0.69, coref_f1: 0.72, mention_recall: 0.96, loss: 83.64 ||: 100%|█████████████████| 348/348 [00:35<00:00,  9.90it/s]
# [2023-10-11 13:02:15] INFO [root.run_evaluation:143] {
#     "coref_precision": 0.7506870085831426,
#     "coref_recall": 0.691400529505569,
#     "coref_f1": 0.718976913234143,
#     "mention_recall": 0.9644196654840345,
#     "loss": 83.63996403610619
# }

#### span-base trained with _coarse_to_fine == True and _coarse_to_fine set False ####
# coref_precision: 0.75, coref_recall: 0.69, coref_f1: 0.72, mention_recall: 0.96, loss: 92.85 ||: 100%|█████████████████| 348/348 [00:34<00:00,  9.97it/s]
# [2023-10-11 13:30:50] INFO [root.run_evaluation:143] {
#     "coref_precision": 0.7491860683166246,
#     "coref_recall": 0.6872522749381492,
#     "coref_f1": 0.7162508073292151,
#     "mention_recall": 0.9646832624974702,
#     "loss": 92.84715273100873
# }

#### span-base trained with _coarse_to_fine == True and _coarse_to_fine set True ####
# coref_precision: 0.76, coref_recall: 0.77, coref_f1: 0.76, mention_recall: 0.96, loss: 97.87 ||: 100%|█████████████████| 348/348 [00:38<00:00,  8.93it/s]
# [2023-10-11 13:34:35] INFO [root.run_evaluation:143] {
#     "coref_precision": 0.7590998697539607,
#     "coref_recall": 0.7655309674513454,
#     "coref_f1": 0.7622506089625696,
#     "mention_recall": 0.9646832624974702,
#     "loss": 97.86582714944176
# }
