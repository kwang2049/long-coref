import logging
import os
import pickle
import random
from typing import Iterable, Iterator, List, Tuple, Type, TypeVar
import git
from git.exc import InvalidGitRepositoryError
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClassType

OBJ_TYPE = TypeVar("OBJ_TYPE")


def set_logger_format(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level)
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)


def all_gather_object(obj: OBJ_TYPE) -> List[OBJ_TYPE]:
    """Broadcast a Python object across all the nodes."""
    if not dist.is_initialized():
        return [obj]

    # Collect object from replicas and form a list
    ngpus = dist.get_world_size()
    obj_list = [None for _ in range(ngpus)]
    dist.all_gather_object(obj_list, obj)
    return obj_list


def broadcast_object(obj: OBJ_TYPE) -> OBJ_TYPE:
    """Broadcast a Python object across all the nodes."""
    if not dist.is_initialized():
        return obj

    rank = dist.get_rank()
    obj_size: torch.LongTensor = torch.zeros(1).long().to(rank)  # long not int!
    if rank == 0:
        data = pickle.dumps(obj)
        data_length = len(data)
        data = data_length.to_bytes(4, "big") + data
        data = np.frombuffer(data, dtype=np.uint8)
        obj_size += len(data)
        tensorized: torch.Tensor = torch.from_numpy(data).to(rank)
        logging.info(f"Going to broacast {obj_size.item() / 2**20:.1f}MB")
    dist.broadcast(obj_size, 0)
    if rank != 0:
        tensorized = torch.zeros(obj_size.item(), dtype=torch.uint8).to(rank)
    dist.broadcast(tensorized, 0)
    tensorized_numpy: np.ndarray = tensorized.cpu().numpy()
    data = tensorized_numpy.tobytes()
    del tensorized
    torch.cuda.empty_cache()
    length = int.from_bytes(data[:4], "big")
    data = data[4 : length + 4]
    obj: OBJ_TYPE = pickle.loads(data)
    return obj


def is_device_zero() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0"


def get_commit_hash() -> str:
    """Return the HEAD commit hash."""
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except InvalidGitRepositoryError:
        return "no_git"


def unravel_index(indices: torch.LongTensor, shape: torch.Size) -> list:
    """Modified from https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3"""
    assert len(indices.shape) == 1, "The indices of the flatten matrix must be 1-D"
    coordinates: List[torch.LongTensor] = []  # num_dimensions * (num_indices,)
    for dim in reversed(shape):
        coordinates.append(indices % dim)
        indices = indices // dim
    stacked_coordinates = torch.stack(
        list(reversed(coordinates)), dim=0
    )  # (num_dimensions, num_indices)
    return stacked_coordinates.tolist()


def balanced_bce_loss(
    pred_probs: torch.FloatTensor, target: torch.FloatTensor
) -> torch.FloatTensor:
    losses: torch.Tensor = torch.nn.functional.binary_cross_entropy(
        pred_probs, target, reduction="none"
    )
    return (losses[target.bool()].mean() + losses[~target.bool()].mean()) / 2


def initialize_ddp(ddp_timeout: int = 360000) -> None:
    from transformers import TrainingArguments

    TrainingArguments(
        "dummy", ddp_timeout=ddp_timeout
    )  # This will invoke dist.init_process_group
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_env.split(",")[get_rank()]
    assert dist.is_initialized()


def sequence_mask(lengths: List[int]) -> torch.BoolTensor:
    """https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036/5"""
    return torch.arange(max(lengths)).unsqueeze(0) < torch.LongTensor(
        lengths
    ).unsqueeze(1)


def split_data(data: Iterable[OBJ_TYPE]) -> Iterable[OBJ_TYPE]:
    if not dist.is_initialized():
        for datum in data:
            yield datum

    rank = dist.get_rank()
    ngpus = dist.get_world_size()
    for i, datum in enumerate(data):
        if i % ngpus == rank:
            yield datum


def split_data_size(data_size: int) -> int:
    if not dist.is_initialized():
        return data_size

    rank = dist.get_rank()
    ngpus = dist.get_world_size()
    return data_size // ngpus + int(rank < data_size % ngpus)


def parse_cli(arguments_class: Type[OBJ_TYPE]) -> OBJ_TYPE:
    parser = HfArgumentParser(arguments_class)
    args = arguments_class(**vars(parser.parse_args()))
    return args


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_ngpus() -> int:
    if not dist.is_initialized():
        return int(torch.cuda.is_available())
    return dist.get_world_size()


from dataclasses import dataclass
import re
from typing import List

import requests


@dataclass
class WikiPage:
    title: str
    paragraphs: List[str]


def get_wiki_pages(titles: List[str]) -> List[WikiPage]:
    wiki_pages: List[WikiPage] = []
    for title in tqdm.tqdm(titles, "Downloading wiki pages"):
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                # 'exintro': True,
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        title: str = page["title"]
        wiki_text: str = page["extract"]
        paragraphs = wiki_text.split("\n\n\n")
        paragraphs_wo_section_titles = []
        for paragraph in paragraphs:
            p = re.sub("=+\s.*\s=+", "", paragraph).strip()
            if len(p):
                paragraphs_wo_section_titles.append(p.replace("\n", ""))
        wiki_pages.append(
            WikiPage(title=title, paragraphs=paragraphs_wo_section_titles)
        )
    return wiki_pages


def color_generator() -> Iterable[str]:
    random_state = random.Random(42)
    while True:
        rgb = [str(random_state.randint(0, 255)) for _ in range(3)]
        yield "rgb(" + ",".join(rgb) + ")"


def expand_and_roll(
    seq_embs: torch.Tensor, shifts: torch.LongTensor, dim: int
) -> torch.Tensor:
    """
    Expand the sequence of embeddings for the specified dim |shifts| times and apply the shift at each cloned sequence.
    :param seq_embs: e.g. of shape (bsz, seq_length, hdim)
    :param shifts: 1-D tensor
    :param dim: the sequence length dim
    :return: e.g. of shape (bsz, seq_length, |shifts|, hdim)
    """
    # seq_embs: (bsz, seq_length, hdim)
    # dim == 1
    dim_insertion = dim + 1  # 2
    shape_list = list(seq_embs.shape)
    shape_list.insert(dim_insertion, len(shifts))
    expanded_seq_embs = seq_embs.unsqueeze(dim=dim_insertion).expand(
        shape_list
    )  # (bsz, seq_length, |shifts|, hdim)
    indices_shape = [
        1 if dim_i != dim_insertion else len(shifts)
        for dim_i in range(expanded_seq_embs.ndim)
    ]  # (1, 1, |shifts|, hdim)
    indices = (
        torch.arange(len(shifts)).reshape(indices_shape).expand_as(expanded_seq_embs)
    )
    seq_length = seq_embs.shape[dim]
    shifts_shape = [
        1 if dim_i != dim else seq_length for dim_i in range(expanded_seq_embs.ndim)
    ]  # (1, seq_length, 1, 1)
    shifts = torch.arange(seq_length).reshape(shifts_shape).expand_as(expanded_seq_embs)
    shifted_indices = (indices - shifts) % seq_length
    return expanded_seq_embs.gather(
        dim=dim, index=shifted_indices
    )  # (bsz, seq_length, |shifts|, hdim)


def log_gpu_memory() -> None:
    total: int = getattr(torch.cuda.get_device_properties(0), "total_memory")
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    logging.info(
        f"GPU memory: total {total/2**30:.2f}GB, reserved {reserved/2**30:.2f}GB, allocated {allocated/2**30:.2f}GB"
    )


def nlines(fpath: str) -> int:
    """Count how many lines in a file."""
    with open(fpath, "r") as f:
        return sum(1 for _ in f)


def tqdm_ropen(fpath: str, desc: str = None, disable: bool = False) -> Iterator[str]:
    """tqdm + open with r mode."""
    if desc is None:
        desc = f"Loading from {fpath}"

    with tqdm.tqdm(open(fpath, "r"), desc, nlines(fpath), disable=disable) as f:
        for line in f:
            yield line
