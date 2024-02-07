from dataclasses import dataclass
import logging
import os
from typing import List, Union
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import extracted_archive, get_weights_path, CONFIG_NAME
from allennlp.common.params import Params


@dataclass
class ArchiveContent:
    archive_dir: str
    weight_path: str
    config: Params


def clone_and_extract(archive_path: str) -> ArchiveContent:
    # Clone model file and extract:
    cached_archive_path = cached_path(archive_path)
    with extracted_archive(cached_archive_path, cleanup=False) as archive_dir:
        pass
    weight_path = get_weights_path(archive_dir)
    config = Params.from_file(os.path.join(archive_dir, CONFIG_NAME))
    return ArchiveContent(
        archive_dir=archive_dir, weight_path=weight_path, config=config
    )


def pop_and_return(kwargs: dict, to_pop: Union[str, List[str]]) -> dict:
    kwargs = dict(kwargs)
    if type(to_pop) is str:
        to_pop = [to_pop]
    for key in to_pop:
        if key in kwargs:
            kwargs.pop(key)
        else:
            logging.info(f"Found no key {key} in kwargs")
    return kwargs
