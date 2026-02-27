import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

import numpy as np
import pycolmap

logger = logging.getLogger(__name__)


def parse_image_list(path, with_intrinsics=False):
    images = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            if with_intrinsics:
                model, width, height, *params = data
                params = np.array(params, float)
                cam = pycolmap.Camera(model=model, width=int(width), height=int(height), params=params)
                images.append((name, cam))
            else:
                images.append(name)

    assert len(images) > 0
    logger.info(f'Imported {len(images)} images from {path.name}')
    return images


def parse_image_lists(paths, with_intrinsics=False):
    images = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0
    for lfile in files:
        images += parse_image_list(lfile, with_intrinsics=with_intrinsics)
    return images


def parse_retrieval(path: Path) -> Dict[str, List[str]]:
    pairs = parse_pairs(path)
    retrieval = defaultdict(list)
    for q, r in pairs:
        retrieval[q].append(r)

    return dict(retrieval)


def parse_pairs(path: Path) -> List[Tuple[str, str]]:
    if path.suffix == '.txt':
        with open(path, mode='r') as f:
            pairs = []
            for line in f:
                line = line.strip('\n')
                if len(line) == 0 or line[0] == '#':
                    continue
                name0, name1 = line.split()
                pairs.append((name0, name1))
    elif path.suffix == '.json':
        with open(path, mode='r', encoding='utf-8') as f:
            pairs = json.load(f)
    else:
        raise ValueError(f'Unsupported pairs file format: {path.suffix}')

    return pairs


def names_to_pair(name0: str, name1: str, separator: str = '/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))


def names_to_pair_old(name0: str, name1: str):
    return names_to_pair(name0, name1, separator='_')
