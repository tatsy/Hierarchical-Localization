import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import h5py
import numpy as np
import torch
from tqdm import tqdm

from . import logger, matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval

"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
"""
confs = {
    'superpoint+lightglue': {
        'output': 'matches-superpoint-lightglue',
        'model': {
            'name': 'lightglue',
            'features': 'superpoint',
        },
    },
    'disk+lightglue': {
        'output': 'matches-disk-lightglue',
        'model': {
            'name': 'lightglue',
            'features': 'disk',
        },
    },
    'aliked+lightglue': {
        'output': 'matches-aliked-lightglue',
        'model': {
            'name': 'lightglue',
            'features': 'aliked',
        },
    },
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'superglue-fast': {
        'output': 'matches-superglue-it5',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 5,
        },
    },
    'NN-superpoint': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
    },
    'NN-ratio': {
        'output': 'matches-NN-mutual-ratio.8',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
        },
    },
    'NN-mutual': {
        'output': 'matches-NN-mutual',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
        },
    },
    'adalam': {
        'output': 'matches-adalam',
        'model': {'name': 'adalam'},
    },
}


class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: list[tuple[str, str]], feature_path_q: Path, feature_path_r: Path):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx: int):
        name0, name1 = self.pairs[idx]

        data: dict[str, Any] = {'name0': name0, 'name1': name1}
        with h5py.File(self.feature_path_q, mode='r') as fd:
            grp = fd[name0]
            assert isinstance(grp, h5py.Group)

            for k, v in grp.items():
                data[k + '0'] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            image_size = grp.get('image_size')
            assert isinstance(image_size, h5py.Dataset)

            data['image0'] = torch.empty((1,) + tuple(image_size)[::-1])

        with h5py.File(self.feature_path_r, mode='r') as fd:
            grp = fd[name1]
            assert isinstance(grp, h5py.Group)

            for k, v in grp.items():
                data[k + '1'] = torch.from_numpy(v.__array__()).float()

            image_size = grp.get('image_size')
            assert isinstance(image_size, h5py.Dataset)

            data['image1'] = torch.empty((1,) + tuple(image_size)[::-1])

        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), mode='a', libver='latest') as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred['matches0'][0].astype(np.int16)
        grp.create_dataset('matches0', data=matches)
        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].astype(np.float16)
            grp.create_dataset('matching_scores0', data=scores)


def main(
    conf: dict[str, Any],
    pairs: Path,
    features: Path | str,
    export_dir: Path | None = None,
    matches: Path | None = None,
    features_ref: Path | None = None,
    batch_size: int = 4,
    overwrite: bool = False,
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = Path(features)
        if matches is None:
            raise ValueError('Either provide both features and matches as Path or both as names.')
    else:
        if export_dir is None:
            raise ValueError(f'Provide an export_dir if features is not a file path: {features}.')
        features_q = Path(export_dir, features + '.h5')
        if matches is None:
            matches = Path(export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q

    match_from_paths(conf, pairs, matches, features_q, features_ref, batch_size=batch_size, overwrite=overwrite)

    return matches


def find_unique_new_pairs(pairs_all: list[tuple[str, str]], match_path: Path | None = None):
    """Avoid to recompute duplicates to save time."""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))

    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), 'r', libver='latest') as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                    names_to_pair(i, j) in fd
                    or names_to_pair(j, i) in fd
                    or names_to_pair_old(i, j) in fd
                    or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(
    conf: dict[str, Any],
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    batch_size: int = 4,
    overwrite: bool = False,
) -> None:
    logger.info(f'Matching local features with configuration:\n{pprint.pformat(conf)}')

    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    if not feature_path_ref.exists():
        raise FileNotFoundError(f'Reference feature file {feature_path_ref}.')
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f'Using GPU ({torch.cuda.get_device_name(0)}) for feature matching.')
    else:
        device = torch.device('cpu')
        logger.info('Using CPU for feature matching.')

    Matcher = dynamic_load(matchers, conf['model']['name'])
    model = Matcher(conf['model']).eval().to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 4)

    for data in tqdm(loader, desc='Matching features'):
        B = len(data['name0'])
        inputs = {}
        for k, v in data.items():
            if k.startswith('image') or k.startswith('name'):
                inputs[k] = v
            else:
                inputs[k] = v.to(device)

        outputs = model(inputs)
        for b in range(B):
            pred = {k: v[b].cpu().numpy() for k, v in outputs.items() if k != 'stop'}
            pred['stop'] = outputs['stop']
            pair = names_to_pair(data['name0'][b], data['name1'][b])
            writer_queue.put((pair, pred))

    writer_queue.join()

    del model

    logger.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str, default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue', choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
