from pathlib import Path

import cv2
import h5py
import numpy as np
import numpy.typing as npt

from .parsers import names_to_pair, names_to_pair_old


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR

    image = cv2.imread(str(path), mode | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')

    if not grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), 'r', libver='latest') as fd:

        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))

        fd.visititems(visit_fn)
    return list(set(names))


def get_descriptors(path: Path, name: str) -> npt.NDArray[np.floating]:
    with h5py.File(str(path), mode='r', libver='latest') as h5:
        dset = h5[name]['descriptors']
        descriptors = np.array(dset)

    return np.transpose(descriptors, axes=(1, 0))


def get_keypoints(path: Path, name: str, return_uncertainty: bool = False) -> npt.NDArray[np.float64]:
    with h5py.File(str(path), mode='r', libver='latest') as h5:
        dset = h5[name]['keypoints']
        kp = np.array(dset, dtype=np.float64)
        uncertainty = dset.attrs.get('uncertainty')

    if return_uncertainty:
        return kp, uncertainty

    return kp


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(f'Could not find pair {(name0, name1)}... Maybe you matched with a different list of pairs? ')


def get_matches(path: Path, name0: str, name1: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(str(path), mode='r', libver='latest') as h5:
        pair, reverse = find_pair(h5, name0, name1)
        matches = np.asarray(h5[pair]['matches0'], dtype=np.int32)
        scores = np.asarray(h5[pair]['matching_scores0'], dtype=np.float64)

    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], axis=-1)
    scores = scores[idx]

    if reverse:
        matches = np.flip(matches, axis=-1)

    return matches, scores
