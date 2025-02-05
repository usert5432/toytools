"""Functions to load toyzero image dataset"""

import os
import re

from typing import List, Tuple, Optional, Set, Union
import numpy as np

from .consts import DIR_FAKE, DIR_REAL

def find_images_in_dir(path : str) -> List[str]:
    """Return sorted list of '*.npz' files in `path`"""
    result = []

    for fname in os.listdir(path):
        fullpath = os.path.join(path, fname)

        if not os.path.isfile(fullpath):
            continue

        ext = os.path.splitext(fname)[1]
        if ext != '.npz':
            continue

        result.append(fname)

    result.sort()
    return result

def validate_toyzero_images(imgs_fake : List[str], imgs_real : List[str]):
    """Verify that image names are the same in fake and real image lists"""

    if len(imgs_fake) != len(imgs_real):
        raise RuntimeError(
            "Number of real/fake images does not match: %d != %d" % (
                len(imgs_fake), len(imgs_real)
            )
        )

    for (img_a,img_b) in zip(imgs_fake, imgs_real):
        if img_a != img_b:
            raise RuntimeError(
                "Fake/Real image name mismatch: %s != %s" % (img_a, img_b)
            )

def collect_toyzero_images(root : str) -> List[str]:
    """Return a list of `toyzero` image names found in `root`"""

    path_fake = os.path.join(root, DIR_FAKE)
    path_real = os.path.join(root, DIR_REAL)

    imgs_fake = find_images_in_dir(path_fake)
    imgs_real = find_images_in_dir(path_real)

    validate_toyzero_images(imgs_fake, imgs_real)

    return imgs_fake

def parse_images(images : List[str]) -> List[Tuple[str, str, int, str]]:
    """Parse image names to infer Event, APA and Wire Plane"""
    result = []

    pattern = re.compile(r'^(.+)-(\d+)-([UVW])\.npz$')

    for image in images:
        match = pattern.match(image)

        if not match:
            raise RuntimeError("Cannot parse image name: %s" % image)

        base  = match.group(1)
        apa   = int(match.group(2))
        plane = match.group(3)

        result.append((image, base, apa, plane))

    return result

def filter_parsed_images(
    parsed_images : List[Tuple[str, str, int, str]],
    apas          : Optional[Set[int]] = None,
    planes        : Optional[Set[str]] = None,
) -> List[Tuple[str, str, int, str]]:
    """Filter parsed images list based on APAs and Wire Planes"""

    if apas is not None:
        parsed_images = [ x for x in parsed_images if x[2] in apas ]

    if planes is not None:
        parsed_images = [ x for x in parsed_images if x[3] in planes ]

    return parsed_images

def filter_images(
    images : List[str],
    apas   : Optional[Set[int]] = None,
    planes : Optional[Set[str]] = None,
) -> List[str]:
    """Filter images list based on APAs and Wire Planes"""

    if (planes is None) and (apas is None):
        return images

    parsed_images = parse_images(images)
    parsed_images = filter_parsed_images(parsed_images, apas, planes)

    return [ x[0] for x in parsed_images ]

def load_image(root : str, is_fake : bool, name : str) -> np.ndarray:
    """Load image `name`"""

    if is_fake:
        subdir = DIR_FAKE
    else:
        subdir = DIR_REAL

    path = os.path.join(root, subdir, name)

    with np.load(path) as f:
        return f[f.files[0]]

def train_val_test_split(
    n         : int,
    val_size  : Union[int, float],
    test_size : Union[int, float],
    shuffle   : bool,
    prg       : np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset of size `n` into training/val/test parts.

    Parameters
    ----------
    n : int
        Size of the dataset to split.
    val_size : int or float
        Fraction of the dataset that will be used as a val sample.
        If `val_size` <= 1, then it is treated as a fraction, i.e.
        (size of test sample) = `val_size` * (size of toyzero dataset)
        Otherwise, (size of val sample) = `val_size`.
    test_size : int or float
        Fraction of the dataset that will be used as a test sample.
        If `test_size` <= 1, then it is treated as a fraction, i.e.
        (size of test sample) = `test_size` * (size of toyzero dataset)
        Otherwise, (size of test sample) = `test_size`.
    shuffle : bool
        Whether to shuffle dataset
    prg : np.random.Generator
        RNG that will be used for shuffling the dataset.
        This parameter has no effect if `shuffle` is False.

    Returns
    -------
    (train_indices, val_indices, test_indices) : (ndarray, ndarray, ndarray)
        Indices of training/val/test samples.

    """
    indices = np.arange(n)

    if shuffle:
        prg.shuffle(indices)

    if test_size <= 1:
        test_size = len(indices) * test_size

    if val_size <= 1:
        val_size = len(indices) * val_size

    test_size  = int(test_size)
    val_size   = int(val_size)
    train_size = max(0, len(indices) - val_size - test_size)

    train_indices = indices[:train_size]
    val_indices   = indices[train_size:train_size+val_size]
    test_indices  = indices[train_size+val_size:]

    return (train_indices, val_indices, test_indices)

def train_test_split(
    n         : int,
    test_size : Union[int, float],
    shuffle   : bool,
    prg       : np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split dataset of size `n` into training/test parts.

    Parameters
    ----------
    n : int
        Size of the dataset to split.
    test_size : int or float
        Fraction of the dataset that will be used as a test sample.
        If `test_size` <= 1, then it is treated as a fraction, i.e.
        (size of test sample) = `test_size` * (size of toyzero dataset)
        Otherwise, (size of val sample) = `test_size`.
    shuffle : bool
        Whether to shuffle dataset
    prg : np.random.Generator
        RNG that will be used for shuffling the dataset.
        This parameter has no effect if `shuffle` is False.

    Returns
    -------
    (train_indices, test_indices) : (np.ndarray, np.ndarray)
        Indices of training and validation samples.

    """

    train_indices, _val_indices, test_indices = train_val_test_split(
        n, val_size = 0, test_size = test_size, shuffle = shuffle, prg = prg
    )

    return (train_indices, test_indices)

