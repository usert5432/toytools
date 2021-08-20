# pylint: disable=missing-module-docstring
import os
import numpy as np
import pandas as pd

from toytools.collect   import load_image, train_test_split
from toytools.transform import crop_image
from .generic_dataset   import GenericDataset

COLUMNS = [
    'image', 'event', 'apa', 'plane', 'x', 'y', 'width', 'height', 'bkg'
]

class PreUnalignedToyzeroDataset(GenericDataset):
    """Toyzero Dataset that loads images according to preprocessed list.

    When `is_train` is set to True, then this dataset will return unpaired
    images. Otherwise, it will return paired images.

    Parameters
    ----------
    path : str
        Path where the toyzero dataset is located.
    fname : str
        Name of the file containing preprocessed list of toyzero images.
        This should be located under `path`.
    is_train : bool
        `is_train` flag controls overall behavior of the dataset.
        If `is_train` is True, then the cropped regions of toyzero images are
        selected at random. Otherwise, the cropped regions are selected in a
        reproducible manner using `seed` parameter.  This flag also controls
        which part of the entire toyzero dataset this object will focus on.
        Default: False
    shuffle : bool
        Controls whether to shuffle the dataset before splitting it into
        training and validation parts.
        Default: True
    seed : int
        Seed used to initialize random number generators.
        Default: 0
    transform : Callable or None,
        Optional transformation to apply to images.
        E.g. torchvision.transforms.RandomCrop.
        Default: None
    val_size : float or int
        Fraction of the toyzero dataset that will be used as a validation
        sample. If val_size <= 1, then it is treated as a fraction, i.e.
        (size of val sample) = `val_size` * (size of toyzero dataset)
        Otherwise, (size of val sample) = `val_size`.
        Default: 0.2
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self, path,
        fname            = None,
        is_train         = False,
        seed             = 0,
        shuffle          = True,
        transform        = None,
        val_size         = 0.2,
    ):
        super().__init__(path)

        self._is_train  = is_train
        self._seed      = seed
        self._transform = transform
        self._prg       = np.random.default_rng(seed)
        self._val_size  = val_size

        self._df_fake = None
        self._df_real = None

        df = pd.read_csv(os.path.join(path, fname), index_col = 'index')
        self._split_dataset(df, shuffle)

    def _split_dataset(self, df, shuffle):
        """Split dataset into training/validation parts."""
        train_indices, val_indices = train_test_split(
            len(df), self._val_size, shuffle, self._prg
        )

        if self._is_train:
            indices_fake = train_indices.copy()
            indices_real = train_indices

            self._prg.shuffle(indices_fake)
            self._prg.shuffle(indices_real)
        else:
            indices_fake = val_indices
            indices_real = val_indices

        self._df_fake = df.iloc[indices_fake]
        self._df_real = df.iloc[indices_real]

    def __len__(self):
        return len(self._df_fake)

    def __getitem__(self, index):
        sample_fake = self._df_fake.iloc[index]
        sample_real = self._df_real.iloc[index]

        image_fake = load_image(self._path, True,  sample_fake.image)
        image_real = load_image(self._path, False, sample_real.image)

        images  = [image_fake,  image_real]
        samples = [sample_fake, sample_real]

        images = [
            crop_image(img, (s.x, s.y, s.width, s.height))
                for (img, s) in zip(images, samples)
        ]
        images = [ x - s.bkg            for (x, s) in zip(images, samples) ]
        images = [ x.astype(np.float32) for x in images ]

        if self._transform is not None:
            images = [ self._transform(x) for x in images ]

        return images

