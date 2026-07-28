"""Loading the 20 Newsgroups corpus.

The dataset is fetched through scikit-learn, which caches it as a single
compressed pickle under data/raw. The cache is committed-around (gitignored),
so the first run downloads ~15 MB and every run afterwards is instant and
offline - CI and tests never touch the network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.datasets import fetch_20newsgroups

from . import config

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """A train/test bundle plus the human-readable class names.

    Bundling them keeps the label order (0..19) tied to its names in one place,
    so nothing downstream has to re-derive which integer means which newsgroup.
    """

    X_train: List[str]
    y_train: List[int]
    X_test: List[str]
    y_test: List[int]
    target_names: List[str]

    @property
    def n_classes(self) -> int:
        return len(self.target_names)


def load_dataset(remove: Tuple[str, ...] = config.REMOVE_PARTS) -> Dataset:
    """Return the official by-date train/test split.

    Args:
        remove: which leaky parts of each post to strip. Defaults to the full
            set (headers, footers, quotes); train.py calls it with `()` to
            measure how much those parts inflate the score.

    Why the official split rather than a fresh shuffle: the 20 Newsgroups test
    set is a *later* time slice than the training set, which is exactly the
    situation a deployed classifier faces. Reshuffling would leak future posts
    into training and flatter the metrics - the same mistake, one level down,
    that a random split makes on time-ordered tabular data.
    """
    logger.info("Loading 20 Newsgroups (remove=%s) ...", remove or "nothing")

    train = fetch_20newsgroups(
        subset="train",
        remove=remove,
        random_state=config.RANDOM_STATE,
        data_home=str(config.RAW_DATA_DIR),
    )
    test = fetch_20newsgroups(
        subset="test",
        remove=remove,
        random_state=config.RANDOM_STATE,
        data_home=str(config.RAW_DATA_DIR),
    )

    logger.info(
        "Loaded %d train / %d test docs across %d topics",
        len(train.data), len(test.data), len(train.target_names),
    )

    return Dataset(
        X_train=list(train.data),
        y_train=list(train.target),
        X_test=list(test.data),
        y_test=list(test.target),
        target_names=list(train.target_names),
    )
