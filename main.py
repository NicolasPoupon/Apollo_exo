#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

SITES = [f"site_{i}" for i in range(3)]
MEASURES = list("abcde")


@dataclass
class Dataset:
    site: str
    data: pd.DataFrame
    is_valid: bool


class DatasetIterator:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __next__(self):
        site = self.rng.choice(SITES, size=1).item()
        names = self.rng.choice(MEASURES, size=(1000))
        values = self.rng.integers(1, 100, size=(1000))
        data = pd.DataFrame({"name": names, "value": values}).astype(
            {"name": "string", "value": "Int64"}
        )
        is_valid = self.rng.choice([False, True], size=1).item()

        return Dataset(site=site, data=data, is_valid=is_valid)

    def __iter__(self):
        return self

def clear_datasets(datasets, n, n_invalid_limit):
    nb_consecutive_invalid_datasets = 0
    res = []

    for dataset in datasets:
        if dataset.is_valid:
            res.append(dataset)
            if len(res) == n:
                break
            nb_consecutive_invalid_datasets = 0
        else:
            nb_consecutive_invalid_datasets += 1
            if nb_consecutive_invalid_datasets == n_invalid_limit:
                raise ValueError(f"Too many consecutive invalid datasets ({n_invalid_limit})")

    if len(res) < n:
        raise ValueError(f"Not enough valid datasets. Nb valid datasets: {len(res)}")

    return res

def compute(datasets: Iterator[Dataset], n=1000, n_invalid_limit=100):
    valid_datasets = clear_datasets(datasets, n, n_invalid_limit)
    valid_data = pd.concat([dataset.data.assign(site=dataset.site) for dataset in valid_datasets])

    result = valid_data.groupby(['site', 'name']).agg(
        total=('value', 'sum'),
        average=('value', 'mean')
    ).reset_index()

    return result

if __name__ == "__main__":
    generator = DatasetIterator()
    try:
        result = compute(generator)
        print(result)
    except ValueError as err:
        print(f"Error: {err}")
