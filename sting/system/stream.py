from more_itertools import transpose
from itertools import tee
from typing import Iterable, Any
#import pandas as pd
import polars as pl

class Stream:
    def __init__(self, iterator: Iterable[Any], index_map: dict):
        self._iterable = iterator
        # Mapping between class __name__ and short hand string
        # Used to create a index when converting to a table.
        self._index_map = index_map

    def __iter__(self):
        return iter(self._iterable)
    
    def copy(self):
        it1, it2 = tee(self._iterable, 2)
        # replace the original stream's iterator with one tee’d branch
        self._iterable = it1
        # return a new Stream built on the other
        return Stream(it2, self._index_map)

    def filter(self, fn):
        return Stream(filter(fn, self), self._index_map)

    def map(self, fn):
        return Stream(map(fn, self), self._index_map)

    def select(self, *attrs: str):
        """Return a namedtuple of generators, each yielding one attribute."""
        if not attrs:
            raise ValueError("select() requires at least one attribute.")

        # Select all attributes from each component
        selection = [
            [getattr(obj, attr) if hasattr(obj, attr) else None for attr in attrs] 
            for obj in self._iterable
        ]
        
        return transpose(selection)

    def to_list(self):
        """Return a list of all items in the generator."""
        return list(self)

    def to_table(self, *attrs) -> pl.DataFrame:
        """Return a dataframe with one column per selected attribute."""
        selection = self.select(*attrs)
        df = pl.DataFrame({a: list(gen) for a, gen in zip(attrs, selection)})
        return df


