from collections import namedtuple
from itertools import tee
from typing import Iterable, Any
import pandas as pd


class Stream:
    def __init__(self, iterator: Iterable[Any]):
        self._iterable = iterator

    def __iter__(self):
        return iter(self._iterable)

    def filter(self, fn):
        return Stream(filter(fn, self))

    def map(self, fn):
        return Stream(map(fn, self))

    def select(self, *attrs: str):
        """Return a namedtuple of generators, each yielding one attribute."""
        if not attrs:
            raise ValueError("select() requires at least one attribute.")

        # Copy the iterable N times
        tees = tee(self._iterable, len(attrs))

        def attr_gen(attr, it):
            for obj in it:
                yield getattr(obj, attr) if hasattr(obj, attr) else None

        gens = [attr_gen(attr, it) for attr, it in zip(attrs, tees)]
        Selection = namedtuple("Selection", attrs)
        return Selection(*gens)

    def to_list(self):
        """Return a list of all items in the generator."""
        return list(self)

    def to_table(self, *attrs):
        """Return a dataframe with one column per selected attribute."""
        selection = self.select(*attrs)
        df = pd.DataFrame({a: list(gen) for a, gen in zip(attrs, selection)})
        return df


# ------------------------------------------------------------
# Common selections
# ------------------------------------------------------------


def generators():
    pass


def shunts():
    pass


def branches():
    pass
