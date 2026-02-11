from collections import namedtuple
from more_itertools import transpose
from itertools import tee
from typing import Iterable, Any
import pandas as pd
import polars as pl
import copy

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

    def to_table(self, *attrs, index=None, index_name=None):
        """Return a dataframe with one column per selected attribute."""
        attrs = list(attrs)
        if index:
            attrs.append(index)
        else:
            attrs.append("id")
            class_name = self.copy().map(lambda x: type(x).__name__).to_list()

        selection = self.select(*attrs)
        df = pd.DataFrame({a: list(gen) for a, gen in zip(attrs, selection)})

        if index:
            df = df.set_index(index)
        else:

            # Create a default index like "inf_src_1"
            df["__name__"] = class_name
            df["index"] = df["__name__"].replace(self._index_map) + "_" + df["id"].astype(str)
            df = df.set_index("index").drop(columns=["__name__", "id"])

        df.index.name = index_name

        return df
    
    def to_table_pl(self, *attrs, index=None, index_name=None):
        """Return a dataframe with one column per selected attribute."""
        attrs = list(attrs)
        if index:
            attrs.append(index)
        else:
            attrs.append("id")
            class_name = self.copy().map(lambda x: type(x).__name__).to_list()

        selection = self.select(*attrs)
        df = pl.DataFrame({a: list(gen) for a, gen in zip(attrs, selection)})

        if index:
            df = df.set_index(index)
        else:
            # Add a column "component" with the class name of each component (e.g. "InfiniteSource") 
            df = df.with_columns( pl.Series("class", class_name) )
            
            # Replace column "component" with type name based on index map (e.g. "inf_src") 
            df = df.with_columns( 
                                 pl.col("class").replace(self._index_map).alias("component") )
            df = df.with_columns(
                                 (pl.col("component") + "_" + pl.col("id").cast(pl.String)).alias("component_id")
                                )
            df = df.select(["component_id"] + attrs).drop("id")
            
            df
            #df["component_id"] = df["__name__"].replace(self._index_map) + "_" + df["id"].astype(str)
            #df = df.set_index("index").drop(columns=["__name__", "id"])

        #df.index.name = index_name

        return df


# ------------------------------------------------------------
# Common selections
# ------------------------------------------------------------

def find_tagged(system, tag_name):
    """
    Return a list of all components tagged with a specific 
    tag name.
    """
    # List of all components with the given tag name
    tagged_components = []
    # Scan over all component types
    for name in system.components["type"]:
        component_list = getattr(system, name)
        # If the component is tagged with the current tag name 
        # add it to the running list
        if  (len(component_list) > 0 and 
            hasattr(component_list[0], "tags") and 
            (tag_name in component_list[0].tags)):
            tagged_components.append(name)

    return tagged_components

def generators():
    """Query over all generators in the system"""
    return lambda system: find_tagged(system, "generator")

def shunts():
    """Query over all shunts in the system"""
    return lambda system: find_tagged(system, "shunt")

def branches():
    """Query over all branches in the system"""
    return lambda system: find_tagged(system, "branch")

def lines():
    """Query over all lines in the system"""
    return lambda system: find_tagged(system, "line")
