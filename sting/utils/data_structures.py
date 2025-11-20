from typing import Any, Dict, Iterable, List, Tuple, Iterator, Optional
import pandas as pd

class ListMap:
    """
    List-backed map with O(1) lookup, insert, and delete operations.

    A ListMap stores values in fixed groups, where each group
    maintains a list indexed by an integer ID. Keys are (group, id)
    tuples. Accessing a group returns all group values in ID order.
    You may also define views, a named collection of groups from 
    which values will be yielded in collection order. 

    ==================
    Basic Use:
    
    >>> library = ListMap(["biology", "math", "poetry"]) 
    >>> # Add some books 
    >>> library[("math", 1)] = Book(id=513.2, title="Linear Algebra") 
    >>> library[("math", 2)] = Book(id=602., title="Linear Controls") 
    >>> library[("biology", 1)] = Book(id=582.1, title="Biology 101") 
    >>> # Selections and views 
    >>> list(library["math"]) # Returns a list of all math books 
    >>> library.add_view("science", ["math", "biology])
    >>> library.view("science") # Returns math books, then biology books
    >>> ids, titles = library.view("science", attrs=["id", "title"])
    """

    def __init__(self, groups: Iterable[str], look_up):
        groups = list(groups)

        self._groups = groups                    
        self._data: Dict[str, List[Any]] = {g: [] for g in groups}
        self.views: Dict[str, List[str]] = {}
        self._len: Dict[str, int] = {g: 0 for g in groups}
        self._hash = look_up

        # Warn about large unintended sparse jumps
        self.max_growth = 1024

    # ------------------------------------------------------------
    # Basic mapping operations
    # ------------------------------------------------------------

    def __getitem__(self, key: str | Tuple[str, int]):
        if isinstance(key, tuple):
            group, idx = key

            arr = self._data[group]
            if idx < 0 or idx >= len(arr):
                raise KeyError(key)

            value = arr[idx]
            if value is None:
                raise KeyError(key)

            return value

        if isinstance(key, str):
            # Return all values from given group
            return self.values(key)

        raise TypeError("Key must be a string group or (group, id) tuple.")

    def __setitem__(self, key: Tuple[str, int], value: Any):
        group, idx = key

        if idx < 0:
            raise KeyError(key)

        arr = self._data[group]

        # Grow list if needed
        if idx >= len(arr):
            n_slots = idx + 1 - len(arr)
            if n_slots > self.max_growth:
                raise UserWarning(
                    f"Attempting to grow '{group}' by {n_slots} slots "
                    f"(limit = {self.max_growth})."
                )
            arr.extend([None] * n_slots)

        # Occupancy update
        if arr[idx] is None:
            self._len[group] += 1

        arr[idx] = value

    def __delitem__(self, key: Tuple[str, int]):
        group, idx = key
        arr = self._data[group]

        if idx < 0 or idx >= len(arr):
            raise KeyError(key)

        if arr[idx] is not None:
            self._len[group] -= 1

        arr[idx] = None

    def __contains__(self, key: Tuple[str, int]):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def add(self, value):
        group, idx = self._hash(value)
        self[group, idx] = value

    # ------------------------------------------------------------
    # Views
    # ------------------------------------------------------------

    def add_view(self, name: str, groups: List[str]):
        """Create a named view over a collection of groups."""
        self.views[name] = groups

    def drop_view(self, name: str):
        """Remove a given view."""
        del self.views[name]

    def view(self, name: str, attrs=None, dataframe=False):
        """Return generator of values inside the view."""
        if isinstance(attrs, list):
            values = (self.values(self.views[name], attr=attr) for attr in attrs)
            if dataframe:
                index = [f"{g}_{idx}" for (g, idx) in self.keys(self.views[name])]
                return pd.DataFrame(index=index, data=dict(zip(attrs, values)))
                
            return values
            
        return self.values(self.views[name], attr=attrs)

    
    def length(self, group=None, view=None):
        """Return the length of a group or view."""
        if group:
            return self._len[group]
        if view:
            return sum([self._len[g] for g in self.views[view]])
            

    # ------------------------------------------------------------
    # Iteration and listing
    # ------------------------------------------------------------

    def next_open(self, group: str):
        """Get the next open ID in O(1) time."""
        return len(self._data[group])

    def values(self, groups: Optional[Iterable[str]] = None, attr: str = None) -> Iterator[Any]:
        for _, _, v in self.items(groups):
            if attr:
                yield getattr(v, attr)
            else:
                yield v

    def items(self, groups: Optional[Iterable[str]] = None):
        if groups is None:
            groups = self._groups
        elif isinstance(groups, str):
            groups = [groups]

        for c in groups:
            arr = self._data[c]
            for idx, item in enumerate(arr):
                if item is not None:
                    yield (c, idx, item)

    def keys(self, groups: Optional[Iterable[str]] = None):
        for cat, idx, _ in self.items(groups):
            yield (cat, idx)

    def apply(self, method, *args):
        for c in self.values():
            if hasattr(c, method):
                getattr(c, method)(*args) 
                
    # ------------------------------------------------------------
    # Group Actions
    # ------------------------------------------------------------

    def clear(self, group: str):
        """Clear all data in a group."""
        self._data[group] = []
        self._len[group] = 0

    def add_group(self, group: str):
        self._data[group] = []
        self._groups.append(group)
        self._len[group] = 0
        
    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------

    def __repr__(self):
        # groups and counts
        data_lines = [
            f"  - ({self._len[c]}) {c}"
            for c in self._groups
        ]
        view_lines = [
            f"  - ({sum(self._len[c] for c in cats)}) {name}: {cats}"
            for name, cats in self.views.items()
        ]

        return (
            "Groups:\n"
            + ("\n".join(data_lines) or "  (none)")
            + "\n\nViews:\n"
            + ("\n".join(view_lines) or "  (none)")
        )
