from typing import Iterable, TypeVar, Iterator

T = TypeVar("T")


def unique_iterator(iterable: Iterable[T]) -> Iterator[T]:
    seen = set()
    for element in iterable:
        if element not in seen:
            seen.add(element)
            yield element
