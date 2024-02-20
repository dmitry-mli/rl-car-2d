import functools
from typing import TypeVar, Callable

T = TypeVar("T")


def then(transform: Callable[[T], T]):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return transform(func(*args, **kwargs))

        return wrapper

    return decorator
