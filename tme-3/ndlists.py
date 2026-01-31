"""This module implements a numpy-like syntax for lists of lists. Everyone sane
will tell you it is a bad idea except in very rare exceptions. For almost all
cases, numpy array can do what you want in an infinitely better way. Seriously,
if you need to use a numpy-like syntax, use numpy, it probably vetter adapted to
yout use case than this.

The use case we developped this for, is, we think, one of the very limited 
exception. This module is developped for a course where we teach computer 
science students the linear algebraic part of quantum physics. Our goal is to
help students develop their intuition in linear algebra by programming some of 
its elementary operations. As we move on to solve more complicated problems, we 
ask the students to use 'actual linear algebra' libraries, that is numpy and 
scipy, hence the need to copy their syntax.

Note that a natural extension of this module is to implement linear algebraic 
operations on lists of list. Please do it, it’s made for this, but we’d prefer
you not to share it. We think a student who struggled to make a half working 
implementation of matrix multiplication learned much more than a student who 
downloaded a way better implementation over the internet."""

__version__ = "0.0.0a"

import numbers
import collections.abc
import itertools


def shape(l: list) -> (int, tuple[int, ...]):
    """Computes the array shape of a list of lists of lists ..., assuming all 
lists at the same recursion level have the same length"""
    match l:
        case []:
            return (0,)
        case [list(), *_]:
            return len(l), *shape(l[0])
        case _:
            return (len(l),)


def int2slice(n: int) -> slice:
    """Convert an int to a slice of size 1
    [*range(int2slice(n).indices(m))==n] #if 0<=n<m """
    return slice(n, n + 1, None)


def ndslicenormalize(ndin: tuple[slice | int] | int) -> tuple[slice]:
    match ndin:
        case tuple():
            return tuple(k if isinstance(k, slice) else int2slice(k) for k in ndin)
        case int():
            return (int2slice(ndin),)
        case slice():
            return (ndin,)
        case _:
            raise ValueError(f"Cannot make a nd-slice out of a {type(ndin)}")


def slicelen(s: slice, l: int) -> int:
    "computes the length of a slice"
    start, stop, step = s.indices(l)
    l, cor = divmod(stop - start, step)
    return l + (cor != 0)


# TODO proper testing
def testslicelen(l, verbose=False):
    for sspec in [(None, None, None), (None, 2, None), (10, None, None),
                  (None, None, -1), (None, 2, -1), (10, None, -1),
                  (None, None, 2), (None, 2, 2), (10, None, 2),
                  (None, None, -2), (None, 2, -2), (10, None, -2),
                  ]:
        s = slice(*sspec)
        if verbose:
            print(s.indices(l))

        assert slicelen(s, l) == len(range(*s.indices(l))), \
            f"Fails for {sspec} with length {l}: sll gives {sll(s, l)} for {range(*s.indices(l))}"
        if verbose:
            print("success")


# verb=False
# testslicelen(0, verbose=verb)
# testslicelen(3, verbose=verb)

#	print("tests successful :-)")


class ndlist(list):
    def __init__(self, lst):
        list.__init__(self, lst)
        self._lst = lst  # self.copy()
        self.shape = shape(self._lst)  #  (self) would imply infinite recursion
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) > self.ndim:
            raise IndexError(f"Too many indices: object is {self.ndim}-dimensional, index is {len(key)}-dimensional ")
        match key:
            case int() | slice():
                return self._lst[key]
            case (k, ):
                return self._lst[k]
            case (int() as k0, *y):
                return ndlist(self[k0])[tuple(y)]
            case (slice() as k0, *y):
                return [ndlist(l)[tuple(y)] for l in self[k0]]
            case _:
                raise IndexError(
                    f"We do not know wat to do here with a key of of type: {type(key)} and of value: {key}")

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) > self.ndim:
            raise IndexError(f"Too many indices: object is {self.ndim}-dimensional, index is {len(key)}-dimensional")

        # Normalize the key to a tuple of slices
        key = ndslicenormalize(key)

        # Compute the shape of the slice
        keyshape = [self._slicelen(k, sh) for k, sh in zip(key, self.shape)]

        # Validate the value's shape if it's an iterable
        if isinstance(value, collections.abc.Iterable):
            value_shape = shape(value)
            if value_shape != tuple(keyshape):
                # Convert both to tuple for comparison
                if tuple(value_shape) != tuple(keyshape):
                    raise ValueError(
                        f"Shape mismatch: cannot assign value of shape {value_shape} to slice of shape {keyshape}")

        # Perform the assignment
        match value:
            case numbers.Number():  # Scalar assignment
                for s in self.ndsliceiteration(key):
                    settupleindex(self._lst, s, value)
            case collections.abc.Iterable():  # Iterable assignment
                for s, sval in zip(self.ndsliceiteration(key),
                                   itertools.product(*[range(k) for k in keyshape if k > 1])):
                    if not isinstance(sval, tuple):  # Ensure sval is a tuple
                        sval = (sval,)
                    settupleindex(self._lst, s, gettupleindex(value, sval))
            case _:
                raise NotImplementedError(f"Assignment of {type(value)} not supported")

    # ---------------- Addition ----------------
    def __add__(self, other):
        """ndlist + ndlist or ndlist + scalar"""
        if isinstance(other, ndlist):
            def add_lists(a, b):
                return [add_lists(x, y) if isinstance(x, list) else x + y
                        for x, y in zip(a, b)]

            return ndlist(add_lists(self._lst, other._lst))
        elif isinstance(other, (int, float, complex)):
            def add_scalar(lst):
                return [add_scalar(x) if isinstance(x, list) else x + other for x in lst]

            return ndlist(add_scalar(self._lst))
        else:
            return NotImplemented

    def __radd__(self, other):
        """scalar + ndlist"""
        if isinstance(other, (int, float, complex)):
            def add_scalar(lst):
                return [add_scalar(x) if isinstance(x, list) else other + x for x in lst]

            return ndlist(add_scalar(self._lst))
        else:
            return NotImplemented

    # ---------------- Subtraction ----------------
    def __sub__(self, other):
        """ndlist - ndlist or ndlist - scalar"""
        if isinstance(other, ndlist):
            def sub_lists(a, b):
                return [sub_lists(x, y) if isinstance(x, list) else x - y
                        for x, y in zip(a, b)]

            return ndlist(sub_lists(self._lst, other._lst))
        elif isinstance(other, (int, float, complex)):
            def sub_scalar(lst):
                return [sub_scalar(x) if isinstance(x, list) else x - other for x in lst]

            return ndlist(sub_scalar(self._lst))
        else:
            return NotImplemented

    def __rsub__(self, other):
        """scalar - ndlist"""
        if isinstance(other, (int, float, complex)):
            def sub_scalar(lst):
                return [sub_scalar(x) if isinstance(x, list) else other - x for x in lst]

            return ndlist(sub_scalar(self._lst))
        else:
            return NotImplemented

    # Helper method for multiplication between scalar and ndlist
    def __rmul__(self, other):
        """For scalar * ndlist"""
        if isinstance(other, (int, float, complex)):
            def mul_scalar(lst):
                return [mul_scalar(x) if isinstance(x, list) else other * x for x in lst]

            return ndlist(mul_scalar(self._lst))
        else:
            return NotImplemented

    # Helper method for division between scalar and ndlist
    def __rtruediv__(self, other):
        """For scalar / ndlist"""
        if isinstance(other, (int, float, complex)):
            def div_scalar(lst):
                return [div_scalar(x) if isinstance(x, list) else other / x for x in lst]

            return ndlist(div_scalar(self._lst))
        else:
            return NotImplemented

    def _slicelen(self, sl: slice | int, shape: int) -> int:
        """returns the length of a (1D) slice specified by sl on a dimension of length shape"""
        # TODO take care of the empty case, where shape==0
        match sl:
            case int():
                return 1
            case slice():
                return slicelen(sl, shape)
            case _:
                raise ValueError(f"{sl} is not a valid index/slice!")

    def ndsliceiteration(self, ndslice):
        "Iterable iterating through the coordinates of ndslice"
        if len(ndslice) > self.ndim:
            raise IndexError("To many indices")  # TODO Improve error msg
        return itertools.product(*[
            range(*sl.indices(sh)) for sl, sh in itertools.zip_longest(ndslicenormalize(ndslice),
                                                                       self.shape,
                                                                       fillvalue=slice(None))
        ])

    def flatten(self):
        if self.ndim == 1:
            return ndlist(self)
        else:
            return ndlist([item for sublist in self for item in ndlist(sublist).flatten()])


def gettupleindex(l: list, i: tuple[int, ...]):
    """To get a value in a list of lists
    gettupleindex(l,(2,1))==l[2][1]"""
    if len(i) == 1:
        return l[i[0]]
    else:
        return gettupleindex(l[i[0]], i[1:])


def settupleindex(l: list, i: tuple[int, ...], value):
    """To get a value in a list of lists
    gettupleindex(l,(2,1))==l[2][1]"""
    if len(i) == 1:
        l[i[0]] = value
    else:
        settupleindex(l[i[0]], i[1:], value)
