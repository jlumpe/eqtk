"""Checks that Numba is installed and usable or defines a dummy implementation of ``numba.jit()``.

Numba supported can be forced on or off by setting the ``EQTK_ENABLE_NUMBA`` environment variable
to "1" or "0", respectively.


.. py:function:: jit

   Function used in ``@jit`` decorators. Equal to ``numba.jit()`` if available,
   otherwise a dummy decorator is substituted which returns its argument
   unchanged.
"""

from typing import overload, Tuple, Callable, TypeVar, TYPE_CHECKING
import os
import logging

import numpy as np


F = TypeVar('F', bound=Callable)


_ENVVAR = 'EQTK_ENABLE_NUMBA'

MIN_NUBMA_VERSION = (0, 42, 0)

#: Whether Numba support has been enabled.
HAVE_NUMBA: bool = False


def parse_version(vstr: str) -> Tuple[int, int, int]:
    """Parse a version string into a (major, minor, patch) integer tuple.

    Raises
    ------
    ValueError
        If the version string isn't in the expected format.
    """
    parts = vstr.split('.')
    if len(parts) != 3:
        raise ValueError('Expected version string to have 3 components')
    return tuple(map(int, parts))


def warn_disabled(msg):
    """Log warning about Numba being disabled."""
    msg2 = f' ({msg})' if msg else ''
    msg3 = f"Numba support disabled{msg2}. EQTK's performance will be severely degraded."
    logging.warning(msg3, stacklevel=2)


def numba_check():
    """Check to see if numba is available and working properly.

    Returns
    -------
    bool
        True if numba is installed and working properly; False
        otherwise.
    """

    if os.environ.get(_ENVVAR) == '1':
        return True

    if os.environ.get(_ENVVAR) == '0':
        # Assume user knows what they're doing, no warning needed
        return False

    # TODO: warn on other nonempty value?

    try:
        import numba
        import scipy  # Required for numba linear algebra functions

    except ImportError as e:
        warn_disabled(f'{e.name} not installed')
        return False

    try:
        version = parse_version(numba.__version__)
    except Exception:
        warn_disabled('unable to determine installed version number')
        return False

    if version < MIN_NUBMA_VERSION:
        req_str = '.'.join(map(str, MIN_NUBMA_VERSION))
        warn_disabled(f'minimum version: {req_str}, installed: {version}')
        return False

    try:
        test_numba()
    except Exception as e:
        logging.warning(e)
        warn_disabled('jit test failed')

    return True


def test_numba():
    """Test we can compile and execute a simple jit function.

    Sometimes linalg things fail with inconsistent BLAS installations.
    """
    import numba

    A = np.array([[0.06, 0.2], [0.2, 0.7]])
    b = np.array([1.2, 0.3])

    @numba.njit
    def my_solve(A, b):
        return np.linalg.solve(A, b)

    x = my_solve(A, b)


@overload
def _dummy_jit() -> Callable[[F], F]:
    # Return wrapper
    pass

@overload
def _dummy_jit(f: F) -> F:
    # Single callable
    pass

@overload
def _dummy_jit(x, *args, **kwargs) -> Callable[[F], F]:
    # Return wrapper
    pass

def _dummy_jit(*args, **kwargs):
    """Dummy wrapper for jitting if numba is not installed."""

    def wrapper(f):
        return f

    if (len(args) > 0 and callable(args[0])) or len(kwargs) > 0:
        # @jit(int32(int32, int32)), @jit(signature="void(int32)")
        return wrapper
    elif len(args) == 0:
        # @jit()
        return wrapper
    else:
        # @jit
        return args[0]


jit = _dummy_jit


# Always disable for type checker
if not TYPE_CHECKING and numba_check():
    HAVE_NUMBA = True
    from numba import jit
