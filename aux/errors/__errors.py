# -*- coding: utf-8 -*-
""" YATES custom exceptions

Description
-----------
This module implements custom made exceptions to use within YATES.


References
----------


....

:First added:  2016-04-21
:Last updated: 2016-04-21
:Copyright: Copyright(C) 2016 apalha
:License: GNU GPL version 3 or any later version
"""


"""
Reviews
-------
1. First implementation. (apalha, 2016-04-21)

"""

__all__ = ['InputError'
           ]


class InputError(Exception):
    r"""
    Exception raised for errors in the input.



    Usage
    -----
    .. code-block:: python

        InputError(expr,msg)


    Parameters
    ----------
    expr : string
           Where the error occurred.
    msg : string
          The message to be displayed with the error.

    Attributes
    ----------
    expr : string
           Where the error occurred.
    msg : string
          The message to be displayed with the error.


    TODO
    ----


    ....

    :First Added:   2016-04-21
    :Last Modified: 2016-04-21
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version
    """

    """
    Reviews:
        1. First implementation. (apalha, 2016-04-21)
    """

    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg