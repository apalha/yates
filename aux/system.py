# -*- coding: utf-8 -*-
""" Definition of auxiliary yates' system functions

Description
-----------
This module implements special system-related functions such as determining the name of calling function, etc.


References
----------

....

:First added:  2016-04-21
:Last updated: 2016-04-21
:Copyright: Copyright(C) 2016 apalha
:License: GNU GPL version 3 or any later version
"""

__all__ = [
           'currentFuncName'            # Functions
          ]

import sys


def currentFuncName(n=0):
    r"""
    Function that returns the name of the current function (n=0) or the name of the nth caller.

    Usage
    -----
    .. code-block :: python

        function_name = currentFuncName(n=0)


    Parameters
    ----------
    n : int
        The number of the caller whose name is to be computed (0 is the current function 1 is the first caller, etc).


    Returns
    -------
    function_name : string
                    The name of the nth caller of the current function.


    REFERENCES
    ----------


    :First Added:   2015-12-02
    :Last Modified: 2015-12-02
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2015-12-02)
    """

    return sys._getframe(n + 1).f_code.co_name
