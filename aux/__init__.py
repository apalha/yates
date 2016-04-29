# -*- coding: utf-8 -*-
""" __init__ file for aux module

Descriptione
-----------
Simply the __init__ file for aux module containing auxiliary functions.

References
----------

:First added:  Tue Apr 19 2016
:Last updated: Tue Apr 20 2016
:Copyright: Copyright(C) __2016__ apalha
:License: GNU GPL version 3 or any later version
"""

"""
Reviews
-------
1. First implementation. (apalha, 2016-04-19)
2. Added the geometry module to the import. (apalha, 2016-04-20)

"""

# import optimized functions
from .function import DolfinFunction, PythonFunction1D, PythonFunction2D,\
                      mesh2triang, mplot_cellfunction, mplot_function, plot,\
                      contour, mcontour_function

# import system related functions
from .system import currentFuncName

from .interp import lobattoQuad, lobattoPoly
