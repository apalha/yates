# -*- coding: utf-8 -*-
""" __init__ file for plasma_shape module

Description
-----------
Simply the __init__ file for plasma_shape module.

References
----------

:First added:  Tue Apr 21 2016
:Last updated: Tue Apr 21 2016
:Copyright: Copyright(C) __2016__ apalha
:License: GNU GPL version 3 or any later version
"""

"""
Reviews
-------
1. First implementation. (apalha, 2016-04-21)
"""


# import Soloviev shapes
import soloviev

# import the interface class for plasma shape definition
from .soloviev.__classes import SolovievShape


# import nonlinear shapes
import nonlinear

# import the interface class for plasma shape definition
from .nonlinear.__classes import NonlinearShape

