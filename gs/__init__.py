# -*- coding: utf-8 -*-
""" __init__ file for gs module

Description
-----------
Simply the __init__ file for gs module.

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

# import Grad-Shafranov solver
from __gs import *


# import geometry related functions and classes including pre-defined plasma shapes and associated plasma boundaries
import geometry

# import the interface class for plasma shape definition
from .geometry.plasma_shape.soloviev.__classes import SolovievShape
from .geometry.plasma_shape.nonlinear.__classes import NonlinearShape

# import analytical solutions to the Grad-Shafranov equation
import solutions

# import the interface classes for analytical solutions to the Grad-Shafranov equations
from .solutions.soloviev.__classes import SolovievSolution