# -*- coding: utf-8 -*-
""" Default values of configuration variables.

Description
-----------
This module defines the defaults values of configuration variables regarding timing calculation, plotting, etc.


References
----------

....

:First added:  2016-04-19
:Last updated: 2016-04-19
:Copyright: Copyright(C) 2016 apalha
:License: GNU GPL version 3 or any later version
"""



# Package wide parameters -------------------------------------------
#                                               <editor-fold desc="">

# default output parameters
OUTPUT = {'runtime_info':True,            # show functions' output information and status at runtime
          'timing':True,                  # enable timing of functions
          'plotting':True}                # enable plotting

# ---------------------------------------------------- </editor-fold>



# Module specific parameters -----------------------------------------
#                                               <editor-fold desc="">

# gs module (grad-shafranov solver)
GS_SOLVER = {'plasma_boundary':'fixed',                # type of boundary condition for plasma: fixed|free
             'linear_algebra_backend':'petsc',         # the linear algebra used by the solver: petsc|ublas|scipy
             'far_field_bc':False,                      # use far field boundary conditions (vanishing fields at infinity) or prescribed Dirichlet boundary conditions: True|False
             'nonlinear_solver':'Newton'}              # use Newton solver to solve the nonlinear system: Picard|Newton

# ---------------------------------------------------- </editor-fold>

