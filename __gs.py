# -*- coding: utf-8 -*-
""" Grad-Shafranov solver.

Description
-----------
This module implements a finite element solver based on FEniCS for the Grad-Shafranov equation, see [lackner1976]_, [moret2015]_ and [heumann2015]_
for an overview of solvers.


References
----------
.. [lackner1976] Lackner, K., Computation of ideal MHD equilibria, Computer Physics Communications 12, pp. 33-44, 1976.

.. [moret2015] Moret, J.-M., Duval, B.P., Le, H.B., Coda, S., Reimerdes, H., Tokamak equilibrium reconstruction code LIUQE and
       its real time implementation, Fusion Engineering and Design 91, pp. 1-15, 2015.

.. [heumann2015] Heumann, H., Blum, J., Boulbe, C., Faugeras, B., Selig, G., Ané, J.-M., Brémond, S., Grandgirard, V., P. Hertout,
       Nardon, E., Quasi-static free-boundary equilibrium of toroidal plasma with CEDRES++: Computational methods and
       and applications, Journal of Plasma Physics, Available on CJO2015 doi:10.1017/S0022377814001251, 2015.

....

:First added:  2015-04-13
:Last updated: 2015-12-08
:Copyright: Copyright(C) 2015 apalha
:License: GNU GPL version 3 or any later version
"""
from traits.trait_types import self

"""
Reviews
-------
1. First implementation. (apalha, 2015-04-13)

"""

__all__ = [
           'GS_Solver', 'PlasmaShape','DolfinFunction', # classes
           'PythonFunction1D','PythonFunction2D',
           'plasma_boundary_pataki', 'plasma_boundary_xpoint_iter', # functions
           'DEFAULT_SOLVER_PARAMETERS','DEFAULT_OUTPUT_PARAMETERS', # constants
           'output_parameters' # variables
          ]

import dolfin
import fenicstools
import numpy
import triangle
import scipy.sparse
import scipy.sparse.linalg

import dolfinmplot
import matplotlib.tri as tri

import time
import sys

# Constants -------------------------------------------------------------------

# default output parameters
DEFAULT_OUTPUT_PARAMETERS = {'runtime_info':True,            # show functions' output information and status at runtime
                             'timing':True,                  # enable timing of functions
                             'plotting':True}                # enable plotting

# default solver parameters
# parameters related to the GS solver
DEFAULT_SOLVER_PARAMETERS = {'plasma_boundary':'fixed',                # type of boundary condition for plasma: fixed|free
                             'linear_algebra_backend':'petsc',         # the linear algebra used by the solver: petsc|ublas|scipy
                             'far_field_bc':False,                      # use far field boundary conditions (vanishing fields at infinity) or prescribed Dirichlet boundary conditions: True|False
                             'nonlinear_solver':'Newton'}              # use Newton solver to solve the nonlinear system: Picard|Newton

# physical constants
MU0 = 1.0#numpy.pi*4.0*1.0e-7 # vacuum permeability
MU0_INV = 1.0/MU0 # inverse of vacuum permeability

#------------------------------------------------------------------------------

# Module-wide variables -------------------------------------------------------

output_parameters = DEFAULT_OUTPUT_PARAMETERS

#------------------------------------------------------------------------------

# Classes ---------------------------------------------------------------------

class GS_Solver():
    r"""
    This class sets up a Grad-Shafranov solver based on piecewise linear finite elements.

    The Grad-Shafranov equation solved is:

    .. math::

        -\nabla\cdot\left(\frac{1}{\mu_{0} r}\nabla\psi\right) = \left\{\begin{array}{ll}r\frac{dp}{d\psi} + \frac{1}{\mu_{0}r}f\frac{df}{d\psi} & \text{in } \Omega_{p}(\psi), \\ \frac{I_{i,j}}{S_{i,j}} & \text{in } \Omega_{c_{i,j}}, \\ -\frac{\sigma_{k}}{r}\frac{\partial\psi}{\partial t} & \text{in } \Omega_{pc_{k}} \\ 0 & \text{elsewhere,}\end{array}\right.

    with

    .. math::

        \psi(0,z) = 0 \quad \text{and} \lim_{\|(r,z)\|\rightarrow +\infty} \psi(r,z) = 0.

    Where the plasma exists in the region :math:`\Omega_{p}`, the coil :math:`i`
    in the circuit :math:`j` is present in the region :math:`\Omega_{c_{i,j}}`,
    passive conductor :math:`k` is defined in the region :math:`\Omega_{pc_{k}}`
    and the remaining domain is vacuum.

    Usage
    -----
    .. code-block:: python

        GS_Solver(mesh, inner_objects=None, outer_objects=None, solver_parameters=DEFAULT_SOLVER_PARAMETERS)


    Parameters
    ----------
    mesh : dolfin.Mesh
           A dolfin mesh specifying the geometry of the finite elements.

    inner_objects : TO_DEFINE
                    A collection of objects to identify the locations and properties of the different
                    objects that lie inside the mesh region.
                    The codes identifying the different objects are: ::
                        0    : vacuum
                        1    : plasma
                        1xxx : coil number xxx
                        2xxx : conductor number xxx

    outer_objects : TO_DEFINE
                    A collection of objects to identify the locations and properties of the different
                    objects that lie outside the mesh region.
                    The codes identifying the different objects are: ::
                        0    : vacuum
                        1    : plasma
                        1xxx : coil number xxx
                        2xxx : conductor number xxx

    solve_parameters : dict
                       Contains the following internal definitions for the solver

                       plasma_boundary : string, single value
                                         A string specifying the type of boundary used for the plasma
                                         domain. The different options are: ::

                                         'free' : plasma boundary is determined from the computation.
                                         'fixed : plasma boundary is prescribed by the user as the boundary of the
                                                  computational domain.

                       linear_algebra_backend : string, single value
                                                Specifies which linear algebra backend to use while solving the Grad-Shafranov problem. Different
                                                backends have different performances in different machines and environments. The user should test
                                                different options to identify which one performs better. Possible options are: ::

                                                'petsc' : for the PETSc linear algebra backend
                                                'ublas' : for the uBLAS linear algebra backend
                                                'scipy' : for the SciPy linear algebra backend

                       far_field_bc : bool, single value
                                      Specifies if far field boundary conditions are to be used or not. If True then
                                      boundary conditions of vanishing fields at infinity are used. If False, prescribed
                                      Dirichlet boundary conditions are used.
                                      
                       nonlinear_solver : string, single value
                                          Specifies which nonlinear solver is used.
                                          Possible options are: ::
                                          
                                          'Newton' : Newton solver
                                          'Picard' : Picard (fixed point) iterative solver


    Attributes
    ----------
    attribute_1 : type
                  Description of attribute 1aaa.

    attribute_2 : type
                  Description of attribute 2.


    TODO
    ----
        1. Define object class where to define coils and other things.
        2. Define an update boundary conditions function. In this function we
           just compute the terms associated to the boundary that we will have
           to add to the right hand side.

    ....

    :First Added:   2015-04-13
    :Last Modified: 2015-04-16
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version
    """

    """
    Reviews:
        1. First implementation. (apalha, 2015-04-13)
    """

    def __init__(self, mesh, inner_objects=None, outer_objects=None, solver_parameters=DEFAULT_SOLVER_PARAMETERS):
        """
        """

        """
        :First Added:   2015-04-13
        :Last Modified: 2015-04-28
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-04-13)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing = {} # initialize timing dictionary, this is only done in __init__
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        # store inputs

        self.mesh = mesh
        self.inner_objects = inner_objects
        self.outer_objects = outer_objects
        self.solver_parameters = solver_parameters

        # define the linear algebra backend in dolfin
        if self.solver_parameters['linear_algebra_backend'] == 'scipy':
            dolfin.parameters['linear_algebra_backend'] = 'uBLAS'
        elif self.solver_parameters['linear_algebra_backend'] == 'ublas':
            dolfin.parameters['linear_algebra_backend'] = 'uBLAS'
        else:
            dolfin.parameters['linear_algebra_backend'] = 'PETSc'

        # generate the solution function space, trial and basis functions
        self._V = dolfin.FunctionSpace(self.mesh,'CG',1)
        self._v_trial = dolfin.TrialFunction(self._V)
        self._v_test = dolfin.TestFunction(self._V)
        # compute the coordinates of the degrees of freedom and reshape the result to be [[x0,y0],[x1,y1],...,]
        self._vertices = self._V.dofmap().tabulate_all_coordinates(self.mesh).reshape(self._V.dim(),2)
        self._vertices_x = self._vertices[:,0].copy()
        self._vertices_y = self._vertices[:,1].copy()

        # define the solution function and its associated vector
        #self.psi = dolfin.Function(self._V)
        self.psi = DolfinFunction(self._V)
        if self.solver_parameters['linear_algebra_backend'] == 'scipy':
            self.psi_vector = numpy.zeros(self.psi.vector().array().shape)
        else:
            self.psi_vector = self.psi.vector()
            
        # if a newton solver is used to solve for the nonlinearity then we
        # need to solve for the variation between two iterations, delta_psi
        if self.solver_parameters['nonlinear_solver'] == 'Newton': 
            self.delta_psi = DolfinFunction(self._V)
            if self.solver_parameters['linear_algebra_backend'] == 'scipy':
                self.delta_psi_vector = numpy.zeros(self.delta_psi.vector().array().shape)
            else:
                self.delta_psi_vector = self.delta_psi.vector()

        # define the current function and the right hand side function and their associated vectors
        self.j = dolfin.Function(self._V)
        if self.solver_parameters['nonlinear_solver'] == 'Newton':
            self.dj = dolfin.Function(self._V) # if Newton method is used to solve
                                               # the nonlinearity then the derivative of
                                               # the current density with respect to psi
                                               # must be defined
            
        self._b = dolfin.Function(self._V)
        if self.solver_parameters['linear_algebra_backend'] == 'scipy':
            self.j_vector = numpy.zeros(self.j.vector().array().shape)
            if self.solver_parameters['nonlinear_solver'] == 'Newton':
                self.dj_vector = numpy.zeros(self.dj.vector().array().shape)
            self.b_vector = numpy.zeros(self.b.vector().array().shape)
            
        else:
            self.j_vector = self.j.vector()
            if self.solver_parameters['nonlinear_solver'] == 'Newton':
                self.dj_vector = self.dj.vector()
            self._b_vector = self._b.vector()

        # define the bilinear form of the Grad-Shafranov equation
        self._inv_mu0r = dolfin.Expression('mu0_inv/x[0]', mu0_inv=MU0_INV) # this is the term \frac{1}{\mu_{0}r}
        self.A_form = dolfin.inner(self._inv_mu0r*dolfin.grad(self._v_trial), dolfin.grad(self._v_test))*dolfin.dx
        
        if self.solver_parameters['nonlinear_solver'] == 'Newton':
            # the Jacobian term associated to the current density
            self.A_dj_form = dolfin.inner(self.dj*self._v_trial,self._v_test)*dolfin.dx
            # the right hand side associated to the previous iteration
            self.B_form = dolfin.inner(self.j,self._v_test)*dolfin.dx
        
        # assemble the bilinear form into a matrix
        self.A = dolfin.assemble(self.A_form)

        # assemble the mass matrix for efficiently computing the right hand side with boundary conditions
        self._M = dolfin.assemble(dolfin.inner(self._v_trial,self._v_test)*dolfin.dx)

        # We only need boundary conditions if we do not have field boundary conditions.
        # To solve the Grad-Shafranov equation in this situation we need to prescribe
        # the values of psi at the boundary of the FEM domain. If we use far field
        # boundary conditions, we do not need to set boundary conditions in our
        # system of equations what we need is to add the contribution of the
        # outer_objects into the solution.

        # define the prescribed Dirichlet boundary conditions and the Dirichlet
        # boundary
        self.bc = dolfin.DirichletBC(self._V, dolfin.Function(self._V), boundary)
        # extract and store the numbering of the degrees of freedom (dof)
        # of the vertices located at the boundary and their coordinates
        self.bc_dof = numpy.array(self.bc.get_boundary_values().keys())
        self.bc_coordinates = self._vertices[self.bc_dof,:]

        if not self.solver_parameters['far_field_bc']:
            # Define the Dirichlet boundary conditions and apply them to the system matrix.
            # Note that Dirichlet boundary conditions in dolfin are implemented in
            # a straighforward way. The rows of system matrix associated to the
            # vertices at the boundary are set to 0 with the exception of the column
            # associated to the vertex. This allows us to pre-apply the boundary
            # conditions to the system matrix without knowing a priori the boundary
            # condition values. This speeds up the process.
            self.bc.apply(self.A)
        
        if self.solver_parameters['nonlinear_solver'] == 'Picard': 
            # setup the LU solver so that the LU decomposition is pre-computed and saved
            if self.solver_parameters['linear_algebra_backend'] == 'petsc':
                self._solver = dolfin.LUSolver('petsc')
            elif self.solver_parameters['linear_algebra_backend'] == 'ublas':
                self._solver = dolfin.LUSolver('umfpack')
            elif self.solver_parameters['linear_algebra_backend'] == 'scipy':
                self._solver = spLUSolver()

            self._solver.parameters['reuse_factorization'] = True # reuse factorization
            self._solver.set_operator(self.A)
            self._solver.solve(self.psi_vector, self.j_vector) # solve once just to avoid any overhead in initializations
                                                               # self.j_vector is still zero, therefore self.psi_vector will
                                                               # remain 0
        
        elif self.solver_parameters['nonlinear_solver'] == 'Newton': 
            print 'Optimization for Newton solver needed!'            
            
        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


    def solve_step(self, j, dj=None, sigma=1.0, bc_function=None):
        r"""
        Solve the Grad-Shafranov equation given the right hand side j (current sources). This function assumes j to
        not depend on \psi. This function therefore can be used to solve one step of a nonlinear
        Grad-Shafranov equation used, for example in fixed point iteration or Newton.

        Usage
        -----
        .. code-block :: python

            self.solve_step(j,dj=None,sigma=1.0,bc_function=None)


        Parameters
        ----------
        j  : python function evaluatable at \psi and (r,z), j(r,z,psi)
             Represents the toroidal current density
        dj : python function evaluatable at \psi and (r,z), dj(r,z,psi)
             Represents the derivative of j with respect to psi

        Returns
        -------



        Attributes
        ----------
        psi
        psi_vector
        j
        j_vector
        dj
        dj_vector

        :First Added:   2015-04-13
        :Last Modified: 2016-04-18
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version

        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-04-13)
        2. Changed function from solve to solve_step, because this function is just one step of a nonlinear solver.
           This is important because in the future we wish to couple it to a transport solver and we need access
           kernel functions in order to design coupled solvers.
        3. j is now always a python function of (r,z,psi).
        4. Expanded the function to include a Newton solver to solve the
           nonlinearity. (apalha, 2016-04-18)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        if self.solver_parameters['nonlinear_solver'] == 'Picard':
            self.__solve_step_picard(j,sigma=sigma, bc_function=bc_function)

        elif self.solver_parameters['nonlinear_solver'] == 'Newton':
            self.__solve_step_newton(j,dj,sigma=sigma, bc_function=bc_function)

        
        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


    def __solve_step_picard(self, j, sigma, bc_function):
        r"""
        Solve one step of the Picard iteration of the nonlinear Grad-Shafranov equation
        given the right hand side j (current sources).

        Usage
        -----
        .. code-block :: python

            self.solve_step(j,sigma=1.0,bc_function=None)


        Parameters
        ----------
        j  : python function evaluatable at \psi and (r,z), j(r,z,psi)
             Represents the toroidal current density

        Returns
        -------



        Attributes
        ----------
        psi
        psi_vector
        j
        j_vector

        :First Added:   2015-04-13
        :Last Modified: 2016-04-18
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version

        """

        """
        Reviews:
        1. First implementation. (apalha, 2016-04-18)
        """

        # use psi(r,z) and j(r,z,psi) to compute j(r,z)
        self.j_vector[:] = sigma*j(self._vertices_x,self._vertices_y,self.psi_vector.array())

        
        # setup the right hand side
        self._b_vector = self._M*self.j_vector
        if bc_function == None:
            self._b_vector[self.bc_dof] = 0.0
        else:
            self._b_vector[self.bc_dof] = bc_function(self.bc_coordinates[:,0],self.bc_coordinates[:,1])

        
        # solve one step
        self._solver.solve(self.psi_vector, self._b_vector) 


    def __solve_step_newton(self, j, dj, sigma, bc_function):
        r"""
        Solve one step of the Newton iteration of the nonlinear Grad-Shafranov equation
        given the right hand side j (current sources).

        Usage
        -----
        .. code-block :: python

            self.solve_step(j,dj,sigma=1.0)


        Parameters
        ----------
        j  : python function evaluatable at \psi and (r,z), j(r,z,psi)
             Represents the toroidal current density
        dj : python function evaluatable at \psi and (r,z), dj(r,z,psi)
             Represents the derivative of j with respect to psi

        Returns
        -------


        Attributes
        ----------
        psi
        psi_vector
        j
        j_vector
        dj
        dj_vector

        :First Added:   2016-04-18
        :Last Modified: 2016-04-18
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version

        """

        """
        Reviews:
        1. First implementation. (apalha, 2016-04-18)
        """

        # use psi(r,z) and j(r,z,psi) to compute j(r,z)
        self.j_vector[:] = sigma*j(self._vertices_x,self._vertices_y,self.psi_vector.array())
        # use psi(r,z) and dj(r,z,psi) to compute dj(r,z)
        self.dj_vector[:] = sigma*dj(self._vertices_x,self._vertices_y,self.psi_vector.array())

        
        # setup the right hand side
        self._b_vector =  dolfin.assemble(self.B_form) - (self.A*self.psi_vector)
        self._b_vector[self.bc_dof] = 0.0
        
        
        # solve one step
        self.A_dj = dolfin.assemble(self.A_dj_form)
        S = self.A - self.A_dj
        self.bc.apply(S)
        dolfin.solve(S, self.delta_psi_vector, self._b_vector)



        # update psi to the new value, using the corrections from the newton solver
        self.psi_vector[:] = self.psi_vector[:] + self.delta_psi_vector[:]




class spLUSolver():
    r"""
    This class sets up an LU solver based on scipy, with similar behaviour to
    dolfin.LUSolver.

    It solves the system of equations:

    .. math::

        A\boldsymbol{x} = \boldsymbol{b}

    It implements LU decomposition in case several successive solves are performed
    with different right hand sides.


    Usage
    -----
    .. code-block:: python

        spLUSolver()

    Parameters
    ----------


    Attributes
    ----------
    attribute_1 : type
                  Description of attribute 1aaa.

    attribute_2 : type
                  Description of attribute 2.

    ....

    :First Added:   2015-04-16
    :Last Modified: 2015-04-16
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version
    """

    """
    Reviews:
        1. First implementation. (apalha, 2015-04-16)
    """

    def __init__(self):
        """
        """

        """
        :First Added:   2015-04-16
        :Last Modified: 2015-04-16
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-04-16)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing = {} # initialize timing dictionary, this is only done in __init__
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        # set the default value for reusing factorization
        self.parameters['reuse_factorization'] = False


        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


    def set_operator(self, operator):
        r"""
        Sets the linear operator that defines the system of equations.

        Usage
        -----
        .. code-block :: python

            self.set_operator(operator)


        Parameters
        ----------
        operator : type scipy.sparse.csc.csc_matrix, scipy.sparse.csr.csr_matrix
                   dolfin.cpp.la.Matrix
                   Matrix that defines the system of equations.

        Returns
        -------


        Attributes
        ----------
        _operator

        :First Added:   2015-04-16
        :Last Modified: 2015-04-16
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version

        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-04-16)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        if type(operator) == dolfin.cpp.la.Matrix:
            # if the operator is a dolfin matrix we need to convert it to a csc
            # scipy sparse matrix because LU factorization in csc format is faster
            self._operator = operator.sparray().tocsc()
        elif type(operator) == scipy.sparse.csc.csc_matrix:
            self._operator = operator
        elif type(operator) == scipy.sparse.csr.csr_matrix:
            # we need to convert it to a csc scipy sparse matrix because LU
            # factorization in csc format is faster
            self._operator = operator.tocsc()

        # pre-compute the LU decomposition
        self._operator_LU = scipy.sparse.linalg.splu(self._operator)

        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------

    def solve(self, x, b):
        r"""
        Solves the linear system of equations Ax=b, where A is given by self._operator.

        Usage
        -----
        .. code-block :: python

            self.solve(x,b)


        Parameters
        ----------
        x : type numpy.array(float64) of size (1,N)
            Vector containing the solution to the linear system. It must have a
            a size compatible with the system size. That is, A.shape must be (N,N).
            The solution of the system will be stored here.
        b : type numpy.array(float64) of size (1,N)
            Vector containing the right hand side to the linear system. It must have a
            a size compatible with the system size. That is, A.shape must be (N,N).

        Returns
        -------


        Attributes
        ----------


        :First Added:   2015-04-28
        :Last Modified: 2015-04-28
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version

        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-04-28)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        # solve the system of equations using the pre-computed LU decomposition
        x[:] = self._operator_LU.solve(b)


        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------



# class DolfinFunction1D():
#     r"""
#     This class sets up a function object in order to unify the interface between different kinds of functions
#     (python function, dolfin function, user defined function, etc).
#
#     Usage
#     -----
#     .. code-block:: python
#
#         Function(f)
#
#
#     Parameters
#     ----------
#     f : dolfin.function in 1D
#         The dolfin function in 1D we want to interface. When evaluating the object generated using function, it evaluates
#         function.
#
#     Attributes
#     ----------
#     f : dolfin.function in 1D
#         The dolfin function in 1D we want to interface. When evaluating the object generated using function, it evaluates
#         function.
#
#     gometric_dimension : int
#                          The geometric dimension of self.function
#
#
#     __f_vector : dolfin.Vector
#                  Is the dolfin.Vector associated to self.f
#     __V : dolfin.FunctionSpace
#           Is the dolfin.FunctionSpace used to express self.f
#     __x : numpy.array, (N,2)
#           Is the coordinates of the vertices of the mesh used to
#           discretize self.f
#
#     References
#     ----------
#
#
#
#     TODO
#     ----
#
#     ....
#
#     :First Added:   2015-12-04
#     :Last Modified: 2015-12-04
#     :Copyright:     Copyright (C) 2015 apalha
#     :License:       GNU GPL version 3 or any later version
#     """
#
#     """
#     Reviews:
#         1. First implementation. (apalha, 2015-12-04)
#         2. Changed it to DolfinFunction1D and created also DolfinFunction2D. Normal functions do not require any
#            special interface.
#     """
#
#     def __init__(self, f):
#         """
#
#         :First Added:   2015-12-04
#         :Last Modified: 2015-12-04
#         :Copyright:     Copyright (C) 2015 apalha
#         :License:       GNU GPL version 3 or any later version
#         """
#
#         """
#         Reviews:
#         1. First implementation. (apalha, 2015-12-04)
#         """
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             self.timing = {} # initialize timing dictionary, this is only done in __init__
#             start_time = time.time() # start the timer for this function
#         #---------------------------------------------------------------------------------------------------------------
#
#         # store the inputs as attributes
#         self.f = f
#         self.geometric_dimension = 1
#         self.type = 'dolfin_function'
#
#         # extract information from the dolfin function to speed up computations
#         self.__f_vector = self.f.vector()
#         self.__V = self.f.function_space()
#         self.__x = self.__V.mesh().coordinates().ravel()
#
#         # since self.function is a dolfin function, an efficient way of evaluating it at random points must be
#         # constructed
#
#         # in 1D, the most efficient way of doing the interpolation is using numpy.interp
#         # for that we need to extract the values of the function at the nodes. This means we need to extract the
#         # global numbering that establishes the connection between a vertex and a degree of freedom. This is done below
#         self.__dofs2vertex = self.__V.dofmap().dofs(self.__V.mesh(),0) # the degrees of freedom (dofs) are extracted with the mesh information
#         self.__f_at_x = numpy.zeros(self.__dofs2vertex.max()+1,numpy.float_)
#         self.__f_at_x[self.__dofs2vertex] = self.__f_vector.array()
#
#         # timing end ---------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
#         #---------------------------------------------------------------------------------------------------------------
#
#     def __call__(self, x):
#         r"""
#         Function that evaluates the function at a set of N points
#
#         Usage
#         -----
#         .. code-block :: python
#
#             f_evaluated = self(x)
#
#
#         Parameters
#         ----------
#         x : numpy.array, (N,1) for 1D points or tuple of numpy.array, (N,1) for nD points
#             The set of N one-dimensional points where to evaluate the function, self.f
#
#
#         Returns
#         -------
#         f_evaluated : numpy.array, (N,1)
#                       The evaluation of the function, self.f, at the set of N points x.
#
#
#         Attributes
#         ----------
#
#
#         REFERENCES
#         ----------
#
#         """
#
#         """
#         :First Added:   2015-12-04
#         :Last Modified: 2015-12-04
#         :Copyright:     Copyright (C) 2015 apalha
#         :License:       GNU GPL version 3 or any later version
#         """
#
#         """
#         Reviews:
#         1. First implementation. (apalha, 2015-12-04)
#         """
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             start_time = time.time() # start the timer for this function
#         #---------------------------------------------------------------------------------------------------------------
#
#         f_evaluated = numpy.interp(x,self.__x,self.__f_at_x)
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
#         #---------------------------------------------------------------------------------------------------------------
#
#         return f_evaluated
#
#
#     def update(self,f_new):
#         r"""
#         Update the function self.f with f_new in an efficient way.
#
#         Usage
#         -----
#         .. code-block :: python
#
#             self.update(f_new)
#
#
#         Parameters
#         ----------
#         f : function, dolfin.function, or any object that is callable with one or two inputs
#             The new function we want to interface.
#
#
#         Returns
#         -------
#
#
#         Attributes
#         ----------
#         f
#         __f_vector
#
#
#         REFERENCES
#         ----------
#
#         """
#
#         """
#         :First Added:   2015-12-07
#         :Last Modified: 2015-12-07
#         :Copyright:     Copyright (C) 2015 apalha
#         :License:       GNU GPL version 3 or any later version
#         """
#
#         """
#         Reviews:
#         1. First implementation. (apalha, 2015-12-07)
#         """
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             start_time = time.time() # start the timer for this function
#         #---------------------------------------------------------------------------------------------------------------
#
#         self.f.assign(f_new)
#         self.__f_vector = self.f.vector()
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
#         #---------------------------------------------------------------------------------------------------------------


# class DolfinFunction2D():
#     r"""
#     This class sets up a function object in order to unify the interface between different kinds of functions
#     (python function, dolfin function, user defined function, etc).
#
#     Usage
#     -----
#     .. code-block:: python
#
#         Function(f)
#
#
#     Parameters
#     ----------
#     f : dolfin.function in 2D
#         The dolfin function in 2D we want to interface. When evaluating the object generated using function, it evaluates
#         function.
#
#     Attributes
#     ----------
#     f : dolfin.function in 2D
#         The dolfin function in 2D we want to interface. When evaluating the object generated using function, it evaluates
#         function.
#
#     gometric_dimension : int
#                          The geometric dimension of self.function
#
#
#     __f_vector : dolfin.Vector
#                  Is the dolfin.Vector associated to self.f
#     __V : dolfin.FunctionSpace
#           Is the dolfin.FunctionSpace used to express self.f
#
#     References
#     ----------
#
#
#
#     TODO
#     ----
#
#     ....
#
#     :First Added:   2015-12-07
#     :Last Modified: 2015-12-07
#     :Copyright:     Copyright (C) 2015 apalha
#     :License:       GNU GPL version 3 or any later version
#     """
#
#     """
#     Reviews:
#         1. First implementation. (apalha, 2015-12-07)
#     """
#
#     def __init__(self, f):
#         """
#
#         :First Added:   2015-12-07
#         :Last Modified: 2015-12-07
#         :Copyright:     Copyright (C) 2015 apalha
#         :License:       GNU GPL version 3 or any later version
#         """
#
#         """
#         Reviews:
#         1. First implementation. (apalha, 2015-12-07)
#         """
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             self.timing = {} # initialize timing dictionary, this is only done in __init__
#             start_time = time.time() # start the timer for this function
#         #---------------------------------------------------------------------------------------------------------------
#
#         # store the inputs as attributes
#         self.f = f
#         self.geometric_dimension = 2
#
#         # extract information from the dolfin function to speed up computations
#         self.__V = self.f.function_space()
#
#         # timing end ---------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
#         #---------------------------------------------------------------------------------------------------------------
#
#     def __call__(self, x,y):
#         r"""
#         Function that evaluates the function at a set of N points
#
#         Usage
#         -----
#         .. code-block :: python
#
#             f_evaluated = self(x,y)
#
#
#         Parameters
#         ----------
#         x : numpy.array, (N,1)
#             The x-coordinates of the set of N 2-dimensional points where to evaluate the function, self.f
#         y : numpy.array, (N,1)
#             The y-coordinates of the set of N 2-dimensional points where to evaluate the function, self.f
#
#
#         Returns
#         -------
#         f_evaluated : numpy.array, (N,1)
#                       The evaluation of the function, self.f, at the set of N points (x,y).
#
#
#         Attributes
#         ----------
#
#
#         REFERENCES
#         ----------
#
#         """
#
#         """
#         :First Added:   2015-12-07
#         :Last Modified: 2015-12-07
#         :Copyright:     Copyright (C) 2015 apalha
#         :License:       GNU GPL version 3 or any later version
#         """
#
#         """
#         Reviews:
#         1. First implementation. (apalha, 2015-12-07)
#         """
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             start_time = time.time() # start the timer for this function
#         #---------------------------------------------------------------------------------------------------------------
#
#         xy = numpy.stack((x,y),axis=1)
#         probes = fenicstools.Probes(xy.flatten(),self.__V)
#         probes(self.f)
#         f_evaluated = probes.array()
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
#         #---------------------------------------------------------------------------------------------------------------
#
#         return f_evaluated
#
#
#     def update(self,f_new):
#         r"""
#         Update the function self.f with f_new in an efficient way.
#
#         Usage
#         -----
#         .. code-block :: python
#
#             self.update(f_new)
#
#
#         Parameters
#         ----------
#         f : function, dolfin.function, or any object that is callable with one or two inputs
#             The new function we want to interface.
#
#
#         Returns
#         -------
#
#
#         Attributes
#         ----------
#         f
#         __f_vector
#
#
#         REFERENCES
#         ----------
#
#         """
#
#         """
#         :First Added:   2015-12-07
#         :Last Modified: 2015-12-07
#         :Copyright:     Copyright (C) 2015 apalha
#         :License:       GNU GPL version 3 or any later version
#         """
#
#         """
#         Reviews:
#         1. First implementation. (apalha, 2015-12-07)
#         """
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             start_time = time.time() # start the timer for this function
#         #---------------------------------------------------------------------------------------------------------------
#
#         self.f.assign(f_new)
#         self.__f_vector = self.f.vector()
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if output_parameters['timing']:
#             self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
#         #---------------------------------------------------------------------------------------------------------------


class DolfinFunction(dolfin.functions.function.Function):
    r"""
    This class sets up a function object in order to unify the interface between different kinds of functions
    (python function, dolfin function, user defined function, etc). It also optimizes the plotting using dolfinmplot.

    Usage
    -----
    .. code-block:: python

        DolfinFunction(f)
        DolfinFunction(V)


    Parameters
    ----------
    f : dolfin.function in 1D or 2D
        The dolfin function in 1D or 2D we want to interface. When evaluating the object generated using function, it evaluates
        function.

    Attributes
    ----------
    f : dolfin.function in 1D or 2D
        The dolfin function in 2D we want to interface. When evaluating the object generated using function, it evaluates
        function.
    triangulation : matplotlib.tri.triangulation.Triangulation
                    (2D functions) Is the triangulation of the underlying mesh of function.
    __dofs2vertex : numpy.array
                    (1D functions) Is the matrix that converts the numbering of the degrees of freedom into vertex
                    numbering.
    __f_at_x : numpy.array
               (1D functions) Evaluation of the function at the vertices of the mesh.
    __x : numpy.array
          (1D functions) Coordinates (1D) of the vertices of the mesh.

    References
    ----------



    TODO
    ----

    ....

    :First Added:   2015-12-07
    :Last Modified: 2015-12-07
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version
    """

    """
    Reviews:
        1. First implementation. (apalha, 2015-12-07)
        2. Merged DolfinFunction1D and DolfinFunction2D into one single class tha inherits from
           dolfin.functions.function.Function. It has all the interface of dolfin.functions.function.Function but
           adds more functionality to speedup calculations and plotting.
    """

    def __init__(self,*args,**kwargs):
        """

        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        2. Merged DolfinFunction1D and DolfinFunction2D into one single class tha inherits from
           dolfin.functions.function.Function. It has all the interface of dolfin.functions.function.Function but
           adds more functionality to speedup calculations and plotting.
        """
        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing = {} # initialize timing dictionary, this is only done in __init__
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        # perform the original initialization
        dolfin.functions.function.Function.__init__(self,*args,**kwargs)

        if self.geometric_dimension() == 1:
            # extract information from the dolfin function to speed up computations
            self.__x = self.function_space().mesh().coordinates().ravel()

            # since self is a dolfin function, an efficient way of evaluating it at random points must be
            # constructed

            # in 1D, the most efficient way of doing the interpolation is using numpy.interp
            # for that we need to extract the values of the function at the nodes. This means we need to extract the
            # global numbering that establishes the connection between a vertex and a degree of freedom. This is done below
            self.__dofs2vertex = self.function_space().dofmap().dofs(self.function_space().mesh(),0) # the degrees of freedom (dofs) are extracted with the mesh information
            self.__f_at_x = numpy.zeros(self.__dofs2vertex.max()+1,numpy.float_)
            self.__f_at_x[self.__dofs2vertex] = self.vector().array()

        else: # geometric dimension 2
            # compute the triangulation to speed up plotting with matplotlib
            self.triangulation = dolfinmplot.mesh2triang(self.function_space().mesh())

        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------

    def __call__(self, **kwargs):
        r"""
        Function that evaluates the function at a set of N points

        Usage
        -----
        .. code-block :: python

            f_evaluated = self(x=x,y=y)


        Parameters
        ----------
        **x : numpy.array, (N,1)
            The x-coordinates of the set of N 2-dimensional points where to evaluate the function, self.f
        **y : numpy.array, (N,1)
            The y-coordinates of the set of N 2-dimensional points where to evaluate the function, self.f


        Returns
        -------
        f_evaluated : numpy.array, (N,1)
                      The evaluation of the function, self.f, at the set of N points (x,y).


        Attributes
        ----------


        REFERENCES
        ----------

        """

        """
        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        2. In order to work with both 1D and 2D functions the input was changed into **kwargs. (apalha, 2015-12-07)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        if self.geometric_dimension() == 1:
            f_evaluated = numpy.interp(kwargs['x'],self.__x,self.__f_at_x)
        else: # if it is 2
            xy = numpy.stack((kwargs['x'],kwargs['y']),axis=1)
            probes = fenicstools.Probes(xy.flatten(),self.function_space())
            probes(self)
            f_evaluated = probes.array()

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------

        return f_evaluated

    def update(self,f_new):
        r"""
        Update the function self.f with f_new in an efficient way.

        Usage
        -----
        .. code-block :: python

            self.update(f_new)


        Parameters
        ----------
        f : function, dolfin.function, or any object that is callable with one or two inputs
            The new function we want to interface.


        Returns
        -------


        Attributes
        ----------
        f
        __f_vector


        REFERENCES
        ----------

        """

        """
        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        self.assign(f_new)
        if self.geometric_dimension() == 1:
            self.__f_at_x[self.__dofs2vertex] = self.vector().array()

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


class PythonFunction1D():
    r"""
    This class sets up a function object in order to unify the interface between different kinds of functions.

    Usage
    -----
    .. code-block:: python

        Function(f)


    Parameters
    ----------
    f : python function in 1D
        The python function in 1D we want to interface. When evaluating the object generated using function, it evaluates
        function.

    Attributes
    ----------
    f : python function in 1D
        The python function in 1D we want to interface. When evaluating the object generated using function, it evaluates
        function.

    gometric_dimension : int
                         The geometric dimension of self.function

    References
    ----------



    TODO
    ----

    ....

    :First Added:   2015-12-07
    :Last Modified: 2015-12-07
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version
    """

    """
    Reviews:
        1. First implementation. (apalha, 2015-12-07)
    """

    def __init__(self, f):
        """

        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing = {} # initialize timing dictionary, this is only done in __init__
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        # store the inputs as attributes
        self.f = f
        self.type = 'python_function'

        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------

    def __call__(self, x):
        r"""
        Function that evaluates the function at a set of N points

        Usage
        -----
        .. code-block :: python

            f_evaluated = self(x)


        Parameters
        ----------
        x : numpy.array, (N,1) for 1D points or tuple of numpy.array, (N,1) for nD points
            The set of N one-dimensional points where to evaluate the function, self.f


        Returns
        -------
        f_evaluated : numpy.array, (N,1)
                      The evaluation of the function, self.f, at the set of N points x.


        Attributes
        ----------


        REFERENCES
        ----------

        """

        """
        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        f_evaluated = self.f(x)

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------

        return f_evaluated

    def geometric_dimension(self):
        r"""
        Function that returns the geometric dimension of the function.

        Usage
        -----
        .. code-block :: python

            geometric_dimension = self.geometric_dimension()


        Parameters
        ----------


        Returns
        -------
        geometric_dimension : int, single value
                              The geometric dimension of the function.


        Attributes
        ----------


        REFERENCES
        ----------

        """

        """
        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        """
        return 1


    def update(self,f_new):
        r"""
        Update the function self.f with f_new in an efficient way.

        Usage
        -----
        .. code-block :: python

            self.update(f_new)


        Parameters
        ----------
        f : function, dolfin.function, or any object that is callable with one or two inputs
            The new function we want to interface.


        Returns
        -------


        Attributes
        ----------
        f
        __f_vector


        REFERENCES
        ----------

        """

        """
        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        self.f = f_new

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


class PythonFunction2D():
    r"""
    This class sets up a function object in order to unify the interface between different kinds of functions.

    Usage
    -----
    .. code-block:: python

        Function(f)


    Parameters
    ----------
    f : python function in 2D
        The python function in 2D we want to interface. When evaluating the object generated using function, it evaluates
        function.

    Attributes
    ----------
    f : python function in 2D
        The python function in 2D we want to interface. When evaluating the object generated using function, it evaluates
        function.

    gometric_dimension : int
                         The geometric dimension of self.function

    References
    ----------



    TODO
    ----

    ....

    :First Added:   2015-12-07
    :Last Modified: 2015-12-07
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version
    """

    """
    Reviews:
        1. First implementation. (apalha, 2015-12-07)
    """

    def __init__(self, f):
        """

        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing = {} # initialize timing dictionary, this is only done in __init__
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        # store the inputs as attributes
        self.f = f
        self.type = 'python_function'

        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------

    def __call__(self, x, y):
        r"""
        Function that evaluates the function at a set of N points

        Usage
        -----
        .. code-block :: python

            f_evaluated = self(x)


        Parameters
        ----------
        x : numpy.array, (N,1) for 1D points or tuple of numpy.array, (N,1) for nD points
            The set of N one-dimensional points where to evaluate the function, self.f


        Returns
        -------
        f_evaluated : numpy.array, (N,1)
                      The evaluation of the function, self.f, at the set of N points x.


        Attributes
        ----------


        REFERENCES
        ----------

        """

        """
        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        f_evaluated = self.f(x,y)

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------

        return f_evaluated

    def geometric_dimension(self):
        r"""
        Function that returns the geometric dimension of the function.

        Usage
        -----
        .. code-block :: python

            geometric_dimension = self.geometric_dimension()


        Parameters
        ----------


        Returns
        -------
        geometric_dimension : int, single value
                              The geometric dimension of the function.


        Attributes
        ----------


        REFERENCES
        ----------

        """

        """
        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        """
        return 2

    def update(self,f_new):
        r"""
        Update the function self.f with f_new in an efficient way.

        Usage
        -----
        .. code-block :: python

            self.update(f_new)


        Parameters
        ----------
        f : function, dolfin.function, or any object that is callable with one or two inputs
            The new function we want to interface.


        Returns
        -------


        Attributes
        ----------
        f
        __f_vector


        REFERENCES
        ----------

        """

        """
        :First Added:   2015-12-07
        :Last Modified: 2015-12-07
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-07)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        self.f = f_new

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


class PlasmaShape():
    r"""
    This class sets up known plasma shapes for the fixed boundary Grad-Shafranov solver.

    Implements the generation of the plasma boundary (self.boundary) and plasma mesh (self.mesh).

    Usage
    -----
    .. code-block:: python

        PlasmaShape(shape,n=20)


    Parameters
    ----------
    shape : string
        The identification of the plasma shape to generate.
        The codes identifying the different shapes are: ::
            "pataki_iter_1" : ITER like smooth plasma shape as presented in [pataki2013] (circle).
            "pataki_iter_2" : ITER like smooth plasma shape as presented in [pataki2013] (intermediate elongation).
            "pataki_iter_3" : ITER like smooth plasma shape as presented in [pataki2013] (full elongation).
            "pataki_nstx" : NSTX like smooth plasma shape as presented in [pataki2013].
            "xpoint_iter" : ITER like x-point plasma shape as presented in [cerfon2010].

    n : int
        The number of points on the boundary to generate.

    Attributes
    ----------
    shape : string
        The identification of the plasma shape to generate.
        The codes identifying the different shapes are: ::
            "pataki_iter_0" : Small circle plasma shape corresponding to a large aspect ratio [pataki2013].
            "pataki_iter_1" : ITER like smooth plasma shape as presented in [pataki2013] (circle).
            "pataki_iter_2" : ITER like smooth plasma shape as presented in [pataki2013] (intermediate elongation).
            "pataki_iter_3" : ITER like smooth plasma shape as presented in [pataki2013] (full elongation).
            "pataki_nstx" : NSTX like smooth plasma shape as presented in [pataki2013].
            "xpoint_iter" : ITER like x-point plasma shape as presented in [cerfon2010].

    n : int
        The number of points on the boundary to generate.

    r : numpy.array(float64), [1,n]
        The r coordinates of the points on the boundary of the plasma.

    z : numpy.array(float64), [1,n]
        The z coordinates of the points on the boundary of the plasma.


    References
    ----------
    .. [pataki2013] Pataki, A., Cerfon, A. J., Freidberg, J. P., Greengard, L., and O’Neil, M. (2013).
                    A fast, high-order solver for the Grad-Shafranov equation.
                    Journal of Computational Physics, 243, 28–45. http://doi.org/10.1016/j.jcp.2013.02.045
    .. [cerfon2010] Cerfon, A. J., & Freidberg, J. P. (2010).
                    ``One size fits al’' analytic solutions to the Grad-Shafranov equation.
                    Physics of Plasmas, 17(3), 032502. http://doi.org/10.1063/1.3328818



    TODO
    ----

    ....

    :First Added:   2015-12-01
    :Last Modified: 2015-12-01
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version
    """

    """
    Reviews:
        1. First implementation. (apalha, 2015-12-01)
    """

    def __init__(self, shape, n=20):
        """
        """

        """
        :First Added:   2015-12-01
        :Last Modified: 2015-12-01
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-01)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing = {} # initialize timing dictionary, this is only done in __init__
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        # store inputs
        self.shape = shape
        self.n = n

        # generate the n points at the boundary
        self.generate_boundary(self.n)

        # generate the mesh of the plasma domain based on the boundary
        self.generate_mesh()


        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


    def generate_boundary(self,n):
        r"""
        Compute the piecewise definition of the boundary of the plasma as an array of n points.

        Usage
        -----
        .. code-block :: python

            self.boundary(n)


        Parameters
        ----------
        n : int
            The number of points on the boundary to generate.


        Returns
        -------


        Attributes
        ----------
        n
        r
        z


        :First Added:   2015-12-01
        :Last Modified: 2015-12-01
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version

        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-01)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        self.n = n # update the number of points

        # compute the n boundary points
        if self.shape == 'pataki_iter_0':
            self.r, self.z = plasma_boundary_pataki(self.n,0.00001,1.0,0.0)
        elif self.shape == 'pataki_iter_1':
            self.r, self.z = plasma_boundary_pataki(self.n,0.32,1.0,0.0)
        elif self.shape == 'pataki_iter_2':
            self.r, self.z = plasma_boundary_pataki(self.n,0.32,1.0,0.33)
        elif self.shape == 'pataki_iter_3':
            self.r, self.z = plasma_boundary_pataki(self.n,0.32,1.7,0.33)
        elif self.shape == 'pataki_nstx':
            self.r, self.z = plasma_boundary_pataki(self.n,0.78,2.0,0.35)
        elif self.shape == 'xpoint_iter':
            self.r, self.z = plasma_boundary_xpoint_iter(self.n)


        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


    def generate_mesh(self,n=None,show=False):
        r"""
        Compute mesh of of the plasma domain, using n vertices on the boundary.

        Usage
        -----
        .. code-block :: python

            self.generate_mesh(n=None,show=False)


        Parameters
        ----------
        n : int
            The number of points on the boundary to generate.

        show : bool
               A flag that determines of the mesh is plotted of not. Note that if gs.output_parameter['plotting'] = False
               no plots will be shown even if show = True.


        Returns
        -------


        Attributes
        ----------
        n
        r
        z
        mesh


        :First Added:   2015-12-02
        :Last Modified: 2015-12-02
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version

        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-02)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        if n != None:
            self.generate_boundary(n)

        # compute the maximum area size of the triangulation
        h = numpy.hypot(numpy.diff(self.r), numpy.diff(self.z)) # start by computing the length of the edges along the boundary
        hmin = h.min()
        hmax = h.max()
        tri_area = 0.25*(hmin+hmax)**2 # the maximum area is the area of an equilateral triangle of side equal to the
                                       # average between the maximum and minimum edge length along the boundary

        # compute the triangulation using triangle
        vertices = numpy.stack((self.r[0:-1],self.z[0:-1]),axis=1)                       # Start by generating the vertices
                                                                                         # at the boundary of the plasma
                                                                                         # in the format used by triangle.
                                                                                         # The last vertex is not used
                                                                                         # in order not to have repeated
                                                                                         # vertices.

        segment_markers = numpy.ones([self.r.size-1,1],dtype=numpy.int32) # there is only one segment marker, this value is arbitrary, so we choose 1

        vertex_markers = 2*numpy.ones([self.r.size-1,1],dtype=numpy.int32) # there is only one vertex marker, this value is arbitrary, so we choose 1

        segments = numpy.stack((numpy.arange(0,self.r.size-1,dtype=numpy.int32),numpy.arange(1,self.r.size,dtype=numpy.int32)),axis=1)
        segments[-1,-1] = 0

        boundary_definition = {'segment_markers':segment_markers,'segments':segments,'vertex_markers':vertex_markers,'vertices':vertices}
        triangle_mesh = triangle.triangulate(boundary_definition,'pq30a%.20f' %(tri_area)) # Triangulate with quality parameters:
                                                                                      # q30, meaning that the minimum angle
                                                                                      # of a triangle of the mesh is 30,
                                                                                      # and a%f %(tri_area), meaning that
                                                                                      # the maximum area of the triangles
                                                                                      # will be equal to tri_area.
        nvertices = triangle_mesh['vertices'].shape[0] # the number of vertices in the mesh
        ncells = triangle_mesh['triangles'].shape[0] # the number of cells in the mesh


        # create the dolfin mesh and open it for editing
        self.mesh = dolfin.Mesh()
        mesh_editor = dolfin.MeshEditor()
        mesh_editor.open(self.mesh,2,2) # the mesh has dimension 2 (surface) and topological dimension 2 also (surfaces)

        # add the vertices
        mesh_editor.init_vertices(nvertices)
        for k,vertex in enumerate(triangle_mesh['vertices']):
            mesh_editor.add_vertex(k, vertex[0], vertex[1])

        # Add the cells (triangular elements)
        mesh_editor.init_cells(ncells)
        for k,cell in enumerate(triangle_mesh['triangles']):
            mesh_editor.add_cell(k, cell[0], cell[1], cell[2])

        mesh_editor.close()

        if output_parameters['runtime_info']:
            print 'Created mesh with %d vertices and %d cells' % (nvertices, ncells)

        if output_parameters['plotting'] and show:
            dolfin.plot(self.mesh)
            dolfin.interactive()


        # timing end ---------------------------------------------------------------------------------------------------
        if output_parameters['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


# Functions -------------------------------------------------------------------

def boundary(x,on_boundary):
    r"""
    Function that identifies the boundary vertices needed to specify Dirichlet
    boundary conditions. This implementation is very simple because all the
    boundary is a Dirichlet boundary.

    Usage
    -----
    .. code-block :: python

        boundary(x,on_boundary)


    Parameters
    ----------
    x : a tuple or a numpy.array, size: (1,2)
        The coordinates of the point to identify if it lies on the boundary or not
    on_boundary : bool, size: single value
                  A flag identifying if the point lies on the boundary or not.

    Returns
    -------
    on_boundary : bool, size: single value
                  A flag identifying if the point lies on the boundary or not.

    :First Added:   2015-04-16
    :Last Modified: 2015-04-16
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2015-04-16)
    """
    return on_boundary


def plasma_boundary_pataki(n,epsilon,kappa,delta):
    r"""
    Function that returns a set of n points that defines the boundary of
    the pataki plasma shape test case [1].

    Usage
    -----
    .. code-block :: python

        plasma_boundary_pataki(n,epsilon,kappa,delta)


    Parameters
    ----------
    n : int
        The number of points on the boundary to generate.
    epsilon : float64
               As given in [1], related to the size of the horizontal axis of the plasma shape.
    kappa : float64
             As given in [1], related to the size of the vertical axis of the plasma shape (kappa*epsilon).
    delta : float64
            As given in [1], related to the misalignment between the top of the plasma shape and the  axis.
            of the plasma shape.


    Returns
    -------
    r : numpy.array(float64), [1,n]
        The r coordinates of the points in the boundary of the plasma.

    z : numpy.array(float64), [1,n]
        The z coordinates of the points in the boundary of the plasma.


    REFERENCES
    ----------
    .. [pataki2013] Pataki, A., Cerfon, A. J., Freidberg, J. P., Greengard, L., and O'Neil, M. (2013).
                    A fast, high-order solver for the Grad-Shafranov equation.
                    Journal of Computational Physics, 243, 28-45 doi:10.1016/j.jcp.2013.02.045


    :First Added:   2015-12-01
    :Last Modified: 2015-12-01
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2015-12-01)
    """

    alpha = numpy.arcsin(delta) # the alpha term appearing in [pataki2013]

    xi = numpy.linspace(0,2.0*numpy.pi,n) # the points along the angle-like variable, which will parametrise the points
                                          # on the boundary

    r = 1.0 + epsilon*numpy.cos(xi + alpha*numpy.sin(xi))
    z = epsilon*kappa*numpy.sin(xi)

    return r, z


def plasma_boundary_xpoint_iter(n):
    r"""
    Function that returns a set of n points that defines the boundary of
    the x-point plasma shape test case [palha2015].

    Usage
    -----
    .. code-block :: python

        plasma_boundary_xpoint(n)


    Parameters
    ----------
    n : int
        The number of points to generate on the boundary.

    Returns
    -------
    r : numpy.array(float64), [1,n]
        The r coordinates of the points in the boundary of the plasma.

    z : numpy.array(float64), [1,n]
        The z coordinates of the points in the boundary of the plasma.


    REFERENCES
    ----------
    .. [palha2015] A. Palha, B. Koren, and F. Felici, “A mimetic spectral element solver for the Grad-Shafranov
                   equation on curved meshes,” http://arxiv.org/abs/1512.05989,
                   submitted to Journal of Computational Physics, 2015.


    :First Added:   2016-01-06
    :Last Modified: 2016-01-06
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-01-06)
    """

    # The boundary of the x point plasma is made up of fours segments. For this reason we divide the number of points
    # by 4 and round the value. To still have n points along the boundary, the missing points are added to the last segment.
    n_per_segment = int(n/4) # recall that this will floor the result and return an int
    n_last_segment = n - 3*n_per_segment # the last segment has more points to compensate the rounding

    # allocate memory space for the point's coordinates
    r = numpy.zeros(n)
    z = numpy.zeros(n)

    # segment 1
    s = numpy.linspace(0.0,1.0,n_per_segment+1)
    cx = numpy.fliplr(numpy.array([[0.88, 0.5521148541571086, -0.20640336053451946, 2.4834931938984552, -5.03596728562579, 2.825936559606835, 0.32138363979983614, -0.5005576013019241]]))[0]
    cy = numpy.fliplr(numpy.array([[-0.6, 0.5999999999999996, 0.0, 0.0, 0.0, 0.0, 0.0]]))[0]

    r[0:n_per_segment] = numpy.polyval(cx,s[0:-1]) # s[0:-1] because boundary points should not be repeated
    z[0:n_per_segment] = numpy.polyval(cy,s[0:-1])

    # segment 2
    s = numpy.linspace(0.0,1.0,n_per_segment+1)
    cx = numpy.fliplr(numpy.array([[1.3200000000000014, 0., -0.42243831504242935, 1.2501879617549474, -5.368764301947667, 11.182100647055545, -11.580478496134559, 4.499392504314161]]))[0]
    cy = numpy.fliplr(numpy.array([[0., 0.8891881373291781, -4.194429431647997, 22.70616098798567, -59.94304165245803, 82.42892663380348, -56.412807237548165, 15.069366375444783]]))[0]

    r[n_per_segment:(2*n_per_segment)] = numpy.polyval(cx,s[0:-1]) # s[0:-1] because boundary points should not be repeated
    z[n_per_segment:(2*n_per_segment)] = numpy.polyval(cy,s[0:-1])

    # segment 3
    s = numpy.linspace(1.0,0.0,n_per_segment+1) # this has to be flipped because the natural order of the points is the other way
    cx = numpy.fliplr(numpy.array([[0.6799999999999871, 0., 0.16248409671487218, -1.14462836550234, 5.240947487862182, -11.313789484633782, 11.659292015452298,-4.404305749893218]]))[0]
    cy = numpy.fliplr(numpy.array([[0.0, 0.8891881373291746, -4.194429431648004, 22.706160987985644, -59.943041652457914, 82.42892663380331, -56.412807237548044, 15.06936637544475]]))[0]

    r[(2*n_per_segment):(3*n_per_segment)] = numpy.polyval(cx,s[0:-1]) # s[0:-1] because boundary points should not be repeated
    z[(2*n_per_segment):(3*n_per_segment)] = numpy.polyval(cy,s[0:-1])

    # segment 4
    s = numpy.linspace(1.0,0.0,n_last_segment) # this has to be flipped because the natural order of the points is the other way
    cx = numpy.fliplr(numpy.array([[0.88, -0.36475148713106564, 0.15669462964640002, -0.9082699105455433, 2.8740467630422044, -3.379335261756935, 1.7746456511237334, -0.35303038437880685]]))[0]
    cy = numpy.fliplr(numpy.array([[-0.6, 0.5999999999999993, 0., 0., 0., 0., 0.]]))[0]

    r[(3*n_per_segment):n] = numpy.polyval(cx,s)
    z[(3*n_per_segment):n] = numpy.polyval(cy,s)

    return r, z


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