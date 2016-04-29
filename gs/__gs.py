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
           'GS_Solver'
          ]

import dolfin
import numpy
import scipy.sparse
import scipy.sparse.linalg

import dolfinmplot
import matplotlib.tri as tri

import time

from .. import constants
from .. import config
from .. import aux



# Constants ---------------------------------------------------------
#                                               <editor-fold desc="">

# -- Module wide constants ------------------------------------------
#                                               <editor-fold desc="">

DEFAULT_SOLVER_PARAMETERS = constants.default_parameters.GS_SOLVER

# ---------------------------------------------------- </editor-fold>


# -- Physical constants ---------------------------------------------
#                                               <editor-fold desc="">

MU0 = constants.physics.MU0 # vacuum permeability
MU0_INV = constants.physics.MU0_INV # inverse of vacuum permeability

# ---------------------------------------------------- </editor-fold>


# ---------------------------------------------------- </editor-fold>



# Classes -----------------------------------------------------------
#                                               <editor-fold desc="">

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
        if config.output['timing']:
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
        self.psi = aux.DolfinFunction(self._V)
        if self.solver_parameters['linear_algebra_backend'] == 'scipy':
            self.psi_vector = numpy.ones(self.psi.vector().array().shape)
        else:
            self.psi_vector = self.psi.vector()

        # if a newton solver is used to solve for the nonlinearity then we
        # need to solve for the variation between two iterations, delta_psi
        if self.solver_parameters['nonlinear_solver'] == 'Newton':
            self.delta_psi = aux.DolfinFunction(self._V)
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
            self._b_vector = numpy.zeros(self.b.vector().array().shape)

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

        # force the initialization of self.psi_vector to be 1
        self.psi_vector[:] = 1.0

        # timing end ---------------------------------------------------------------------------------------------------
        if config.output['timing']:
            self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------

    def solve_step(self, j, dj=None, sigma=1.0, bc_values=0.0):
        r"""
        Solve the Grad-Shafranov equation given the right hand side j (current sources). This function assumes j to
        not depend on \psi. This function therefore can be used to solve one step of a nonlinear
        Grad-Shafranov equation used, for example in fixed point iteration or Newton.

        Usage
        -----
        .. code-block :: python

            self.solve_step(j,dj=None,sigma=1.0,bc_values=None)


        Parameters
        ----------
        j  : python function evaluatable at \psi and (r,z), j(r,z,psi)
             Represents the toroidal current density
        dj : python function evaluatable at \psi and (r,z), dj(r,z,psi)
             Represents the derivative of j with respect to psi
        sigma : float64, single value
                The eigenvalue to use when solve an Grad-Shafranov eigenvalue problem.
        bc_values : numpy.array, self.bc_dof.shape
                    Contains the values of the solution at the boundary. These values must correspond to
                    the points in self.bc_coordinates

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
        if config.output['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        if self.solver_parameters['nonlinear_solver'] == 'Picard':
            self.__solve_step_picard(j,sigma=sigma, bc_values=bc_values)

        elif self.solver_parameters['nonlinear_solver'] == 'Newton':
            self.__solve_step_newton(j,dj,sigma=sigma, bc_values=bc_values)


        # timing end ---------------------------------------------------------------------------------------------------
        if config.output['timing']:
            self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


    def __solve_step_picard(self, j, sigma, bc_values):
        r"""
        Solve one step of the Picard iteration of the nonlinear Grad-Shafranov equation
        given the right hand side j (current sources).

        Usage
        -----
        .. code-block :: python

            self.solve_step(j,sigma=1.0,bc_values=None)


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
        if bc_values == None:
            self._b_vector[self.bc_dof] = 0.0
        else:
            self._b_vector[self.bc_dof] = bc_values


        # solve one step
        self._solver.solve(self.psi_vector, self._b_vector)


    def __solve_step_newton(self, j, dj, sigma, bc_values):
        r"""
        Solve one step of the Newton iteration of the nonlinear Grad-Shafranov equation
        given the right hand side j (current sources).

        Usage
        -----
        .. code-block :: python

            self.solve_step(j,dj,sigma=1.0,bc_values=0.0)


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

        # force the boundary values of the solution to be the boundary condition
        self.psi_vector[self.bc_dof] = bc_values

        # use psi(r,z) and j(r,z,psi) to compute j(r,z)
        self.j_vector[:] = sigma*j(self._vertices_x,self._vertices_y,self.psi_vector.array())
        # use psi(r,z) and dj(r,z,psi) to compute dj(r,z)
        self.dj_vector[:] = sigma*dj(self._vertices_x,self._vertices_y,self.psi_vector.array())


        # setup the right hand side
        self._b_vector = dolfin.assemble(self.B_form) - (self.A*self.psi_vector)
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
        if config.output['timing']:
            self.timing = {} # initialize timing dictionary, this is only done in __init__
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        # set the default value for reusing factorization
        self.parameters['reuse_factorization'] = False


        # timing end ---------------------------------------------------------------------------------------------------
        if config.output['timing']:
            self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
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
        if config.output['timing']:
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
        if config.output['timing']:
            self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
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
        if config.output['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        # solve the system of equations using the pre-computed LU decomposition
        x[:] = self._operator_LU.solve(b)


        # timing end ---------------------------------------------------------------------------------------------------
        if config.output['timing']:
            self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------



#                                               <editor-fold desc="Commented out functions">
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
#         if config.output['timing']:
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
#         if config.output['timing']:
#             self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
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
#         if config.output['timing']:
#             start_time = time.time() # start the timer for this function
#         #---------------------------------------------------------------------------------------------------------------
#
#         f_evaluated = numpy.interp(x,self.__x,self.__f_at_x)
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if config.output['timing']:
#             self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
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
#         if config.output['timing']:
#             start_time = time.time() # start the timer for this function
#         #---------------------------------------------------------------------------------------------------------------
#
#         self.f.assign(f_new)
#         self.__f_vector = self.f.vector()
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if config.output['timing']:
#             self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
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
#         if config.output['timing']:
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
#         if config.output['timing']:
#             self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
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
#         if config.output['timing']:
#             start_time = time.time() # start the timer for this function
#         #---------------------------------------------------------------------------------------------------------------
#
#         xy = numpy.stack((x,y),axis=1)
#         probes = fenicstools.Probes(xy.flatten(),self.__V)
#         probes(self.f)
#         f_evaluated = probes.array()
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if config.output['timing']:
#             self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
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
#         if config.output['timing']:
#             start_time = time.time() # start the timer for this function
#         #---------------------------------------------------------------------------------------------------------------
#
#         self.f.assign(f_new)
#         self.__f_vector = self.f.vector()
#
#         # timing start -------------------------------------------------------------------------------------------------
#         if config.output['timing']:
#             self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
#         #---------------------------------------------------------------------------------------------------------------
#---------------------------------------------- </editor-fold>


# ---------------------------------------------------- </editor-fold>



# Functions ---------------------------------------------------------
#                                               <editor-fold desc="">


# -- Aux functions --------------------------------------------------
#                                               <editor-fold desc="">


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


#    ------------------------------------------------- </editor-fold>


# ---------------------------------------------------- </editor-fold>

