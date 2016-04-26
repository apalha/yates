# -*- coding: utf-8 -*-
""" Definition of auxiliary yates' functions

Description
-----------
This module implements special functions for fast and efficient evaluation and computation.


References
----------

....

:First added:  2015-04-13
:Last updated: 2015-12-08
:Copyright: Copyright(C) 2015 apalha
:License: GNU GPL version 3 or any later version
"""

__all__ = [
           'DolfinFunction', 'PythonFunction1D', 'PythonFunction2D'            # Classes
          ]

import dolfin
import numpy
import time

from .. import config
from .system import *

# Classes --------------------------------------------------------------------------------------------------------------
#                                                                                                  <editor-fold desc="">


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
        if config.output['timing']:
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
        if config.output['timing']:
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
        if config.output['timing']:
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
        if config.output['timing']:
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
        if config.output['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        self.assign(f_new)
        if self.geometric_dimension() == 1:
            self.__f_at_x[self.__dofs2vertex] = self.vector().array()

        # timing start -------------------------------------------------------------------------------------------------
        if config.output['timing']:
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
        if config.output['timing']:
            self.timing = {} # initialize timing dictionary, this is only done in __init__
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        # store the inputs as attributes
        self.f = f
        self.type = 'python_function'

        # timing end ---------------------------------------------------------------------------------------------------
        if config.output['timing']:
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
        if config.output['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        f_evaluated = self.f(x)

        # timing start -------------------------------------------------------------------------------------------------
        if config.output['timing']:
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
        if config.output['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        self.f = f_new

        # timing start -------------------------------------------------------------------------------------------------
        if config.output['timing']:
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
        if config.output['timing']:
            self.timing = {} # initialize timing dictionary, this is only done in __init__
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        # store the inputs as attributes
        self.f = f
        self.type = 'python_function'

        # timing end ---------------------------------------------------------------------------------------------------
        if config.output['timing']:
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
        if config.output['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        f_evaluated = self.f(x,y)

        # timing start -------------------------------------------------------------------------------------------------
        if config.output['timing']:
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
        if config.output['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------

        self.f = f_new

        # timing start -------------------------------------------------------------------------------------------------
        if config.output['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------- </editor-fold>
