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
           'DolfinFunction', 'PythonFunction1D', 'PythonFunction2D',            # Classes
           'mesh2triang', 'mplot_cellfunction', 'mplot_function', 'plot', 'contour', 'mcontour_function' # functions
          ]

import dolfin
import numpy
import time
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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
            self.triangulation = mesh2triang(self.function_space().mesh())

        # timing end ---------------------------------------------------------------------------------------------------
        if config.output['timing']:
            self.timing[currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------

    def probe(self, **kwargs):
        #__call__
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




# Functions ------------------------------------------------------------------------------------------------------------
#                                                                                                  <editor-fold desc="">


def mesh2triang(mesh):
    r"""
        Converts a dolfin mesh into a matplotlib triangulation.

        Usage
        -----
        .. code-block :: python

            mesh2triang(mesh)


        Parameters
        ----------
        mesh : dolfin.Mesh
               The dolfin mesh to convert to matplotlib triangulation.

        Returns
        -------
        triangulation : matplotlib.tri.triangulation.Triangulation
                        The matplotlib triangulation of the dolfin mesh.

        :First Added:   Thu Mar 19 12:47:25 2015
        :Last Modified: Thu Mar 19 12:47:25 2015
        :Copyright:     Copyright (C) 2015 crichardson, apalha
        :License:       GNU GPL version 3 or any later version

    """

    """
        Reviews:
            1. First implementation. (crichardson)
    """

    xy = mesh.coordinates()
    triangulation = tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())
    return triangulation


def mplot_cellfunction(cellfn):
    r"""
        Plots a dolfin cell function in matplotlib. Triangles are colored with
        a single color.

        Usage
        -----
        .. code-block :: python

            mplot_cellfunction(cellfn)


        Parameters
        ----------
        cellfn : dolfin.CellFunction or
                 dolfin.MeshFunction of topological dimension 2 (cells)
                 The dolfin cell or mesh function to plot.

        Returns
        -------
        cellfn_plot : matplotlib.collections.PolyCollection
                      The matplotlib plot of the cell function.

        :First Added:   Thu Mar 19 12:47:25 2015
        :Last Modified: Thu Mar 19 12:47:25 2015
        :Copyright:     Copyright (C) 2015 crichardson, apalha
        :License:       GNU GPL version 3 or any later version

    """

    """
        Reviews:
            1. First implementation. (crichardson)
    """
    C = cellfn.array()
    tri = mesh2triang(cellfn.mesh())
    cellfn_plot = plt.tripcolor(tri, facecolors=C)
    return cellfn_plot


def mplot_function(f):
    r"""
        Plots a dolfin function in matplotlib. The following types of functions
        can be plotted:
            DG0
            scalar functions interpolated to vertices
            vector functions interpolated to vertices

        Usage
        -----
        .. code-block :: python

            mplot_function(f)


        Parameters
        ----------
        f : dolfin.Function
            The dolfin function to plot. Only 2D functions (scalar or vector
            valued) can be plotted.

        Returns
        -------
        f_plot : matplotlib.collections.PolyCollection (for scalar valued functions)
                 matplotlib.quiver.Quiver (for vector valued functions)
                 The matplotlib plots of the functions.

        :First Added:   Thu Mar 19 12:47:25 2015
        :Last Modified: Thu Dec 08 12:47:25 2015
        :Copyright:     Copyright (C) 2015 crichardson, apalha
        :License:       GNU GPL version 3 or any later version

    """

    """
        Reviews:
            1. First implementation. (crichardson)
            2. Added the option to plot DolfinFunction. (apalha, 2015-12-08)
    """

    mesh = f.function_space().mesh()
    if (mesh.geometry().dim() != 2):
        raise AttributeError('Mesh must be 2D')

    if isinstance(f,DolfinFunction):
        # DG0 cellwise function
        if f.vector().size() == mesh.num_cells():
            C = f.vector().array()
            return plt.tripcolor(f.triangulation, facecolors=C)
        # Scalar function, interpolated to vertices
        elif f.value_rank() == 0:
            C = f.compute_vertex_values(mesh)
            return plt.tripcolor(f.triangulation, C, shading='gouraud')
        # Vector function, interpolated to vertices
        elif f.value_rank() == 1:
            w0 = f.compute_vertex_values(mesh)
            if (len(w0) != 2*mesh.num_vertices()):
                raise AttributeError('Vector field must be 2D')
            X = mesh.coordinates()[:, 0]
            Y = mesh.coordinates()[:, 1]
            U = w0[:mesh.num_vertices()]
            V = w0[mesh.num_vertices():]
            return plt.quiver(X,Y,U,V)
    else:
        # DG0 cellwise function
        if f.vector().size() == mesh.num_cells():
            C = f.vector().array()
            return plt.tripcolor(mesh2triang(mesh), facecolors=C)
        # Scalar function, interpolated to vertices
        elif f.value_rank() == 0:
            C = f.compute_vertex_values(mesh)
            return plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
        # Vector function, interpolated to vertices
        elif f.value_rank() == 1:
            w0 = f.compute_vertex_values(mesh)
            if (len(w0) != 2*mesh.num_vertices()):
                raise AttributeError('Vector field must be 2D')
            X = mesh.coordinates()[:, 0]
            Y = mesh.coordinates()[:, 1]
            U = w0[:mesh.num_vertices()]
            V = w0[mesh.num_vertices():]
            return plt.quiver(X,Y,U,V)


def mcontour_function(f,levels=None,colors=None):
    r"""
        Plots the contour plot of a dolfin function in matplotlib. The following types of functions
        can be plotted:
            scalar functions interpolated to vertices

        Usage
        -----
        .. code-block :: python

            mcontour_function(f,levels=None,colors=None)


        Parameters
        ----------
        f : dolfin.Function, DolfinFunction
            The dolfin function to plot. Only 2D functions (scalar
            valued) can be plotted.
        levels: numpy.array, size: [1,N] or [N,1]
                The levels of the contours to plot.
        colors: string, size: single value
                Color and line type specification for the contours, as defined in pylab.

        Returns
        -------
        f_plot : matplotlib.collections.PolyCollection (for scalar valued functions)
                 matplotlib.quiver.Quiver (for vector valued functions)
                 The matplotlib plots of the functions.

        :First Added:   Tue Dec 08 12:47:25 2015
        :Last Modified: Tue Dec 08 12:47:25 2015
        :Copyright:     Copyright (C) 2015 crichardson, apalha
        :License:       GNU GPL version 3 or any later version

    """

    """
        Reviews:
            1. First implementation. (apalha, 2015-12-08)
    """

    mesh = f.function_space().mesh()
    if (mesh.geometry().dim() != 2):
        raise AttributeError('Mesh must be 2D')

    if isinstance(f,DolfinFunction):
        # DG0 cellwise function
        if f.vector().size() == mesh.num_cells():
            C = f.vector().array()
            return plt.tricontour(f.triangulation, C, levels=levels, colors=colors)
        # Scalar function, interpolated to vertices
        elif f.value_rank() == 0:
            C = f.compute_vertex_values(mesh)
            return plt.tricontour(f.triangulation, C, levels=levels, colors=colors)
    else:
        # DG0 cellwise function
        if f.vector().size() == mesh.num_cells():
            C = f.vector().array()
            return plt.tricontour(mesh2triang(mesh),  C, levels=levels, colors=colors)
        # Scalar function, interpolated to vertices
        elif f.value_rank() == 0:
            C = f.compute_vertex_values(mesh)
            return plt.tricontour(mesh2triang(mesh),  C, levels=levels, colors=colors)


# Plot a generic dolfin object (if supported)
def plot(obj):
    r"""
        Plots a dolfin object in matplotlib. The following types of objects
        can be plotted:
            dolfin.Mesh
            dolfin.CellFunction
            dolfin.Function
                DG0
                scalar functions interpolated to vertices
                vector functions interpolated to vertices
            DolfinFunction

        Usage
        -----
        .. code-block :: python

            plot(obj)


        Parameters
        ----------
        obj : dolfin.Function
              dolfin.CellFunction (Sizet, Double or Int)
              dolfin.Mesh
              DolfinPlot
              The dolfin object to plot.

        Returns
        -------
        obj_plot : matplotlib.collections.PolyCollection (for scalar valued functions)
                   matplotlib.quiver.Quiver (for vector valued functions)
                   The matplotlib plot of the object.

        :First Added:   Thu Mar 19 12:47:25 2015
        :Last Modified: Tue Dec 08 12:47:25 2015
        :Copyright:     Copyright (C) 2015 crichardson, apalha
        :License:       GNU GPL version 3 or any later version

    """

    """
        Reviews:
            1. First implementation. (crichardson)
            2. Added the option to plot DolfinFunction. (apalha, 2015-12-08)
    """
    plt.gca().set_aspect('equal')

    if isinstance(obj,DolfinFunction):
        obj_plot = mplot_function(obj)
        return obj_plot
    elif isinstance(obj, dolfin.Function):
        obj_plot = mplot_function(obj)
        return obj_plot
    elif isinstance(obj, dolfin.CellFunctionSizet):
        obj_plot = mplot_cellfunction(obj)
        return obj_plot
    elif isinstance(obj, dolfin.CellFunctionDouble):
        obj_plot = mplot_cellfunction(obj)
        return obj_plot
    elif isinstance(obj, dolfin.CellFunctionInt):
        obj_plot = mplot_cellfunction(obj)
        return obj_plot
    elif isinstance(obj, dolfin.Mesh):
        if (obj.geometry().dim() != 2):
            raise AttributeError('Mesh must be 2D')
        obj_plot = plt.triplot(mesh2triang(obj), color='#808080')
        return obj_plot

    raise AttributeError('Failed to plot %s'%type(obj))


# Plot a generic dolfin object (if supported)
def contour(obj,levels=None, colors=None):
    r"""
        Plots the contour plot of a dolfin object in matplotlib. The following types of objects
        can be plotted:
            dolfin.Function
                scalar functions interpolated to vertices
            DolfinFunction

        Usage
        -----
        .. code-block :: python

            plot(obj,levels=None,colors=None)


        Parameters
        ----------
        obj : dolfin.Function
                  scalar functions interpolated to vertices
              DolfinFunction
              The dolfin object to plot.
        levels: numpy.array, size: [1,N] or [N,1]
                The levels of the contours to plot.
        colors: string, size: single value
                Color and line type specification for the contours, as defined in pylab.

        Returns
        -------
        obj_plot : matplotlib.collections.PolyCollection (for scalar valued functions)
                   matplotlib.quiver.Quiver (for vector valued functions)
                   The matplotlib plot of the object.

        :First Added:   Tue Dec 08 12:47:25 2015
        :Last Modified: Tue Dec 08 12:47:25 2015
        :Copyright:     Copyright (C) 2015 crichardson, apalha
        :License:       GNU GPL version 3 or any later version

    """

    """
        Reviews:
            1. First implementation. (apalha, 2015-12-08)
    """

    if isinstance(obj,DolfinFunction):
        obj_plot = mcontour_function(obj,levels,colors)
        return obj_plot
    elif isinstance(obj, dolfin.Function):
        obj_plot = mcontour_function(obj,levels,colors)
        return obj_plot

    raise AttributeError('Failed to plot %s'%type(obj))


# ------------------------------------------------------------------------------------------------------- </editor-fold>
