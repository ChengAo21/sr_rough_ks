# pce_utils.py
# Smolyak grid. Not called when employing least squares PCE

import numpy as np
from scipy.special import binom
from scipy import stats
import math


def cos(x):
    return np.cos(x)
def tanh(x):
    return np.tanh(x)
def exp(x):
    return np.exp(x)
def square(x):
    return np.square(x)
def abs(x):
    return np.abs(x)


def columnize(*args):

    column_arrays = []
    for ele in args:
        if isinstance(ele, (list, tuple)):
            ele = np.array(ele)
        ele.shape = (-1, 1)
        column_arrays.append(ele)

    return column_arrays


def clencurt(rule=5, method='FFT'):

    if method.upper() == 'FFT':
        
        if rule == 1:
            x = np.array([0])
            w = np.array([2])
        else:
            n = rule - 1
            c = np.zeros((rule, 2))
            c[:rule:2,0] = 2 / np.append([1], np.array([1-np.arange(2,rule,2)**2]))
            c[1,1] = 1
            f = np.fft.ifft(np.concatenate((c[:rule,:], c[-2:0:-1,:]),axis=0), axis=0).real
            w = 2 * np.concatenate(([f[0,0]], 2*f[1:rule-1,0], [f[rule-1,0]]))/2
            x = (rule-1) * f[:rule,1]
            x = x[-1::-1]
        return x, w
    
    elif method.upper() == 'EXPLICIT':
        if rule == 1:
            x = np.array([0])
            w = np.array([2])
        else:
            n = rule - 1
            # build points
            theta = np.arange(rule) * np.pi / n
            x = np.cos(theta)
            w = 0

            c = np.ones((rule))
            c[1:-1] = 2
            j = np.arange((n/2)//1) + 1
            
            b = 2 * np.ones(j.shape)
            if np.mod(n/2,1) == 0:
                b[int(n/2) - 1] = 1
            
            j.shape = (1, -1)
            b.shape = (1, -1)
            j_theta = j * theta.reshape(-1,1)
            w = c/n * (1 - np.sum(b/(4*j**2 - 1) * np.cos(2*j_theta), axis=1))

            x = np.round(x[-1::-1] * 1e13) / 1e13
        return x, w


def quad_coeff(rule=5, kind='GL'):

    if kind.upper() == 'GL':
        x, w = np.polynomial.legendre.leggauss(rule)
        
    elif kind.upper() == 'GH':
        x, w = np.polynomial.hermite.hermgauss(rule)
        
    elif kind.upper() == 'CC':
        x, w = clencurt(rule)

    return x, w


class PceSmolyakGrid():

    def __init__(self, polynom, level):
        
        (self.x,
         self.eps,
         self.weight,
         self.unique_x,
         self.inverse_index) = self.smolyak_sparse_grid(polynom, level)


    def smolyak_sparse_grid(self, polynom, level):

        # min and max level
        o_min = max([level + 1, polynom.dim])
        o_max = level + polynom.dim
        
        # get multi-index for all level
        comb = np.empty((0, polynom.dim), dtype=int)
        for k in range(o_min, o_max+1):
            multi_index, _ = self.index_with_sum(polynom.dim, k)
            comb = np.append(comb, multi_index, axis=0)
        
        # initialize final array
        x_points = np.empty((0, polynom.dim))
        eps_points = np.empty((0, polynom.dim))
        weights = np.empty((0,))

        # define integration points and weights
        for lev in comb:
            local_x = []
            local_eps = []
            local_w = []
            coeff = (-1)**(level + polynom.dim - np.sum(lev)) * binom(polynom.dim - 1, level + polynom.dim - np.sum(lev))
            
            # cycle on integration variables
            for l, k, p in zip(lev, polynom.distrib, polynom.param):
                # set integration type depending on distrib
                if k.upper() == 'U':
                    kind = 'CC'
                elif k.upper() == 'N':
                    kind = 'GH'
                
                # get number of integration points
                n = self.growth_rule(l, kind=kind)
                # get gauss points and weight
                x0, w0 = quad_coeff(rule=n, kind=kind)

                # change of variable
                if (k.upper() == 'U'):
                    eps = np.copy(x0)
                    w = np.copy(w0) * 0.5
                    if p != [-1, 1]:
                        x = (p[1] - p[0]) / 2 * x0 + (p[1] + p[0]) / 2
                    else:
                        x = np.copy(x0)
                    
                elif (k.upper() == 'N'):
                    eps = np.sqrt(2) * 1 * x0 + 0 
                    x = eps * p[1] + p[0]
                    w = w0 * (1 / np.sqrt(np.pi))

                # store local points
                local_x.append(x)
                local_eps.append(eps)
                local_w.append(w)
            
            # update final array
            x_points = np.concatenate((x_points, np.concatenate(columnize(*np.meshgrid(*local_x)), axis=1)))
            eps_points = np.concatenate((eps_points, np.concatenate(columnize(*np.meshgrid(*local_eps)), axis=1)))
            weights = np.concatenate((weights, coeff * np.prod(np.concatenate(columnize(*np.meshgrid(*local_w)), axis=1),axis=1)))
        
        # get unique points
        unique_x_points, inverse_index = np.unique(x_points, axis=0, return_inverse=True)

        return x_points, eps_points, weights, unique_x_points, inverse_index


    def index_with_sum(self, dim, val):

        # check feasibility
        if val < dim:
            raise ValueError(f'dim={dim} --> minimum value for val=dim={dim} (here {val} is provided)')

        k = np.ones(dim,dtype=int)
        khat = k * (val - dim + 1)

        index = np.empty((0, dim), dtype=int)

        cnt = 0
        p = 0

        while k[dim - 1] < val:
            if k[p] > khat[p]:
                if (p + 1) != dim:
                    k[p] = 1
                    p += 1
            else:
                khat[:p] = khat[p] - k[p] + 1
                k[0] = khat[0]
                p = 0
                cnt += 1
                index = np.append(index, k.reshape(1,-1), axis=0)
            
            k[p] = k[p] + 1
        return index, cnt


    def growth_rule(self, level, kind='Legendre'):

        method_map = {'GL':1, 'GH':1, 'CC':2}

        if method_map[kind.upper()] == 1: 
            n = 2*level - 1
        elif method_map[kind.upper()] == 2:
            n = 1 if level == 1 else 2**(level - 1) + 1
        
        return n