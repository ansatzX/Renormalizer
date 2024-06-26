#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Extension to scipy.linalg module
'''

import sys
import warnings
import tempfile

import numpy
np = numpy
import scipy.linalg
import h5py

from renormalizer.lib.davidson import logger


SAFE_EIGH_LINDEP = 1e-15
DAVIDSON_LINDEP = 1e-14
DSOLVE_LINDEP = 1e-15
MAX_MEMORY = 2000

# sort by similarity has problem which flips the ordering of eigenvalues when
# the initial guess is closed to excited state.  In this situation, function
# _sort_by_similarity may mark the excited state as the first eigenvalue and
# freeze the first eigenvalue.
SORT_EIG_BY_SIMILARITY = False
# Projecting out converged eigenvectors has problems when conv_tol is loose.
# In this situation, the converged eigenvectors may be updated in the
# following iterations.  Projecting out the converged eigenvectors may lead to
# large errors to the yet converged eigenvectors.
PROJECT_OUT_CONV_EIGS = False

FOLLOW_STATE = False


def _fill_heff_hermitian(heff, xs, ax, xt, axt, dot):
    nrow = len(axt)
    row1 = len(ax)
    row0 = row1 - nrow
    for ip, i in enumerate(range(row0, row1)):
        for jp, j in enumerate(range(row0, i)):
            heff[i,j] = dot(xt[ip].conj(), axt[jp])
            heff[j,i] = heff[i,j].conj()
        heff[i,i] = dot(xt[ip].conj(), axt[ip]).real

    for i in range(row0):
        axi = numpy.asarray(ax[i])
        for jp, j in enumerate(range(row0, row1)):
            heff[j,i] = dot(xt[jp].conj(), axi)
            heff[i,j] = heff[j,i].conj()
        axi = None
    return heff


def davidson(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
             lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
             dot=numpy.dot, callback=None,
             nroots=1, lessio=False, pick=None, verbose=logger.WARN,
             follow_state=FOLLOW_STATE):
    r'''Davidson diagonalization method to solve  a c = e c.  Ref
    [1] E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    [2] http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Note: This function has an overhead of memory usage ~4*x0.size*nroots

    Args:
        aop : function(x) => array_like_x
            aop(x) to mimic the matrix vector multiplication :math:`\sum_{j}a_{ij}*x_j`.
            The argument is a 1D array.  The returned value is a 1D array.
        x0 : 1D array or a list of 1D array
            Initial guess.  The initial guess vector(s) are just used as the
            initial subspace bases.  If the subspace is smaller than "nroots",
            eg 10 roots and one initial guess, all eigenvectors are chosen as
            the eigenvectors during the iterations.  The first iteration has
            one eigenvector, the next iteration has two, the third iterastion
            has 4, ..., until the subspace size > nroots.
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value
        lessio : bool
            How to compute a*x0 for current eigenvector x0.  There are two
            ways to compute a*x0.  One is to assemble the existed a*x.  The
            other is to call aop(x0).  The default is the first method which
            needs more IO and less computational cost.  When IO is slow, the
            second method can be considered.
        pick : function(w,v,nroots) => (e[idx], w[:,idx], idx)
            Function to filter eigenvalues and eigenvectors.
        follow_state : bool
            If the solution dramatically changes in two iterations, clean the
            subspace and restart the iteration with the old solution.  It can
            help to improve numerical stability.  Default is False.

    Returns:
        e : float or list of floats
            Eigenvalue.  By default it's one float number.  If :attr:`nroots` > 1, it
            is a list of floats for the lowest :attr:`nroots` eigenvalues.
        c : 1D array or list of 1D arrays
            Eigenvector.  By default it's a 1D array.  If :attr:`nroots` > 1, it
            is a list of arrays for the lowest :attr:`nroots` eigenvectors.
    '''
    e, x = davidson1(lambda xs: [aop(x) for x in xs],
                     x0, precond, tol, max_cycle, max_space, lindep,
                     max_memory, dot, callback, nroots, lessio, pick, verbose,
                     follow_state)[1:]
    if nroots == 1:
        return e[0], x[0]
    else:
        return e, x


def davidson1(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
              lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
              dot=numpy.dot, callback=None,
              nroots=1, lessio=False, pick=None, verbose=logger.WARN,
              follow_state=FOLLOW_STATE, tol_residual=None,
              fill_heff=_fill_heff_hermitian):
    r'''Davidson diagonalization method to solve  a c = e c.  Ref
    [1] E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    [2] http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Note: This function has an overhead of memory usage ~4*x0.size*nroots

    Args:
        aop : function([x]) => [array_like_x]
            Matrix vector multiplication :math:`y_{ki} = \sum_{j}a_{ij}*x_{jk}`.
        x0 : 1D array or a list of 1D arrays or a function to generate x0 array(s)
            Initial guess.  The initial guess vector(s) are just used as the
            initial subspace bases.  If the subspace is smaller than "nroots",
            eg 10 roots and one initial guess, all eigenvectors are chosen as
            the eigenvectors during the iterations.  The first iteration has
            one eigenvector, the next iteration has two, the third iterastion
            has 4, ..., until the subspace size > nroots.
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value
        lessio : bool
            How to compute a*x0 for current eigenvector x0.  There are two
            ways to compute a*x0.  One is to assemble the existed a*x.  The
            other is to call aop(x0).  The default is the first method which
            needs more IO and less computational cost.  When IO is slow, the
            second method can be considered.
        pick : function(w,v,nroots) => (e[idx], w[:,idx], idx)
            Function to filter eigenvalues and eigenvectors.
        follow_state : bool
            If the solution dramatically changes in two iterations, clean the
            subspace and restart the iteration with the old solution.  It can
            help to improve numerical stability.  Default is False.

    Returns:
        conv : bool
            Converged or not
        e : list of floats
            The lowest :attr:`nroots` eigenvalues.
        c : list of 1D arrays
            The lowest :attr:`nroots` eigenvectors.
    '''
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if tol_residual is None:
        toloose = numpy.sqrt(tol)
    else:
        toloose = tol_residual
    log.debug1('tol %g  toloose %g', tol, toloose)

    if not callable(precond):
        precond = make_diag_precond(precond)

    if callable(x0):  # lazy initialization to reduce memory footprint
        x0 = x0()
    if isinstance(x0, numpy.ndarray) and x0.ndim == 1:
        x0 = [x0]
    #max_cycle = min(max_cycle, x0[0].size)
    max_space = max_space + (nroots-1) * 3
    # max_space*2 for holding ax and xs, nroots*2 for holding axt and xt
    _incore = max_memory*1e6/x0[0].nbytes > max_space*2+nroots*3
    lessio = lessio and not _incore
    log.debug1('max_cycle %d  max_space %d  max_memory %d  incore %s',
               max_cycle, max_space, max_memory, _incore)
    dtype = None
    heff = None
    fresh_start = True
    e = 0
    v = None
    conv = [False] * nroots
    emin = None
    norm_min = 1

    for icyc in range(max_cycle):
        if fresh_start:
            if _incore:
                xs = []
                ax = []
            else:
                xs = _Xlist()
                ax = _Xlist()
            space = 0
# Orthogonalize xt space because the basis of subspace xs must be orthogonal
# but the eigenvectors x0 might not be strictly orthogonal
            xt = None
            x0len = len(x0)
            xt = _qr(x0, dot, lindep)[0]
            if len(xt) != x0len:
                log.warn('QR decomposition removed %d vectors.  The davidson may fail.',
                         x0len - len(xt))
                if callable(pick):
                    log.warn('Check to see if `pick` function %s is providing '
                             'linear dependent vectors', pick.__name__)
                if len(xt) == 0:
                    if icyc == 0:
                        msg = 'Initial guess is empty or zero'
                    else:
                        msg = ('No more linearly independent basis were found. '
                               'Unless loosen the lindep tolerance (current value '
                               f'{lindep}), the diagonalization solver is not able '
                               'to find eigenvectors.')
                    raise LinearDependenceError(msg)
            x0 = None
            max_dx_last = 1e9
            if SORT_EIG_BY_SIMILARITY:
                conv = [False] * nroots
        elif len(xt) > 1:
            xt = _qr(xt, dot, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most

        axt = aop(xt)
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        rnow = len(xt)
        head, space = space, space+rnow

        if dtype is None:
            try:
                dtype = numpy.result_type(axt[0], xt[0])
            except IndexError:
                raise LinearDependenceError('No linearly independent basis found '
                                            'by the diagonalization solver.')
        if heff is None:  # Lazy initilize heff to determine the dtype
            heff = numpy.empty((max_space+nroots,max_space+nroots), dtype=dtype)
        else:
            heff = numpy.asarray(heff, dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv

        fill_heff(heff, xs, ax, xt, axt, dot)
        xt = axt = None
        w, v = scipy.linalg.eigh(heff[:space,:space])
        if callable(pick):
            w, v, idx = pick(w, v, nroots, locals())
            if len(w) == 0:
                raise RuntimeError(f'Not enough eigenvalues found by {pick}')

        e = w[:nroots]
        v = v[:,:nroots]

        x0 = None
        x0 = _gen_x0(v, xs)
        if lessio:
            ax0 = aop(x0)
        else:
            ax0 = _gen_x0(v, ax)

        if SORT_EIG_BY_SIMILARITY:
            dx_norm = [0] * nroots
            xt = [None] * nroots
            for k, ek in enumerate(e):
                if not conv[k]:
                    xt[k] = ax0[k] - ek * x0[k]
                    dx_norm[k] = numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                    if abs(de[k]) < tol and dx_norm[k] < toloose:
                        log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                                  k, dx_norm[k], ek, de[k])
                        conv[k] = True
        else:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v,
                                           fresh_start, log)
            de = e - elast
            dx_norm = []
            xt = []
            conv = [False] * nroots
            for k, ek in enumerate(e):
                xt.append(ax0[k] - ek * x0[k])
                dx_norm.append(numpy.sqrt(dot(xt[k].conj(), xt[k]).real))
                conv[k] = abs(de[k]) < tol and dx_norm[k] < toloose
                if conv[k] and not conv_last[k]:
                    log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                              k, dx_norm[k], ek, de[k])
        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = numpy.argmax(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide])
            break
        elif (follow_state and max_dx_norm > 1 and
              max_dx_norm/max_dx_last > 3 and space > nroots+2):
            log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide], norm_min)
            log.debug('Large |r| detected, restore to previous x0')
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue

        if SORT_EIG_BY_SIMILARITY:
            if any(conv) and e.dtype == numpy.double:
                emin = min(e)

        # remove subspace linear dependency
        if any(((not conv[k]) and n**2>lindep) for k, n in enumerate(dx_norm)):
            for k, ek in enumerate(e):
                if (not conv[k]) and dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1/numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
        else:
            for k, ek in enumerate(e):
                if dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1/numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
                    log.debug1('Throwing out eigenvector %d with norm=%4.3g', k, dx_norm[k])
        xt = [xi for xi in xt if xi is not None]

        for i in range(space):
            xsi = numpy.asarray(xs[i])
            for xi in xt:
                xi -= xsi * dot(xsi.conj(), xi)
            xsi = None
        norm_min = 1
        for i,xi in enumerate(xt):
            norm = numpy.sqrt(dot(xi.conj(), xi).real)
            if norm**2 > lindep:
                xt[i] *= 1/norm
                norm_min = min(norm_min, norm)
            else:
                xt[i] = None
        xt = [xi for xi in xt if xi is not None]
        xi = None
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max_dx_norm, e, de[ide], norm_min)
        if len(xt) == 0:
            log.debug('Linear dependency in trial subspace. |r| for each state %s',
                      dx_norm)
            conv = [conv[k] or (norm < toloose) for k,norm in enumerate(dx_norm)]
            break

        max_dx_last = max_dx_norm
        fresh_start = space+nroots > max_space

        if callable(callback):
            callback(locals())

    x0 = [x for x in x0]  # nparray -> list

    # Check whether the solver finds enough eigenvectors.
    h_dim = x0[0].size
    if len(x0) < min(h_dim, nroots):
        # Two possible reasons:
        # 1. All the initial guess are the eigenvectors. No more trial vectors
        # can be generated.
        # 2. The initial guess sits in the subspace which is smaller than the
        # required number of roots.
        msg = 'Not enough eigenvectors (len(x0)=%d, nroots=%d)' % (len(x0), nroots)
        warnings.warn(msg)

    return numpy.asarray(conv), e, x0


def make_diag_precond(diag, level_shift=0):
    '''Generate the preconditioner function with the diagonal function.'''
    def precond(dx, e, *args):
        diagd = diag - (e - level_shift)
        diagd[abs(diagd)<1e-8] = 1e-8
        return dx/diagd
    return precond


def _qr(xs, dot, lindep=1e-14):
    '''QR decomposition for a list of vectors (for linearly independent vectors only).
    xs = (r.T).dot(qs)
    '''
    nvec = len(xs)
    dtype = xs[0].dtype
    qs = numpy.empty((nvec,xs[0].size), dtype=dtype)
    rmat = numpy.empty((nvec,nvec), order='F', dtype=dtype)

    nv = 0
    for i in range(nvec):
        xi = numpy.array(xs[i], copy=True)
        rmat[:,nv] = 0
        rmat[nv,nv] = 1
        for j in range(nv):
            prod = dot(qs[j].conj(), xi)
            xi -= qs[j] * prod
            rmat[:,nv] -= rmat[:,j] * prod
        innerprod = dot(xi.conj(), xi).real
        norm = numpy.sqrt(innerprod)
        if innerprod > lindep:
            qs[nv] = xi/norm
            rmat[:nv+1,nv] /= norm
            nv += 1
    return qs[:nv], numpy.linalg.inv(rmat[:nv,:nv])

def _gen_x0(v, xs):
    space, nroots = v.shape
    x0 = numpy.einsum('c,x->cx', v[space-1], numpy.asarray(xs[space-1]))
    for i in reversed(range(space-1)):
        xsi = numpy.asarray(xs[i])
        for k in range(nroots):
            x0[k] += v[i,k] * xsi
    return x0


def _sort_elast(elast, conv_last, vlast, v, fresh_start, log):
    '''
    Eigenstates may be flipped during the Davidson iterations.  Reorder the
    eigenvalues of last iteration to make them comparable to the eigenvalues
    of the current iterations.
    '''
    if fresh_start:
        return elast, conv_last
    head, nroots = vlast.shape
    ovlp = abs(numpy.dot(v[:head].conj().T, vlast))
    idx = numpy.argmax(ovlp, axis=1)

    if log.verbose >= logger.DEBUG:
        ordering_diff = (idx != numpy.arange(len(idx)))
        if numpy.any(ordering_diff):
            log.debug('Old state -> New state')
            for i in numpy.where(ordering_diff)[0]:
                log.debug('  %3d     ->   %3d ', idx[i], i)

    return [elast[i] for i in idx], [conv_last[i] for i in idx]


class LinearDependenceError(RuntimeError):
    pass


class H5TmpFile(h5py.File):
    '''Create and return an HDF5 temporary file.
    Kwargs:
        filename : str or None
            If a string is given, an HDF5 file of the given filename will be
            created. The temporary file will exist even if the H5TmpFile
            object is released.  If nothing is specified, the HDF5 temporary
            file will be deleted when the H5TmpFile object is released.
    The return object is an h5py.File object. The file will be automatically
    deleted when it is closed or the object is released (unless filename is
    specified).
    '''
    def __init__(self, filename=None, mode='a', *args, **kwargs):
        if filename is None:
            tmpfile = tempfile.NamedTemporaryFile(dir=".")
            filename = tmpfile.name
        h5py.File.__init__(self, filename, mode, *args, **kwargs)
#FIXME: Does GC flush/close the HDF5 file when releasing the resource?
# To make HDF5 file reusable, file has to be closed or flushed
    def __del__(self):
        try:
            self.close()
        except AttributeError:  # close not defined in old h5py
            pass
        except ValueError:  # if close() is called twice
            pass
        except ImportError:  # exit program before de-referring the object
            pass


class _Xlist(list):
    def __init__(self):
        self.scr_h5 = H5TmpFile()
        self.index = []

    def __getitem__(self, n):
        key = self.index[n]
        return self.scr_h5[str(key)]

    def append(self, x):
        length = len(self.index)
        key = length + 1
        index_set = set(self.index)
        if key in index_set:
            key = set(range(length)).difference(index_set).pop()
        self.index.append(key)

        self.scr_h5[str(key)] = x
        self.scr_h5.flush()

    def extend(self, x):
        for xi in x:
            self.append(xi)

    def __setitem__(self, n, x):
        key = self.index[n]
        self.scr_h5[str(key)][:] = x
        self.scr_h5.flush()

    def __len__(self):
        return len(self.index)

    def pop(self, index):
        key = self.index.pop(index)
        del (self.scr_h5[str(key)])

del (SAFE_EIGH_LINDEP, DAVIDSON_LINDEP, DSOLVE_LINDEP, MAX_MEMORY)
