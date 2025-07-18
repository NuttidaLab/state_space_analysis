"""Linear ballistic accumulator model."""

import math
import numpy as np
import pandas as pd
import scipy.stats as st
import pytensor.tensor as pt
from pytensor.scan import scan, reduce
from pytensor.compile import function

def normpdf(x):
    # 1/sqrt(2π) * exp(-x^2/2)
    return pt.exp(-0.5 * x**2) / pt.sqrt(2 * math.pi)

def normcdf(x):
    # 0.5 * [1 + erf(x/√2)]
    return 0.5 * (1 + pt.erf(x / math.sqrt(2)))

def tpdf(t, A, b, v, s):
    """Probability distribution function over time."""
    g = (b - A - t * v) / (t * s)
    h = (b - t * v) / (t * s)
    return (-v * normcdf(g) + s * normpdf(g)
            + v * normcdf(h) - s * normpdf(h)) / A

def tcdf(t, A, b, v, s):
    """Cumulative distribution function over time."""
    g = (b - A - t * v) / (t * s)
    h = (b - t * v) / (t * s)
    e1 = ((b - A - t * v) / A) * normcdf(g)
    e2 = ((b - t * v) / A) * normcdf(h)
    e3 = ((t * s) / A) * normpdf(g)
    e4 = ((t * s) / A) * normpdf(h)
    return 1 + e1 - e2 + e3 - e4

def resp_pdf(t, ind, A, b, v, s):

    m = v.type.shape[0]

    # 1) First‐passage densities for each accumulator: shape (m, n_trials)
    f = pt.stack([tpdf(t, A, b, vi, s) for vi in v], axis=0)

    # 2) CDFs for each accumulator
    F = pt.stack([tcdf(t, A, b, vi, s) for vi in v], axis=0)

    # 3) Probability that none ever finish
    p_zero = pt.prod(normcdf(-v / s))

    # 4) For each i: f_i * ∏_{j≠i} (1 − F_j)
    numer = pt.stack([
        f[i] * pt.prod(1 - pt.concatenate([F[:i], F[i+1:]]), axis=0)
        for i in range(m)
    ], axis=0)

    # 5) Select the density for the chosen accumulator and normalize
    pdf = numer[ind, pt.arange(t.shape[0])] / (1 - p_zero)

    # 6) Zero‐out any non‐positive RTs
    return pt.switch(pt.gt(t, 0), pdf, 0)

class LBA():
    """Linear Ballistic Accumulator model."""
    def tensor_pdf(self, rt, response, param):
        tau = param['tau']
        sub_param = param.copy()
        del sub_param['tau']
        return resp_pdf(rt - tau, response, **sub_param)

    def function_pdf(self):
        t = pt.dvector('t')
        i = pt.ivector('i')
        A = pt.dscalar('A')
        b = pt.dscalar('b')
        v = pt.dvector('v')
        s = pt.dscalar('s')
        tau = pt.dscalar('tau')
        
        pdf = resp_pdf(t - tau, i, A, b, v, s)
        
        return function([t, i, A, b, v, s, tau], pdf)

    def rvs_test(self, test, param, size):
        A, b, v, s, tau = param['A'], param['b'], param['v'], param['s'], param['tau']
        def sample_finish_time(A, b, v, s, tau, size):
            k = st.uniform.rvs(loc=0, scale=A, size=size)
            tmat = np.zeros((len(v), size))
            for idx, vi in enumerate(v):
                d = st.norm.rvs(loc=vi, scale=s, size=size)
                ti = tau + (b - k) / d
                ti[d < 0] = np.nan
                tmat[idx, :] = ti
            return tmat

        t = sample_finish_time(A, b, v, s, tau, size)
        valid = ~np.isnan(t).all(axis=0)
        t_sub = t[:, valid]
        rt = np.full(size, np.nan)
        resp = np.full(size, np.nan)
        rt[valid] = np.nanmin(t_sub, axis=0)
        resp[valid] = np.nanargmin(t_sub, axis=0)
        return rt, resp
    
    def rvs(self, test, param):
        n_trial = len(test)
        response = np.zeros(n_trial)
        rt = np.zeros(n_trial)
        test_types = np.unique(test)
        for this_test in test_types:
            ind = test == this_test
            test_rt, test_response = self.rvs_test(this_test, param,
                                                   size=np.count_nonzero(ind))
            response[ind] = test_response
            rt[ind] = test_rt
        return rt, response

    def gen(self, test, param, nrep=1):
        rts, resps = self.rvs(test, param)
        return pd.DataFrame({'test': test, 'rt': rts, 'response': resps.astype(int)})
