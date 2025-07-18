"""Linear ballistic accumulator model."""

import math
import numpy as np
import scipy.stats as st
import aesara
import aesara.tensor as aet
import pymc as pm

def normpdf(x):
    return (1 / pm.math.sqrt(2 * math.pi)) * pm.math.exp(-(x ** 2) / 2)

def normcdf(x):
    return (1 / 2) * (1 + pm.math.erf(x / pm.math.sqrt(2)))

def tpdf(t, A, b, v, sv):
    """Probability distribution function over time."""
    g = (b - A - t * v) / (t * sv)
    h = (b - t * v) / (t * sv)
    f = (-v * normcdf(g) + sv * normpdf(g) +
         v * normcdf(h) - sv * normpdf(h)) / A
    return f


def tcdf(t, A, b, v, s):
    """Cumulative distribution function over time."""
    e1 = ((b - A - t * v) / A) * normcdf((b - A - t * v) / (t * s))
    e2 = ((b - t * v) / A) * normcdf((b - t * v) / (t * s))
    e3 = ((t * s) / A) * normpdf((b - A - t * v) / (t * s))
    e4 = ((t * s) / A) * normpdf((b - t * v) / (t * s))
    F = 1 + e1 - e2 + e3 - e4
    return F


def ncdf(t, A, b, v, s):
    """Probability of no response from a set of accumulators."""
    ncdf_all, updates = aesara.reduce(
        fn=lambda v_i, tot, t, A, b, s: (1 - tcdf(t, A, b, v_i, s)) * tot,
        sequences=v, outputs_info=aet.ones_like(t),
        non_sequences=[t, A, b, s])
    return ncdf_all


def resp_pdf(t, ind, A, b, v, s):
    """Probability density function for response i at time t."""
    p_neg, updates = aesara.reduce(
        fn=lambda v_i, tot, s: normcdf(-v_i / s) * tot,
        sequences=v, outputs_info=aet.ones(1, dtype='float64'), non_sequences=s)

    # PDF for i and no finish yet for others
    v_ind = aet.arange(v.shape[0])
    i = aet.cast(ind, 'int64')
    res, updates = aesara.scan(
        fn=(lambda t_j, i_j, v_ind, A, b, v, s:
            (tpdf(t_j, A, b, v[i_j], s) *
             ncdf(t_j, A, b, v[aet.nonzero(aet.neq(v_ind, i_j))], s))),
        sequences=[t, i], non_sequences=[v_ind, A, b, v, s])

    # conditionalize on any response
    pdf = res / (1 - p_neg)

    # define probability of negative times to zero
    pdf_cond = aet.switch(aet.gt(t, 0), pdf, 0)
    return pdf_cond

class LBA():
    """Linear Ballistic Accumulator model."""
    def tensor_pdf(self, rt, response, test, param):
        tau = param['tau']
        sub_param = param.copy()
        del sub_param['tau']
        return resp_pdf(rt - tau, response, **sub_param)

    def function_pdf(self):
        # time and response vary by trial
        t = aet.dvector('t')
        i = aet.ivector('i')

        # parameters are fixed over trial
        A = aet.dscalar('A')
        b = aet.dscalar('b')
        v = aet.dvector('v')
        s = aet.dscalar('s')
        tau = aet.dscalar('tau')
        pdf = resp_pdf(t - tau, i, A, b, v, s)
        f = aesara.function([t, i, A, b, v, s, tau], pdf)
        return f

    def rvs_test(self, test, param, size):
        def sample_response(A, b, v, s, tau, size):
            """Sample response from a set of accumulators."""
            
            def sample_finish_time(A, b, v, s, tau, size):
                """Sample finish time for a set of accumulators."""
                # select starting point
                k = st.uniform.rvs(loc=0, scale=A, size=size)

                t = np.zeros((len(v), size))
                for i, vi in enumerate(v):
                    # sample drift rate, calculate time to threshold
                    d = st.norm.rvs(loc=vi, scale=s, size=size)
                    ti = tau + ((b - k) / d)

                    # time is invalid if drift rate is negative
                    ti[d < 0] = np.nan
                    t[i, :] = ti
                return t
            
            # get finish time for each accumulator
            t = sample_finish_time(A, b, v, s, tau, size)

            # determine winner on each valid trial
            valid = np.any(np.logical_not(np.isnan(t)), 0)
            t_valid = t[:, valid]
            t_winner = np.nanmin(t_valid, 0)
            i_winner = np.nanargmin(t_valid, 0)

            # initialize full matrix
            response = np.empty(size)
            response.fill(np.nan)
            rt = np.empty(size)
            rt.fill(np.nan)

            # fill in valid trials
            rt[valid] = t_winner
            response[valid] = i_winner
            return rt, response
        
        rt, resp = sample_response(**param, size=size)
        return rt, resp
    
    
    def rvs(self, test, param):
        """
        Generate responses for all test types.

        Parameters
        ----------
        test : numpy.ndarray
            Test trial type.

        param : dict of (str: float)
            Parameter values.

        Returns
        -------
        rt : numpy.ndarray
            Simulated response times.

        response : numpy.ndarray
            Simulated responses.
        """
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

    def gen(self, test, param, subj_idx=None, nrep=1, subj_param=None):
        """
        Generate a simulated dataset.

        Parameters
        ----------
        test : numpy.ndarray
            Test type of each trial.

        param : dict of (str: float)
            Parameter values.

        subj_idx : numpy.ndarray, optional
            Index of the subject to simulate for each trial.

        nrep : int, optional
            Number of replications to simulate.

        subj_param : list of (dict of (str: float)), optional
            Parameter values for each subject.

        Returns
        -------
        data : pandas.DataFrame
            Simulated data.
        """
        data_list = []
        for i in range(nrep):
            if subj_param is not None:
                rt, response = self.rvs_subj(test, subj_idx, param, subj_param)
            else:
                rt, response = self.rvs(test, param)
            rep = pd.DataFrame({'test': test, 'rt': rt,
                                'response': response.astype('int32')})
            if subj_idx is not None:
                rep.loc[:, 'subj_idx'] = subj_idx
            rep.loc[:, 'rep'] = i
            data_list.append(rep)
        data = pd.concat(data_list, ignore_index=True)
        return data

    def tensor_logp(self, param):
        """
        Function to evaluate the log PDF for a given response.

        Parameters
        ----------
        param : dict of (str: float)
            Parameter values.

        Returns
        -------
        logp : callable
            Function that takes rt, response, and test and returns log
            probability.
        """
        def logp(rt, response, test):
            p = self.tensor_pdf(rt, response, test, param)
            return log_prob(p)
        return logp
