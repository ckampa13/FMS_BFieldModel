#! /usr/bin/env python
"""Module for fitting magnetic field data with a parametric expression.

This is the main workhorse module for fitting the magnetic field data.  It is assumed that the data
has passed through the :class:`mu2e.dataframeprod.DataFrameMaker`, and thus in the expected format.
In most cases the data has also passed through the :mod:`mu2e.hallprober` module, to imitate the act
of surveying one of the Mu2E solenoids with a series of hall probes.  The
:class:`mu2e.fieldfitter.FieldFitter` preps the data, flattens it into 1D, and uses the :mod:`lmfit`
package extensively for parameter-handling, fitting, optimization, etc.

Example:
    Incomplete excerpt, see :func:`mu2e.hallprober.field_map_analysis` and `scripts/hallprobesim`
    for more typical use cases:

    .. code-block:: python

        # assuming config files already defined...

        In [10]: input_data = DataFileMaker(cfg_data.path, use_pickle=True).data_frame
        ...      input_data.query(' and '.join(cfg_data.conditions))

        In [11]: hpg = HallProbeGenerator(
        ...         input_data, z_steps = cfg_geom.z_steps,
        ...         r_steps = cfg_geom.r_steps, phi_steps = cfg_geom.phi_steps,
        ...         x_steps = cfg_geom.x_steps, y_steps = cfg_geom.y_steps)

        In [12]: ff = FieldFitter(hpg.get_toy())

        In [13]: ff.fit(cfg_params, cfg_pickle)
        ...      # This will take some time, especially for many data points and free params

        In [14]: ff.merge_data_fit_res() # merge the results in for easy plotting

        In [15]: make_fit_plots(ff.input_data, cfg_data, cfg_geom, cfg_plot, name)
        ...      # defined in :class:`mu2e.hallprober`

*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
from time import time
from copy import deepcopy
import numpy as np
import six.moves.cPickle as pkl
import pandas as pd
from lmfit import Model, Parameters, report_fit
from mu2e import mu2e_ext_path
from mu2e.tools import fit_funcs_redux as ff
import os 

class FitStatus:
    def __init__(self, n):
        self.every_N = n
    def __call__(self, params, i, resid, *args, **kws):
        if i % self.every_N == 1:
            print(f'Iter {i} sum sq resid {np.sum(resid**2)}')

class FieldFitter:
    """Input field measurements, perform parametric fit, return relevant quantities.

    The :class:`mu2e.fieldfitter.FieldFitter` takes a 3D set of field measurements and their
    associated position values, and performs a parametric fit.  The parameters and fit model are
    handled by the :mod:`lmfit` package, which in turn wraps the :mod:`scipy.optimize` module, which
    actually performs the parameter optimization.  The default optimizer is the Levenberg-Marquardt
    algorithm.

    The :func:`mu2e.fieldfitter.FieldFitter.fit` requires multiple cfg `namedtuples`, and performs
    the actual fitting (or recreates a fit for a given set of saved parameters).  After fitting, the
    generated class members can be used for further analysis.

    Args:
        input_data (pandas.DataFrame): DF that contains the field component values to be fit.
        cfg_geom (namedtuple): namedtuple with the following members:
            'geom z_steps r_steps phi_steps x_steps y_steps bad_calibration'

    Attributes:
        input_data (pandas.DataFrame): The input DF, with possible modifications.
        phi_steps (List[float]): The axial values of the field data (cylindrial coords)
        r_steps (List[float]): The radial values of the field data (cylindrial coords)
        x_steps (List[float]): The x values of the field data (cartesian coords)
        y_steps (List[float]): The y values of the field data (cartesian coords)
        pickle_path (str): Location to read/write the pickled fit parameter values
        params (lmfit.Parameters): Set of Parameters, inherited from `lmfit`
        result (lmfit.ModelResult): Container for resulting fit information, inherited from `lmfit`

    """
    def __init__(self, input_data, calc_data=None):
        self.input_data = input_data
        self.pickle_path = mu2e_ext_path+'fit_params/'
        if calc_data is not None:
            self.br_calc_data = calc_data['br']
            self.bphi_calc_data = calc_data['bphi']
            self.bz_calc_data = calc_data['bz']
        else:
            self.br_calc_data = None
            self.bphi_calc_data = None
            self.bz_calc_data = None

    def fit(self, cfg_params, cfg_pickle, iterative=False):
        if iterative:
            self.fit_iter(cfg_params, cfg_pickle)
        else:
            self.fit_simple(cfg_params, cfg_pickle)

    def fit_simple(self, cfg_params, cfg_pickle):
        """Helper function that chooses one of the subsequent fitting functions."""
        self.prep_fit_func(cfg_params, cfg_pickle)
        self.fit_solenoid(cfg_params, cfg_pickle)

    def fit_iter(self, cfg_params, cfg_pickle):
        """Helper function that chooses one of the subsequent fitting functions."""
        self.prep_fit_func(cfg_params, cfg_pickle)

        # self.fit_solenoid(cfg_params, cfg_pickle)

        # # check if any param uncertainties could not be estimated. if yes, fit again with them set to not vary
        # self.N_nan_stderr = self.set_nan_stderr_non_vary()
        # if self.N_nan_stderr > 0:
        #     self.fit_solenoid(cfg_params, cfg_pickle)
        # self.redchi_hist = []
        # self.i_fit = 0
        self.redchi_best = np.inf
        self.redchi_dec = True
        self.params_best = deepcopy(self.params)
        self.N_nan_stderr = 1
        # repeat fit up to 2 extra times
        #while (self.N_nan_stderr > 0) and (self.i_fit < 3):
        while (self.N_nan_stderr > 0) and (self.redchi_dec):
            self.fit_solenoid(cfg_params, cfg_pickle)
            redchi = self.result.redchi
            if redchi < self.redchi_best:
                self.redchi_dec = True
                self.redchi_best = redchi
                self.params_best = deepcopy(self.params)
                try:
                    self.correl_dict_best = deepcopy(self.correl_dict)
                except:
                    pass
            else:
                self.redchi_dec = False
                # CAUTION!
                # self.redchi_dec = True
            # self.N_nan_stderr, self.non_vary_list = self.set_nan_stderr_non_vary()
            self.N_nan_stderr = self.set_nan_stderr_non_vary()
            # self.i_fit += 1
        # if we start doing worse, go to best params for final fit
        if not self.redchi_dec:
            # self.revert_nan_stderr_vary(self.non_vary_list, self.params_best = deepcopy(self.params))
            # final fit
            self.params = deepcopy(self.params_best)
            # reset ?
            # for k in self.params.keys():
            #     if self.params[k].vary:
            #         self.params[k].value = 0.
            self.fit_solenoid(cfg_params, cfg_pickle)


    def prep_fit_func(self, cfg_params, cfg_pickle):
        print("Preparing fit func")
        # error if calc data not supplied and needed
        if (cfg_params.cfg_calc_data is not None) and (self.br_calc_data is None):
            raise ValueError("Supplied cfg_params expected calculated data, but none were supplied in FieldFitter constructor. Please note this is only supported for function versions: [1000,1004,1005]")
        func_version = cfg_params.version

        # Load pre-defined starting values for parameters, or start a new set
        if cfg_pickle.recreate:
            try:
                self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
                                            "rb"))
            except UnicodeDecodeError:
                self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
                                            "rb"), encoding='latin1')
        elif cfg_pickle.use_pickle:
            try:
                self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
                                            "rb"))
            except UnicodeDecodeError:
                self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
                                            "rb"), encoding='latin1')
            self.add_params_default(cfg_params)
        else:
            self.params = Parameters()
            self.add_params_default(cfg_params)

        self.ZZ = self.input_data.Z.values
        self.RR = self.input_data.R.values
        self.PP = self.input_data.Phi.values
        self.Bz = self.input_data.Bz.values
        self.Br = self.input_data.Br.values
        self.Bphi = self.input_data.Bphi.values
        self.XX = self.input_data.X.values
        self.YY = self.input_data.Y.values

        # Choose the type of fitting function we'll be using.
        pvd = self.params.valuesdict()  # Quicker way to grab params and init the fit functions

        if func_version == 1000:
            self.fit_func = ff.brzphi_3d_producer_giant_function(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'],
                self.bz_calc_data, self.br_calc_data, self.bphi_calc_data)
        elif func_version == 1001:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1001(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'])
        elif func_version == 1002:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1002(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'])
        elif func_version == 1004:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1004(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'])
        elif func_version == 1005:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1005(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'],
                pvd['z0'],
                self.bz_calc_data, self.br_calc_data, self.bphi_calc_data)
        elif func_version == 1006:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1006(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'],
                self.bz_calc_data, self.br_calc_data, self.bphi_calc_data)
        elif func_version == 1007:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1007(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'],
                self.bz_calc_data, self.br_calc_data, self.bphi_calc_data)
        elif func_version == 1008:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1008(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'],
                pvd['z0'],
                self.bz_calc_data, self.br_calc_data, self.bphi_calc_data)
        elif func_version == 1009:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1009(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'],
                self.bz_calc_data, self.br_calc_data, self.bphi_calc_data)
        elif func_version == 1010:
            self.fit_func = ff.brzphi_3d_producer_giant_function_v1010(
                self.ZZ, self.RR, self.PP,
                pvd['pitch1'], pvd['ms_h1'], pvd['ns_h1'],
                pvd['pitch2'], pvd['ms_h2'], pvd['ns_h2'],
                pvd['length1'], pvd['ms_c1'], pvd['ns_c1'],
                pvd['length2'], pvd['ms_c2'], pvd['ns_c2'],
                pvd['z0'],
                n_list_c1=cfg_params.n_list_c1, n_list_k0=cfg_params.n_list_k0,
                R_ref=cfg_params.R_ref,
                bz_input=self.bz_calc_data, br_input=self.br_calc_data,
                bphi_input=self.bphi_calc_data)
        else:
            raise NotImplementedError(f'Function version={func_version} not implemented.')

        # Start loading in additional parameters based on the function version.
        # This is coded POORLY.  THIS SHOULD BE HANDLED IN `hallprobesim`, not here!
        if func_version == 1000:
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl(1)
            self.add_params_cyl(2)
            self.add_params_cart_simple(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
        elif func_version == 1001:
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1001(1)
            self.add_params_cyl(2)
            self.add_params_cart_simple(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
        elif func_version == 1002:
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1002(1)
            self.add_params_cyl(2)
            self.add_params_cart_simple(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
        elif func_version == 1004:
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1004(1)
            self.add_params_cyl(2)
            self.add_params_cart_simple(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
        elif func_version == 1005:
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1005(1)
            self.add_params_cyl(2)
            self.add_params_cart_simple(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
        # z0 offset, introduction of new k scheme (can control whether each is varied)
        #elif func_version == 1006:
        elif (func_version == 1006):
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1006(1, AB_lim=cfg_params.AB_lim)
            self.add_params_cyl(2)
            self.add_params_cart_simple_fixable(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
            #z0
            if not cfg_params.z0 is None:
                self.params.add('z0', value=cfg_params.z0, vary=False)
            else:
                self.params.add('z0', value=0.0, vary=False)
        elif (func_version == 1007):
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1007(1, AB_lim=cfg_params.AB_lim)
            self.add_params_cyl(2)
            self.add_params_cart_simple_fixable(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
            #z0
            if not cfg_params.z0 is None:
                self.params.add('z0', value=cfg_params.z0, vary=False)
            else:
                self.params.add('z0', value=0.0, vary=False)
        elif (func_version == 1008):
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1008(1, AB_lim=cfg_params.AB_lim)
            self.add_params_cyl(2)
            self.add_params_cart_simple_fixable(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
            #z0
            if not cfg_params.z0 is None:
                self.params.add('z0', value=cfg_params.z0, vary=False)
            else:
                self.params.add('z0', value=0.0, vary=False)
        elif (func_version == 1009):
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1009(1, AB_lim=cfg_params.AB_lim)
            self.add_params_cyl(2)
            self.add_params_cart_simple_fixable(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
            #z0
            if not cfg_params.z0 is None:
                self.params.add('z0', value=cfg_params.z0, vary=False)
            else:
                self.params.add('z0', value=0.0, vary=False)
        # v1008 + k=0 multipole terms, optional Bessel order subset
        elif (func_version == 1010):
            self.add_params_hel(1)
            self.add_params_hel(2)
            self.add_params_cyl_v1010(1, n_list=cfg_params.n_list_c1, AB_lim=cfg_params.AB_lim)
            self.add_params_cyl(2)
            self.add_params_k0_v1010(cfg_params.n_list_k0, R_ref=cfg_params.R_ref, AB_lim=cfg_params.AB_lim)
            self.add_params_cart_simple_fixable(cfg_params)
            self.add_params_biot_savart(cfg_params, cfg_pickle.recreate)
            #z0
            if 'z0' not in self.params:
                if not cfg_params.z0 is None:
                    self.params.add('z0', value=cfg_params.z0, vary=False)
                else:
                    self.params.add('z0', value=0.0, vary=False)
            if not cfg_pickle.recreate:
                self.check_degeneracy_v1010(cfg_params.n_list_k0)

    def model(self):
        return Model(self.fit_func, independent_vars=['r', 'z', 'phi', 'x', 'y'])

    def fit_solenoid(self, cfg_params, cfg_pickle):
        """Main fitting function for FieldFitter class.

        The typical magnetic field geometry for the Mu2E experiment is determined by one or more
        solenoids, with some contaminating external fields.  The purpose of this function is to fit
        a set of sparse magnetic field data that would, in practice, be generated by a field
        measurement device.

        The following assumptions must hold for the input data:
           * The data is represented in a cylindrical coordiante system.
           * The data forms a series of planes, where all planes intersect at R=0.
           * All planes has the same R and Z values.
           * All positive Phi values have an associated negative phi value, which uniquely defines a
             single plane in R-Z space.

        Args:
           cfg_params (namedtuple): 'ns ms cns cms Reff func_version'
           cfg_pickle (namedtuple): 'use_pickle save_pickle load_name save_name recreate'

        Returns:
            Nothing.  Generates class attributes after fitting, and saves parameter values, if
            saving is specified.
        """
        print("Running fit")
        func_version = cfg_params.version
        # Generate an lmfit Model
        self.mod = Model(self.fit_func, independent_vars=['r', 'z', 'phi', 'x', 'y'])

        if not cfg_pickle.recreate:
            print(f'fitting with func_version={func_version},')
            print(cfg_params)

        else:
            print(f'recreating fit with func_version={func_version},')
            print(cfg_params)
        start_time = time()

        if cfg_params.noise != None:
            print(f'Applying weights for noise {cfg_params.noise}')
            noise_err  = 1/np.full_like(self.Bz,cfg_params.noise)
            weights = np.concatenate(3*[noise_err]).ravel()
        else:
            print('Using default weights (all = 1)')
            weights = None

        if cfg_pickle.recreate:
            for param in self.params:
                self.params[param].vary = False
            self.result = self.mod.fit(np.concatenate([self.Br, self.Bz, self.Bphi]).ravel(),
                                       weights=weights, scale_covar=False,
                                       r=self.RR, z=self.ZZ, phi=self.PP, x=self.XX, y=self.YY, params=self.params,
                                       method='leastsq', fit_kws={'maxfev': 1})
        else:
            print_status = FitStatus(1000)
            # mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
            #if cfg_params.method == 'leastsq' or cfg_params.method == 'brute':
            if cfg_params.method not in ('least_squares', 'linear'):
                self.result = self.mod.fit(np.concatenate([self.Br, self.Bz, self.Bphi]).ravel(),
                                           weights=weights, scale_covar=False,
                                           r=self.RR, z=self.ZZ, phi=self.PP, x=self.XX, y=self.YY, params=self.params,
                                           iter_cb=print_status,
                                           method=cfg_params.method) #max_nfev if wanting to limit function calls
            elif cfg_params.method == 'least_squares':
                self.result = self.mod.fit(np.concatenate([self.Br, self.Bz, self.Bphi]).ravel(),
                                           weights=weights, scale_covar=False,
                                           r=self.RR, z=self.ZZ, phi=self.PP, x=self.XX, y=self.YY, params=self.params,
                                           method='least_squares', iter_cb=print_status, fit_kws={'verbose': 1,
                                                                                                  # original
                                                                                                  'gtol': 1e-8,
                                                                                                  'ftol': 1e-8,
                                                                                                  #'xtol': None,
                                                                                                  # testing
                                                                                                  # 'gtol': 1e-6,
                                                                                                  # 'ftol': 1e-6,
                                                                                                  # 'xtol': 1e-6,
                                                                                                  'xtol': 1e-8,
                                                                                                  'loss': cfg_params.loss,
                                                                                                  'jac': '2-point', # default, faster
                                                                                                  #'jac': '3-point', # more accurate, slower
                                                                                                  #'x_scale': 'jac', # default is None -- how to scale each free param
                                                                                                  #'diff_step': 1e-6*np.ones_like(self.params.keys()), # default is some optimal value based on machine epsilon
                                           })
            elif cfg_params.method == 'linear':
                self.result = self.fit_linear(weights, print_status)
            else:
                print('Error, only supported methods are leastsq and least_squares')
                exit()

        if not cfg_pickle.recreate:
            self.params = self.result.params
        end_time = time()
        print(("Elapsed time was %g seconds" % (end_time - start_time)))
        report_fit(self.result, show_correl=False)
        if cfg_pickle.save_pickle:
            self.pickle_results(self.pickle_path+cfg_pickle.save_name)
            # can only save the correlations when we aren't recreating
            if not cfg_pickle.recreate:
                self.pickle_correl(self.pickle_path+cfg_pickle.save_name)

    def fit_external(self, cfg_params, cfg_pickle, profile=False):
        raise NotImplementedError('Oh no! you got lazy during refactoring')

    def fit_linear(self, weights, print_status=None):
        '''Direct weighted linear least squares (every free parameter must enter linearly),
        followed by an lmfit leastsq polish from the solution so self.result (covar, stderr,
        correl, eval_uncertainty) is a standard ModelResult.'''
        from scipy import linalg
        t0 = time()
        data = np.concatenate([self.Br, self.Bz, self.Bphi]).ravel()
        kw = dict(r=self.RR, z=self.ZZ, phi=self.PP, x=self.XX, y=self.YY)
        free = [p for p in self.params if self.params[p].vary]
        pv = self.params.valuesdict()
        p0 = {k: v for k, v in pv.items()}
        for k in free:
            p0[k] = 0.
        offset = self.fit_func(**kw, **p0)
        J = None
        if hasattr(self.fit_func, 'design_matrix'):
            try:
                J = self.fit_func.design_matrix(free, **kw)
                print(f'fit_linear: fast design matrix, shape {J.shape}')
            except NotImplementedError as e:
                print(f'fit_linear: {e} Using unit-step differences.')
                J = None
        if J is None:
            # exact for a model linear in the free parameters
            J = np.empty((len(offset), len(free)))
            for j, k in enumerate(free):
                p0[k] = 1.
                J[:, j] = self.fit_func(**kw, **p0) - offset
                p0[k] = 0.
            print(f'fit_linear: design matrix by unit steps, shape {J.shape}')
        if weights is None:
            w = np.ones_like(data)
        else:
            w = weights
        A = J * w[:, np.newaxis]
        b = (data - offset) * w
        del J
        col = np.linalg.norm(A, axis=0)
        col[col == 0] = 1.
        A /= col[np.newaxis, :]
        sol, _, rank, sv = linalg.lstsq(A, b, lapack_driver='gelsd', check_finite=False)
        x = sol / col
        cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
        chi2 = np.sum((A @ sol - b)**2)
        print(f'fit_linear: P={len(free)}, rank={rank}, cond(column-normalised)={cond:0.3e}, '
              +f'chi2={chi2:0.6e}, redchi={chi2/(len(b)-len(free)):0.6e}, time={time()-t0:0.1f} s')
        del A
        for k, v in zip(free, x):
            self.params[k].value = v
        # polish (should converge in ~1 Jacobian) to get a standard lmfit result
        print('fit_linear: lmfit leastsq polish from the linear solution...')
        result = self.mod.fit(data, weights=weights, scale_covar=False, params=self.params,
                              iter_cb=print_status, method='leastsq', **kw)
        print(f'fit_linear: polish redchi={result.redchi:0.6e}, nfev={result.nfev}')
        return result

    def pickle_results(self, pickle_name='default'):
        """Pickle the resulting Parameters after a fit is performed."""

        pkl.dump(self.result.params, open(pickle_name+'_results.p', "wb"), pkl.HIGHEST_PROTOCOL)

    def pickle_correl(self, pickle_name='default'):
        """Pickle the resulting Parameters after a fit is performed."""
        # compute correlation matrix, if possible
        try:
            # v = np.sqrt(np.diag(self.result.covar))
            # #v[np.isnan(v)] = 1e6
            # outer_v = np.outer(v, v)
            # self.correlation = self.result.covar / outer_v
            # self.correlation[self.result.covar == 0] = 0
            vn = self.result.var_names
            N = len(vn)
            self.correl = np.identity(N)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        self.correl[i, j] = self.result.params[vn[i]].correl[vn[j]]

            self.correl_dict = {'variables': vn,
                               'covar': self.result.covar,
                               'correl': self.correl}

            pkl.dump(self.correl_dict, open(pickle_name+'_results_correl.p', "wb"), pkl.HIGHEST_PROTOCOL)
        except:
            print('Correlation could not be retrieved from the fit.')
            # Cleanup -- don't want mismatching results and results_correl, in case fit is overwriting a previous verison
            try:
                os.remove(pickle_name+'_results_correl.p')
            except OSError:
                pass

    def merge_data_fit_res(self, saveunc, iscart):
        """Combine the fit results and the input data into one dataframe for easier
        comparison of results.
        """
        print('In FieldFitter.merge_data_fit_res...')
        bf = self.result.best_fit

        self.input_data.loc[:, 'Br_fit'] = bf[0:len(bf)//3]
        self.input_data.loc[:, 'Bz_fit'] = bf[len(bf)//3:2*len(bf)//3]
        self.input_data.loc[:, 'Bphi_fit'] = bf[2*len(bf)//3:]

        # If cartesian geometry, also evaluate Bx_fit / By_fit
        if iscart:
            self.input_data.eval('Bx_fit = Br_fit*cos(Phi)-Bphi_fit*sin(Phi)', inplace=True)
            self.input_data.eval('By_fit = Br_fit*sin(Phi)+Bphi_fit*cos(Phi)', inplace=True)
        
        # Only save uncertainty for true nominal fit
        if saveunc :
            print('Evaluating uncertainties...')
            unc = self.result.eval_uncertainty(self.params)
            self.input_data.loc[:, 'Br_unc'] = unc[0:len(unc)//3]
            self.input_data.loc[:, 'Bz_unc'] = unc[len(unc)//3:2*len(unc)//3]
            self.input_data.loc[:, 'Bphi_unc'] = unc[2*len(unc)//3:]
        print('FieldFitter.merge_data_fit_res done.')

    def set_nan_stderr_non_vary(self):
        # Note this function should only be run after a fit. Before the fit,
        # params will not be seeded and will not contain 'stderr'.
        # non_vary_list = []
        vn = self.result.var_names
        N = 0
        for v in vn:
            if (np.isnan(self.params[v].stderr)):
            # nan stderr OR value too close to zero.
            # if (np.isnan(self.params[v].stderr)) or (abs(self.params[v].value) < 1e-6):
                self.params[v].value = 0.
                self.params[v].vary = False
                N += 1
                #non_vary_list.append(v)
        return N
        #return N, non_vary_list

    # def revert_nan_stderr_vary(self, non_vary_list):
    #     for v in non_vary_list:
    #         self.params[v].vary = True

    def add_params_default(self, cfg_params):
        if 'pitch1' not in self.params:
            self.params.add('pitch1', value=cfg_params.pitch1, vary=False)
        else:
            self.params['pitch1'].value = cfg_params.pitch1
        if 'ms_h1' not in self.params:
            self.params.add('ms_h1', value=cfg_params.ms_h1, vary=False)
        else:
            self.params['ms_h1'].value = cfg_params.ms_h1
        if 'ns_h1' not in self.params:
            self.params.add('ns_h1', value=cfg_params.ns_h1, vary=False)
        else:
            self.params['ns_h1'].value = cfg_params.ns_h1
        if 'pitch2' not in self.params:
            self.params.add('pitch2', value=cfg_params.pitch2, vary=False)
        else:
            self.params['pitch2'].value = cfg_params.pitch2
        if 'ms_h2' not in self.params:
            self.params.add('ms_h2', value=cfg_params.ms_h2, vary=False)
        else:
            self.params['ms_h2'].value = cfg_params.ms_h2
        if 'ns_h2' not in self.params:
            self.params.add('ns_h2', value=cfg_params.ns_h2, vary=False)
        else:
            self.params['ns_h2'].value = cfg_params.ns_h2
        if 'length1' not in self.params:
            self.params.add('length1', value=cfg_params.length1, vary=False)
        else:
            self.params['length1'].value = cfg_params.length1
        if 'ms_c1' not in self.params:
            self.params.add('ms_c1', value=cfg_params.ms_c1, vary=False)
        else:
            self.params['ms_c1'].value = cfg_params.ms_c1
        if 'ns_c1' not in self.params:
            self.params.add('ns_c1', value=cfg_params.ns_c1, vary=False)
        else:
            self.params['ns_c1'].value = cfg_params.ns_c1
        if 'length2' not in self.params:
            self.params.add('length2', value=cfg_params.length2, vary=False)
        else:
            self.params['length2'].value = cfg_params.length2
        if 'ms_c2' not in self.params:
            self.params.add('ms_c2', value=cfg_params.ms_c2, vary=False)
        else:
            self.params['ms_c2'].value = cfg_params.ms_c2
        if 'ns_c2' not in self.params:
            self.params.add('ns_c2', value=cfg_params.ns_c2, vary=False)
        else:
            self.params['ns_c2'].value = cfg_params.ns_c2
        # max asymmetric terms
        if 'ms_asym_max' not in self.params:
            self.params.add('ms_asym_max', value=cfg_params.ms_asym_max, vary=False)
        else:
            self.params['ms_asym_max'].value = cfg_params.ms_asym_max
        # z0
        if 'z0' not in self.params:
            if cfg_params.z0 is None:
                z0_ = 0.
            else:
                z0_ = cfg_params.z0
            self.params.add('z0', value=z0_, vary=False)
        else:
            if cfg_params.z0 is None:
                z0_ = 0.
            else:
                z0_ = cfg_params.z0
            self.params['z0'].value = z0_

    def add_params_hel(self, num):
        ms_range = range(self.params[f'ms_h{num}'].value)
        ns_range = range(self.params[f'ns_h{num}'].value)

        for m in ms_range:
            for n in ns_range:
                if num == 1:
                    if f'Ah{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ah{num}_{m}_{n}', value=0, vary=False)
                    else:
                        self.params[f'Ah{num}_{m}_{n}'].vary = False

                    if f'Bh{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Bh{num}_{m}_{n}', value=0, vary=False)
                    else:
                        self.params[f'Bh{num}_{m}_{n}'].vary = False

                    if f'Ch{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ch{num}_{m}_{n}', value=-1e-6, vary=True)
                    else:
                        self.params[f'Ch{num}_{m}_{n}'].vary = False

                    if f'Dh{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Dh{num}_{m}_{n}', value=-1e-6, vary=True)
                    else:
                        self.params[f'Dh{num}_{m}_{n}'].vary = False
                else:
                    if f'Ah{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ah{num}_{m}_{n}', value=-1e-6, vary=True)
                    else:
                        self.params[f'Ah{num}_{m}_{n}'].vary = False

                    if f'Bh{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Bh{num}_{m}_{n}', value=1e-6, vary=True)
                    else:
                        self.params[f'Bh{num}_{m}_{n}'].vary = False

                    if f'Ch{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Ch{num}_{m}_{n}', value=0, vary=False)
                    else:
                        self.params[f'Ch{num}_{m}_{n}'].vary = False

                    if f'Dh{num}_{m}_{n}' not in self.params:
                        self.params.add(f'Dh{num}_{m}_{n}', value=0, vary=False)
                    else:
                        self.params[f'Dh{num}_{m}_{n}'].vary = False

    def add_params_cyl_v1009(self, num, AB_lim=None):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        #np.random.seed(0)
        m_max = self.params['ms_asym_max'].value
        if m_max < 0:
            m_max = np.inf

        # limits on A/B?
        if AB_lim is None:
            AB_min = None
            AB_max = None
        else:
            AB_min = -AB_lim
            AB_max = AB_lim

        # cos like terms
        '''
        cosmax = 1.
        cosmin = -1.

        for m in ms_range:
            for n in ns_range:
                #if (m > m_max) & (n > 0):
                if (m > m_max) & (n > 1): # good for: m=50, n=5, m_asym_max = 10
                # if (m > m_max) & (n > 2): # can't find a good result
                    var = False
                    val0 = 0.
                    val1 = 0.
                    val2 = 0.
                else:
                    var = True
                    val0 = 10.+ m * (-1.)**m / m_max
                    val1 = 0.
                    val2 = 10.+ m * (-1.)**(m+1)/ m_max
                # N (normalization) and cos_alpha_m_n, cos_beta_m_n to control phase
                # or: N, M normalizations, cos_alpha_m_n phase
                if f'N_{m}_{n}' not in self.params:
                    #self.params.add(f'N_{m}_{n}', value=val0, vary=var, min=AB_min, max=AB_max)
                    self.params.add(f'N_{m}_{n}', value=val0, vary=var, min=AB_min, max=AB_max)
                else:
                    self.params[f'N_{m}_{n}'].vary = var
                # M normalization
                # if f'M_{m}_{n}' not in self.params:
                #     self.params.add(f'M_{m}_{n}', value=val2, vary=var, min=AB_min, max=AB_max)
                # else:
                #     self.params[f'M_{m}_{n}'].vary = var
                if f'cos_alpha_{m}_{n}' not in self.params:
                    self.params.add(f'cos_alpha_{m}_{n}', value=val1, vary=var, min=cosmin, max=cosmax)
                else:
                    self.params[f'cos_alpha_{m}_{n}'].vary = var
                if f'cos_beta_{m}_{n}' not in self.params:
                   self.params.add(f'cos_beta_{m}_{n}', value=val1, vary=var, min=cosmin, max=cosmax)
                else:
                   self.params[f'cos_beta_{m}_{n}'].vary = var
                # n = 0, fixed C and D -- can achieve by fixing cos_beta_m_n to 1.
                if n == 0:
                   self.params[f'cos_beta_{m}_{n}'].value = 1.0
                   self.params[f'cos_beta_{m}_{n}'].vary = False
                # if n == 0:
                #    self.params[f'M_{m}_{n}'].value = 0.0
                #    self.params[f'M_{m}_{n}'].vary = False
                #    self.params[f'cos_beta_{m}_{n}'].value = 1.0
                #    self.params[f'cos_beta_{m}_{n}'].vary = False
        '''
        # single normalization, 3 + 1 constrained
        max_bcd = 3.**(-1/2)
        cosmax = 1. * max_bcd
        cosmin = -1. * max_bcd

        for m in ms_range:
            for n in ns_range:
                #if (m > m_max) & (n > 0):
                if (m > m_max) & (n > 1): # good for: m=50, n=5, m_asym_max = 10
                # if (m > m_max) & (n > 2): # can't find a good result
                    var = False
                    val0 = 0.
                    val1 = 0.
                    val2 = 0.
                else:
                    var = True
                    val0 = 10.+ m * (-1.)**m / m_max
                    val1 = 0.5 # equal contributions from each, to start
                    val2 = 0.5
                # N (normalization) and cos_alpha_m_n, cos_beta_m_n to control phase
                # or: N, M normalizations, cos_alpha_m_n phase
                if f'N_{m}_{n}' not in self.params:
                    #self.params.add(f'N_{m}_{n}', value=val0, vary=var, min=AB_min, max=AB_max)
                    self.params.add(f'N_{m}_{n}', value=val0, vary=var, min=0., max=AB_max)
                else:
                    self.params[f'N_{m}_{n}'].vary = var
                if f'b_{m}_{n}' not in self.params:
                    self.params.add(f'b_{m}_{n}', value=val1, vary=var, min=cosmin, max=cosmax)
                else:
                    self.params[f'b_{m}_{n}'].vary = var
                if f'c_{m}_{n}' not in self.params:
                    self.params.add(f'c_{m}_{n}', value=val1, vary=var, min=cosmin, max=cosmax)
                else:
                    self.params[f'c_{m}_{n}'].vary = var
                if f'd_{m}_{n}' not in self.params:
                    self.params.add(f'd_{m}_{n}', value=val1, vary=var, min=cosmin, max=cosmax)
                else:
                    self.params[f'd_{m}_{n}'].vary = var
                # sign of A parameter
                if f'lam_a_{m}_{n}' not in self.params:
                    self.params.add(f'lam_a_{m}_{n}', value=val2, vary=var, min=-2, max=2)
                else:
                    self.params[f'lam_a_{m}_{n}'].vary = var
                # n = 0, fixed C and D -- can achieve by fixing c=d=0.
                if n == 0:
                   self.params[f'c_{m}_{n}'].value = 0.0
                   self.params[f'c_{m}_{n}'].vary = False
                   self.params[f'd_{m}_{n}'].value = 0.0
                   self.params[f'd_{m}_{n}'].vary = False
                # if n == 0:
                #    self.params[f'M_{m}_{n}'].value = 0.0
                #    self.params[f'M_{m}_{n}'].vary = False
                #    self.params[f'cos_beta_{m}_{n}'].value = 1.0
                #    self.params[f'cos_beta_{m}_{n}'].vary = False


    def add_params_cyl_v1010(self, num, n_list=None, AB_lim=None):
        # same as v1008, but only for the Bessel orders in n_list (default: range(ns_c{num}))
        ms_range = range(self.params[f'ms_c{num}'].value)
        if n_list is None:
            n_list = range(self.params[f'ns_c{num}'].value)
        n_list = [int(n) for n in n_list]
        m_max = self.params['ms_asym_max'].value
        # avoid division by zero for ms = 1
        m_f = max(len(ms_range) - 1, 1)
        if m_max < 0:
            m_max = np.inf

        # limits on A/B?
        if AB_lim is None:
            AB_min = None
            AB_max = None
        else:
            AB_min = -AB_lim
            AB_max = AB_lim

        for m in ms_range:
            for n in n_list:
                if (m > m_max) & (n > 0):
                    var = False
                    val0 = 0.
                    val1 = 0.
                else:
                    var = True
                    val0 = 100 * (-1. + m/m_f)*(-1.)**m
                    val1 = -100 * (-1. + m/m_f)*(-1.)**m
                # A, B, C, D are linear
                for P, val in zip(['A', 'B', 'C', 'D'], [val0, val1, val0, val1]):
                    if f'{P}c{num}_{m}_{n}' not in self.params:
                        self.params.add(f'{P}c{num}_{m}_{n}', value=val, vary=var, min=AB_min, max=AB_max)
                    else:
                        self.params[f'{P}c{num}_{m}_{n}'].vary = var
                # n = 0, fixed for C and D
                if n == 0:
                    self.params[f'Cc{num}_{m}_{n}'].value = 0.0
                    self.params[f'Dc{num}_{m}_{n}'].value = 0.0
                    self.params[f'Cc{num}_{m}_{n}'].vary = False
                    self.params[f'Dc{num}_{m}_{n}'].vary = False

    def add_params_k0_v1010(self, n_list_k0, R_ref=None, AB_lim=None):
        # k=0 (z-independent) 2D multipoles: Ek0_n (skew, A_n) and Fk0_n (normal, B_n) at R_ref
        if n_list_k0 is None:
            n_list_k0 = []
        if R_ref is None:
            R_ref = 0.05
        if 'R_ref' not in self.params:
            self.params.add('R_ref', value=R_ref, vary=False)
        else:
            self.params['R_ref'].value = R_ref
        if AB_lim is None:
            AB_min = None
            AB_max = None
        else:
            AB_min = -AB_lim
            AB_max = AB_lim
        for n in n_list_k0:
            for P in ['E', 'F']:
                if f'{P}k0_{int(n)}' not in self.params:
                    self.params.add(f'{P}k0_{int(n)}', value=0., vary=True, min=AB_min, max=AB_max)
                else:
                    self.params[f'{P}k0_{int(n)}'].vary = True

    def check_degeneracy_v1010(self, n_list_k0):
        # k=0 terms that duplicate a cartesian k term exactly
        if n_list_k0 is None:
            return
        pairs = [('Ek0_1', 'k1', 'uniform Bx'), ('Fk0_1', 'k2', 'uniform By'),
                 ('Fk0_2', 'k4', 'normal quad (Phi = k4 x y)')]
        bad = []
        for p_k0, p_k, desc in pairs:
            if (p_k0 in self.params) and (p_k in self.params):
                if self.params[p_k0].vary and self.params[p_k].vary:
                    bad.append(f'{p_k0} <-> {p_k} ({desc})')
        if len(bad) > 0:
            raise ValueError('v1010: degenerate free parameters: ' + '; '.join(bad)
                             + '. Fix the cartesian k terms (vary=False in ks_dict).')

    def add_params_cyl_v1008(self, num, AB_lim=None):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        #np.random.seed(0)
        m_max = self.params['ms_asym_max'].value
        if len(ms_range) > 0:
            m_f = ms_range[-1]
        if m_max < 0:
            m_max = np.inf

        # limits on A/B?
        if AB_lim is None:
            AB_min = None
            AB_max = None
        else:
            AB_min = -AB_lim
            AB_max = AB_lim

        for m in ms_range:
            for n in ns_range:
                if (m > m_max) & (n > 0):
                # if (m > m_max) & (n > 1): # good for: m=50, n=5, m_asym_max = 10
                # if (m > m_max) & (n > 2): # can't find a good result
                    var = False
                    val0 = 0.
                    val1 = 0.
                else:
                    var = True
                    #val0 = 10. + m*(-1.)**m / m_f
                    #val1 = 10. + m*(-1.)**(m+1) / m_f
                    val0 = 100 * (-1. + m/m_f)*(-1.)**m
                    val1 = -100 * (-1. + m/m_f)*(-1.)**m
                    #print(f'm={m}, n={n}: val0={val0}, val1={val1}')
                    # val0 = 0.
                    # val1 = 0.
                # A, B, C, D are linear
                if f'Ac{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Ac{num}_{m}_{n}', value=val0, vary=var, min=AB_min, max=AB_max)
                else:
                    self.params[f'Ac{num}_{m}_{n}'].vary = var
                if f'Bc{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Bc{num}_{m}_{n}', value=val1, vary=var, min=AB_min, max=AB_max)
                else:
                    self.params[f'Bc{num}_{m}_{n}'].vary = var
                if f'Cc{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Cc{num}_{m}_{n}', value=val0, vary=var, min=AB_min, max=AB_max)
                else:
                    self.params[f'Cc{num}_{m}_{n}'].vary = var
                if f'Dc{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Dc{num}_{m}_{n}', value=val1, vary=var, min=AB_min, max=AB_max)
                else:
                    self.params[f'Dc{num}_{m}_{n}'].vary = var
                # n = 0, fixed for C and D
                if n == 0:
                    self.params[f'Cc{num}_{m}_{n}'].value = 0.0
                    self.params[f'Dc{num}_{m}_{n}'].value = 0.0
                    self.params[f'Cc{num}_{m}_{n}'].vary = False
                    self.params[f'Dc{num}_{m}_{n}'].vary = False

    def add_params_cyl_v1007(self, num, AB_lim=None):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        #np.random.seed(0)
        m_max = self.params['ms_asym_max'].value
        if m_max < 0:
            m_max = np.inf

        # limits on A/B?
        if AB_lim is None:
            AB_min = None
            AB_max = None
        else:
            AB_min = -AB_lim
            AB_max = AB_lim

        for m in ms_range:
            for n in ns_range:
                if (m > m_max) & (n > 0):
                    var = False
                    phase0 = np.pi/2. # constant cos
                else:
                    var = True
                    phase0 = np.pi/4.
                # A and B are linear
                if f'Ac{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Ac{num}_{m}_{n}', value=0.0, vary=var, min=AB_min, max=AB_max)
                    # self.params.add(f'Ac{num}_{m}_{n}', value=10.0, vary=var, min=AB_min, max=AB_max)
                else:
                    self.params[f'Ac{num}_{m}_{n}'].vary = var
                # if f'Bc{num}' not in self.params:
                #if f'Bc{num}_{m}' not in self.params:
                if f'Bc{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Bc{num}_{m}_{n}', value=phase0, vary=var, min=0.0, max=2.*np.pi)
                    # self.params.add(f'Bc{num}_{m}_{n}', value=phase0, vary=var, min=-np.pi/8., max=2.*np.pi)
                    # self.params.add(f'Bc{num}_{m}', value=phase0, vary=var, min=-np.pi/8., max=2.*np.pi)
                    # self.params.add(f'Bc{num}', value=phase0, vary=var, min=-np.pi/8., max=2.*np.pi)
                else:
                    self.params[f'Bc{num}_{m}_{n}'].vary = var
                    #self.params[f'Bc{num}_{m}'].vary = var
                    # self.params[f'Bc{num}'].vary = True
                # D is the phi phase
                # if f'Dc{num}' not in self.params:
                if f'Dc{num}_{n}' not in self.params:
                #if f'Dc{num}_{m}_{n}' not in self.params:
                    if n > 0:
                        #self.params.add(f'Dc{num}_{m}_{n}', value=phase0, min=0, max=np.pi, vary=var)
                        #self.params.add(f'Dc{num}_{m}_{n}', value=phase0, min=-np.pi/8., max=np.pi, vary=var)
                        self.params.add(f'Dc{num}_{n}', value=phase0, min=0.0, max=np.pi, vary=var)
                        # self.params.add(f'Dc{num}_{n}', value=phase0, min=-np.pi/8., max=np.pi, vary=var)
                        # self.params.add(f'Dc{num}', value=phase0, min=-np.pi/8., max=np.pi, vary=var)
                    # n=0 is constant term, no phase
                    else:
                        #self.params.add(f'Dc{num}_{m}_{n}', value=np.pi/2, vary=False)
                        self.params.add(f'Dc{num}_{n}', value=np.pi/2, vary=False)
                        # self.params.add(f'Dc{num}', value=phase0, min=-np.pi/8., max=np.pi, vary=var)
                elif n > 0:
                    #self.params[f'Dc{num}_{m}_{n}'].min = 0.0
                    #self.params[f'Dc{num}_{m}_{n}'].min = -np.pi/8.
                    #self.params[f'Dc{num}_{m}_{n}'].max = np.pi
                    # self.params[f'Dc{num}_{n}'].min = -np.pi/8.
                    self.params[f'Dc{num}_{n}'].min = 0.0
                    self.params[f'Dc{num}_{n}'].max = np.pi
                    # self.params[f'Dc{num}'].min = -np.pi/8.
                    # self.params[f'Dc{num}'].max = np.pi

    def add_params_cyl_v1006(self, num, AB_lim=None):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        #np.random.seed(0)
        m_max = self.params['ms_asym_max'].value
        if m_max < 0:
            m_max = np.inf

        # limits on A/B?
        if AB_lim is None:
            AB_min = None
            AB_max = None
        else:
            AB_min = -AB_lim
            AB_max = AB_lim

        for m in ms_range:
            for n in ns_range:
                if (m > m_max) & (n > 0):
                    var = False
                else:
                    var = True
                # A and B are linear
                if f'Ac{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Ac{num}_{m}_{n}', value=0.0, vary=var, min=AB_min, max=AB_max)
                else:
                    self.params[f'Ac{num}_{m}_{n}'].vary = var
                if f'Bc{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Bc{num}_{m}_{n}', value=0.0, vary=var, min=AB_min, max=AB_max)
                else:
                    self.params[f'Bc{num}_{m}_{n}'].vary = var
                # D is the phi phase
                if f'Dc{num}_{n}' not in self.params:
                    if n > 0:
                        self.params.add(f'Dc{num}_{n}', value=np.pi/4., min=0, max=np.pi, vary=True)
                    # n=0 is constant term, no phase
                    else:
                        self.params.add(f'Dc{num}_{n}', value=np.pi/2, vary=False)
                elif n > 0:
                    self.params[f'Dc{num}_{n}'].min = 0.0
                    self.params[f'Dc{num}_{n}'].max = np.pi

    def add_params_cyl_v1005(self, num):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        #np.random.seed(0)
        m_max = self.params['ms_asym_max'].value
        if m_max < 0:
            m_max = np.inf

        for m in ms_range:
            for n in ns_range:
                if (m > m_max) & (n > 0):
                    var = False
                else:
                    var = True
                # A and B are linear
                if f'Ac{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Ac{num}_{m}_{n}', value=0.0, vary=var)
                if f'Bc{num}_{m}_{n}' not in self.params:
                    self.params.add(f'Bc{num}_{m}_{n}', value=0.0, vary=var)
                # D is the phi phase
                if f'Dc{num}_{n}' not in self.params:
                    if n > 0:
                        self.params.add(f'Dc{num}_{n}', value=np.pi/4., min=0, max=np.pi, vary=True)
                    # n=0 is constant term, no phase
                    else:
                        self.params.add(f'Dc{num}_{n}', value=np.pi/2, vary=False)
                elif n > 0:
                    self.params[f'Dc{num}_{n}'].min = 0.0
                    self.params[f'Dc{num}_{n}'].max = np.pi
                        
    def add_params_cyl_v1004(self, num):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        np.random.seed(0)
        b_vals = np.random.uniform(0.0,np.pi,len(ms_range))
        m_max = self.params['ms_asym_max'].value
        if m_max < 0:
            m_max = np.inf

        # Get initial values for Ac1_m_0, Bc1_m_0 from Fourier transform of r=0.8 line
        #with open('../data/fit_params/params_fourier_v2.pkl','rb') as infile:
        #    params_fourier = pkl.load(infile)

        # Get initial values for Ac1_m_0, Bc1_m_0 from converted V1 old parameters
        with open('../data/fit_params/params_convert.pkl','rb') as infile:
            params_convert = pkl.load(infile)
        
        for m in ms_range:
            for n in ns_range:
                if (m > m_max) & (n > 0):
                    var = False
                else:
                    var = True
                # Overall normalization
                #if params_fourier[f'Bc{num}_{m}_{n}'] < 0.0:
                #    params_fourier[f'Bc{num}_{m}_{n}'] = params_fourier[f'Bc{num}_{m}_{n}'] + np.pi
                #    params_fourier[f'Ac{num}_{m}_{n}'] = -1.0*params_fourier[f'Ac{num}_{m}_{n}']
                if f'Ac{num}_{m}_{n}' not in self.params:
                    #self.params.add(f'Ac{num}_{m}_{n}', value=params_fourier[f'Ac{num}_{m}_{n}'], min=0.0, vary=var) #V4
                    self.params.add(f'Ac{num}_{m}_{n}', value=0.0, vary=var) #V2,V3
                    #self.params.add(f'Ac{num}_{m}_{n}', value=params_convert['Ac1_new_conv'][m], min=0.0, vary=var) #V6
                # Z phase
                if f'Bc{num}_{m}_{n}' not in self.params:
                    #self.params.add(f'Bc{num}_{m}_{n}', value=params_fourier[f'Bc{num}_{m}_{n}'], min=-np.pi, max=np.pi, vary=var) #V4
                    #self.params.add(f'Bc{num}_{m}_{n}', value=np.pi/2, min=0.0, max=np.pi, vary=var) #V2
                    self.params.add(f'Bc{num}_{m}_{n}', value=b_vals[m], min=0.0, max=np.pi, vary=var) #V3
                    #self.params.add(f'Bc{num}_{m}_{n}', value=params_convert['Bc1_new_conv'][m], min=-np.pi, max=np.pi, vary=var) #V6
                else: # If bootstrapping, still want to keep bounds on phase
                    self.params[f'Bc{num}_{m}_{n}'].min = 0.0
                    self.params[f'Bc{num}_{m}_{n}'].max = np.pi
                # Phi phase
                if f'Dc{num}_{n}' not in self.params:
                    if n > 0:
                        self.params.add(f'Dc{num}_{n}', value=np.pi/2, min=0, max=np.pi, vary=True)
                    # n=0 is constant term, no phase
                    else:
                        self.params.add(f'Dc{num}_{n}', value=0., vary=False)
                elif n > 0: # If bootstrapping, still want to keep bounds on phase
                    self.params[f'Dc{num}_{n}'].min = 0.0
                    self.params[f'Dc{num}_{n}'].max = np.pi

    def add_params_cyl_v1002(self, num):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        # np.random.seed(101)
        # why do we initialize d like this?
        # d_vals = np.linspace(0, 1, len(ns_range))[::-1]
        d_vals = 0.5*np.ones(len(ns_range))

        # kludge for now...FIXME!
        if self.params[f'ns_c{num}'].value < 2:
            for m in ms_range:
                for n in ns_range:
                    # if (n-1) % 4!= 0:
                    if n == -1:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=False)
                        if f'Cc{num}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{n}', value=0, min=0, max=1, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                    else:
                        # seems like a weird initialization...
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            # self.params.add(f'Ac{num}_{m}_{n}', value=1e-6, vary=True)
                            self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e5, max=1e5, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=0, min=-1e4, max=1e4, vary=True)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1e-6, vary=True)
                            self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e5, max=1e5, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=0, min=-1e4, max=1e4, vary=True)
                        if f'Cc{num}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{n}', value=1.,
                                            min=0, max=1, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0.,
                                            min=0, max=1, vary=False)
        else:
            for m in ms_range:
                for n in ns_range:
                    # if (n-1) % 4!= 0:
                    if n == -1:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=False)
                        if f'Cc{num}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{n}', value=0, min=0, max=1, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                    else:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0., vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0., vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                        if f'Cc{num}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{n}', value=d_vals[n], vary=True)
                                            #min=0, max=1, vary=True)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=d_vals[n], vary=True)
                                            #min=0, max=1, vary=True)
                                            # min=-1, max=1, vary=True)

    def add_params_cyl_v1001(self, num):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        # np.random.seed(101)
        # why do we initialize d like this?
        # d_vals = np.linspace(0, 1, len(ns_range))[::-1]
        d_vals = 0.5*np.ones(len(ns_range))

        # kludge for now...FIXME!
        if self.params[f'ns_c{num}'].value < 2:
            for m in ms_range:
                for n in ns_range:
                    # if (n-1) % 4!= 0:
                    if n == -1:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, min=0, max=1, vary=False)
                        if f'Cc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{m}_{n}', value=0, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                    else:
                        # seems like a weird initialization...
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            # self.params.add(f'Ac{num}_{m}_{n}', value=1e-6, vary=True)
                            self.params.add(f'Ac{num}_{m}_{n}', value=0.5, min=0, max=1, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e5, max=1e5, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=0, min=-1e4, max=1e4, vary=True)
                        if f'Cc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{m}_{n}', value=0., vary=True)
                        #     self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                        #     # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e5, max=1e5, vary=True)
                        #     # self.params.add(f'Bc{num}_{m}_{n}', value=0, min=-1e4, max=1e4, vary=True)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=d_vals[n],
                                            min=0, max=1, vary=False)
        else:
            for m in ms_range:
                for n in ns_range:
                    # if (n-1) % 4!= 0:
                    if n == -1:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                        if f'Cc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{m}_{n}', value=0, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                    else:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            # self.params.add(f'Ac{num}_{m}_{n}', value=0., vary=True)
                            self.params.add(f'Ac{num}_{m}_{n}', value=0.5, min=0, max=1, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                        if f'Cc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Cc{num}_{m}_{n}', value=0., vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=d_vals[n],
                                            min=0, max=1, vary=True)
                                            # min=-1, max=1, vary=True)


    def add_params_cyl(self, num):
        ms_range = range(self.params[f'ms_c{num}'].value)
        ns_range = range(self.params[f'ns_c{num}'].value)
        d_vals = 0.5*np.ones(len(ns_range))
        #np.random.seed(101)
        #d_vals = np.random.random(len(ns_range))
        # is there a max for the asymmetric terms?
        m_max = self.params['ms_asym_max'].value
        if m_max < 0:
            m_max = np.inf

        # kludge for now...FIXME!
        # no asymmetric terms here
        if self.params[f'ns_c{num}'].value < 2:

            # Get initial values for Ac1_m_0, Bc1_m_0 from converted V1 old parameters
            #with open('../data/fit_params/params_convert.pkl','rb') as infile:
            #    params_convert = pkl.load(infile)

            for m in ms_range:
                for n in ns_range:
                    # if (n-1) % 4!= 0:
                    if n == -1:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                    else:
                        # seems like a weird initialization...
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=True)
                            #self.params.add(f'Ac{num}_{m}_{n}', value=-8000, min=-8000, max=-7000, brute_step=100, vary=True)
                            #self.params.add(f'Ac{num}_{m}_{n}', value=params_convert['Ac1_old_conv'][m], vary=True) #V5
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=True)
                            #self.params.add(f'Bc{num}_{m}_{n}', value=12000, min=12000, max=14000, brute_step=200, vary=True)
                            #self.params.add(f'Bc{num}_{m}_{n}', value=params_convert['Bc1_old_conv'][m], vary=True) #V5
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0.0,
                                            min=0, max=1, vary=False)
        else:
            for m in ms_range:
                for n in ns_range:
                    if (m > m_max) & (n > 0):
                        var = False
                    else:
                        var = True
                    # if (n-1) % 4!= 0:
                    if n == -1:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            self.params.add(f'Dc{num}_{n}', value=0, min=0, max=1, vary=False)
                    else:
                        if f'Ac{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Ac{num}_{m}_{n}', value=0., vary=var)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                            # self.params.add(f'Ac{num}_{m}_{n}', value=0, vary=False)
                        if f'Bc{num}_{m}_{n}' not in self.params:
                            self.params.add(f'Bc{num}_{m}_{n}', value=0., vary=var)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=-1*(-1)**m, min=-1e4, max=1e4, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=0., min=-1e4, max=1e4, vary=True)
                            # self.params.add(f'Bc{num}_{m}_{n}', value=0, vary=False)
                        if f'Dc{num}_{n}' not in self.params:
                            # self.params.add(f'Dc{num}_{n}', value=d_vals[n],
                            #                 min=0, max=1, vary=True)
                            if n > 0:
                                D_var = True
                                dval = d_vals[n]
                            else:
                                D_var = False
                                dval = 0.
                            self.params.add(f'Dc{num}_{n}', value=dval,
                                            min=0, max=1, vary=D_var)
                        elif n > 0:
                            self.params[f'Dc{num}_{n}'].min = 0
                            self.params[f'Dc{num}_{n}'].max = 1

    def add_params_cart_simple(self, cfg_params):
        ks_dict = cfg_params.ks_dict
        if ks_dict is None:
            ks_dict = {}
        cart_names = [f'k{i}' for i in range(1, 11)]

        for k in cart_names:
            if k not in self.params and k in ks_dict:
                if k == 'k3':
                    # Get initial value for k3 from Fourier transform of r=0 line
                    # with open('../data/fit_params/params_fourier.pkl','rb') as infile:
                    #     params_fourier = pkl.load(infile)
                    # self.params.add(k, value=params_fourier['k3'], vary=True)#True
                    # self.params.add(k, value=ks_dict[k], vary=True)
                    self.params.add(k, value=ks_dict[k], min=0.0, vary=True)
                    #self.params.add(k, value=13000, vary=True, min=13000, max=14000, brute_step=100)
                    #with open('../data/fit_params/DS_only_V1_results.p','rb') as infile:
                    #    params_V1 = pkl.load(infile)
                    #self.params.add(k, value=params_V1['k3'].value, vary=True)#True
                else:
                    self.params.add(k, value=ks_dict[k], vary=True)
            elif k not in self.params:
                self.params.add(k, value=0, vary=False)
            elif k == 'k3':
                self.params[k].min=0.0

    def add_params_cart_simple_fixable(self, cfg_params):
        ks_dict = cfg_params.ks_dict
        if ks_dict is None:
            ks_dict = {}
        cart_names = [f'k{i}' for i in range(1, 11)]

        # limits for k
        k_lim = cfg_params.k_lim
        if k_lim is None:
            k_min = None
            k_max = None
        else:
            k_min = -k_lim
            k_max = k_lim

        for k in cart_names:
            if k not in self.params and k in ks_dict:
                self.params.add(k, value=ks_dict[k][0], vary=ks_dict[k][1], min=k_min, max=k_max)
                # if k == 'k3':
                #     # Get initial value for k3 from Fourier transform of r=0 line
                #     # with open('../data/fit_params/params_fourier.pkl','rb') as infile:
                #     #     params_fourier = pkl.load(infile)
                #     # self.params.add(k, value=params_fourier['k3'], vary=True)#True
                #     # self.params.add(k, value=ks_dict[k], vary=True)
                #     self.params.add(k, value=ks_dict[k], min=0.0, vary=True)
                #     #self.params.add(k, value=13000, vary=True, min=13000, max=14000, brute_step=100)
                #     #with open('../data/fit_params/DS_only_V1_results.p','rb') as infile:
                #     #    params_V1 = pkl.load(infile)
                #     #self.params.add(k, value=params_V1['k3'].value, vary=True)#True
                # else:
                #     self.params.add(k, value=ks_dict[k], vary=True)
            elif k not in self.params:
                self.params.add(k, value=0, vary=False)
            #elif k == 'k3':
            #    self.params[k].min=0.0

    def add_params_finite_wire(self):
        if 'k1' not in self.params:
            self.params.add('k1', value=0, vary=True)
        if 'k2' not in self.params:
            self.params.add('k2', value=0, vary=True)
        if 'xp1' not in self.params:
            self.params.add('xp1', value=1050, vary=False, min=900, max=1200)
        if 'xp2' not in self.params:
            self.params.add('xp2', value=1050, vary=False, min=900, max=1200)
        if 'yp1' not in self.params:
            self.params.add('yp1', value=0, vary=False, min=-100, max=100)
        if 'yp2' not in self.params:
            self.params.add('yp2', value=0, vary=False, min=-100, max=100)
        if 'zp1' not in self.params:
            self.params.add('zp1', value=4575, vary=False, min=4300, max=4700)
        if 'zp2' not in self.params:
            self.params.add('zp2', value=-4575, vary=False, min=-4700, max=-4300)

    def add_params_biot_savart(self, cfg_params, recreate=False):
        xyz_tuples = cfg_params.bs_tuples
        bounds = cfg_params.bs_bounds
        if xyz_tuples is None:
            return
        if bounds is None:
            bounds = (0.1, 0.1, 5)

        for i in range(1, len(xyz_tuples)+1):
            if len(xyz_tuples[i-1]) == 3:
                x, y, z = xyz_tuples[i-1]
                vx = vy = vz = 0
                do_vary = True
            else:
                x, y, z, vx, vy, vz = xyz_tuples[i-1]
                do_vary = False

            if f'x{i}' not in self.params:
                self.params.add(f'x{i}', value=x, vary=do_vary,
                                min=x-bounds[0], max=x+bounds[0])
            elif not recreate:
                self.params[f'x{i}'].value = x
                self.params[f'x{i}'].min = x-bounds[0]
                self.params[f'x{i}'].max = x+bounds[0]

            if f'y{i}' not in self.params:
                self.params.add(f'y{i}', value=y, vary=do_vary,
                                min=y-bounds[0], max=y+bounds[0])
            elif not recreate:
                self.params[f'y{i}'].value = y
                self.params[f'y{i}'].min = y-bounds[0]
                self.params[f'y{i}'].max = y+bounds[0]

            if f'z{i}' not in self.params:
                self.params.add(f'z{i}', value=z, vary=do_vary,
                                min=z-bounds[1], max=z+bounds[1])
            elif not recreate:
                self.params[f'z{i}'].value = z
                self.params[f'z{i}'].min = z-bounds[1]
                self.params[f'z{i}'].max = z+bounds[1]

            if f'vx{i}' not in self.params:
                self.params.add(f'vx{i}', value=vx, vary=do_vary,
                                min=-bounds[2], max=bounds[2])
            elif not recreate:
                self.params[f'vx{i}'].value = vx
                self.params[f'vx{i}'].min = -bounds[2]
                self.params[f'vx{i}'].max = bounds[2]

            if f'vy{i}' not in self.params:
                self.params.add(f'vy{i}', value=vy, vary=do_vary,
                                min=-bounds[2], max=bounds[2])
            elif not recreate:
                self.params[f'vy{i}'].value = vy
                self.params[f'vy{i}'].min = -bounds[2]
                self.params[f'vy{i}'].max = bounds[2]

            if f'vz{i}' not in self.params:
                self.params.add(f'vz{i}', value=vz, vary=do_vary,
                                min=-bounds[2], max=bounds[2])
            elif not recreate:
                self.params[f'vz{i}'].value = vz
                self.params[f'vz{i}'].min = -bounds[2]
                self.params[f'vz{i}'].max = bounds[2]
