#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file
"""
If Simulation without temperature and slope data (Itmix for instance):
    - Use either slidingco.nc (which may need to be regridded)., a dataset computed using oggmshopCook for a given RGI id.
        This case is usefull for instance with one sc value per subglacier and thus a non unifrom slidingco field.
    -  Or use a float input as slidingco on the whole glacier. This case is useful for moutain glacier with uniform slidingco field.
          You can submit up to 3 scalar values to return a non unifrom slidingco values in order to better match velocity field (see skip infer_params)


Use velmag instead of vecotized velocity in the cost function

"""

import numpy as np
import os, copy
import matplotlib.pyplot as plt
import matplotlib
import datetime, time
import math
import tensorflow as tf
from scipy import stats
from netCDF4 import Dataset

from igm.modules.utils import *
from igm.modules.process.iceflow import initialize as initialize_iceflow
from igm.modules.process.iceflow import params as params_iceflow

from igm.modules.process.iceflow.iceflow import (
    fieldin_to_X,
    update_2d_iceflow_variables,
    iceflow_energy_XY,
    _update_iceflow_emulator,
    Y_to_UV,
    save_iceflow_model
)


def params(parser):
    # dependency on iceflow parameters...
    params_iceflow(parser)

    parser.add_argument(
        "--opti_vars_to_save",
        type=list,
        default=[
            "usurf",
            "thk",
            "slidingco",
            "velsurf_mag",
            "velsurfobs_mag",
            "divflux",
            "icemask",
        ],
        help="List of variables to be recorded in the ncdef file",
    )
    parser.add_argument(
        "--opti_init_zero_thk",
        type=str2bool,
        default="False",
        help="Initialize the optimization with zero ice thickness",
    )
    parser.add_argument(
        "--opti_regu_param_thk",
        type=float,
        default=10.0,
        help="Regularization weight for the ice thickness in the optimization",
    )
    parser.add_argument(
        "--opti_regu_param_slidingco",
        type=float,
        default=1,
        help="Regularization weight for the strflowctrl field in the optimization",
    )
    parser.add_argument(
        "--opti_regu_param_arrhenius",
        type=float,
        default=1,
        help="Regularization weight for the strflowctrl field in the optimization",
    )
    parser.add_argument(
        "--opti_regu_laplacian",
        type=float,
        default=1,
        help="Regularization weight for the laplacian of thk/usurf field in the optimization",
    )
    
    parser.add_argument(
        "--opti_regu_param_div",
        type=float,
        default=1,
        help="Regularization weight for the divrgence field in the optimization",
    )
    parser.add_argument(
        "--opti_smooth_anisotropy_factor",
        type=float,
        default=0.2,
        help="Smooth anisotropy factor for the ice thickness regularization in the optimization",
    )
    parser.add_argument(
        "--opti_convexity_weight",
        type=float,
        default=0.00,
        help="Convexity weight for the ice thickness regularization in the optimization",
    )
    parser.add_argument(
        "--opti_convexity_power",
        type=float,
        default=1.3,
        help="Power b in the area-volume scaling V ~ a * A^b taking fom 'An estimate of global glacier volume', A. Grinste, TC, 2013",
    )
    parser.add_argument(
        "--opti_usurfobs_std",
        type=float,
        default=2.0,
        help="Confidence/STD of the top ice surface as input data for the optimization",
    )
    parser.add_argument(
        "--opti_velsurfobs_std",
        type=float,
        default=1.0,
        help="Confidence/STD of the surface ice velocities as input data for the optimization (if 0, velsurfobs_std field must be given)",
    )
    parser.add_argument(
        "--opti_thkobs_std",
        type=float,
        default=3.0,
        help="Confidence/STD of the ice thickness profiles (unless given)",
    )
    parser.add_argument(
        "--opti_divfluxobs_std",
        type=float,
        default=1.0,
        help="Confidence/STD of the flux divergence as input data for the optimization (if 0, divfluxobs_std field must be given)",
    )
    parser.add_argument(
        "--opti_divflux_method",
        type=str,
        default="upwind",
        help="Compute the divergence of the flux using the upwind or centered method",
    )
    parser.add_argument(
        "--opti_force_zero_sum_divflux",
        type=str2bool,
        default="False",
        help="Add a penalty to the cost function to force the sum of the divergence of the flux to be zero",
    )
    parser.add_argument(
        "--opti_scaling_thk",
        type=float,
        default=2.0,
        help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_scaling_usurf",
        type=float,
        default=0.5,
        help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_scaling_slidingco",
        type=float,
        default=0.0001,
        help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_scaling_arrhenius",
        type=float,
        default=0.01,
        help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_control",
        type=list,
        default=["thk"],  # "slidingco", "usurf"
        help="List of optimized variables for the optimization",
    )
    parser.add_argument(
        "--opti_cost",
        type=list,
        default=["velsurf", "thk", "icemask"],  # "divfluxfcz", ,"usurf"
        help="List of cost components for the optimization",
    )
    parser.add_argument(
        "--opti_nbitmin",
        type=int,
        default=50,
        help="Min iterations for the optimization",
    )
    parser.add_argument(
        "--opti_nbitmax",
        type=int,
        default=500,
        help="Max iterations for the optimization",
    )
    parser.add_argument(
        "--opti_step_size",
        type=float,
        default=1,
        help="Step size for the optimization",
    )
    parser.add_argument(
        "--opti_step_size_decay",
        type=float,
        default=0.9,
        help="Decay step size parameter for the optimization",
    )
    parser.add_argument(
        "--opti_output_freq",
        type=int,
        default=50,
        help="Frequency of the output for the optimization",
    )
    parser.add_argument(
        "--opti_save_result_in_ncdf",
        type=str,
        default="geology-optimized.nc",
        help="Geology input file",
    )
    parser.add_argument(
        "--opti_plot2d_live",
        type=str2bool,
        default=True,
        help="plot2d_live_inversion",
    )
    parser.add_argument(
        "--opti_plot2d",
        type=str2bool,
        default=True,
        help="plot 2d inversion",
    )
    parser.add_argument(
        "--opti_save_iterat_in_ncdf",
        type=str2bool,
        default=True,
        help="write_ncdf_optimize",
    )
    parser.add_argument(
        "--opti_editor_plot2d",
        type=str,
        default="vs",
        help="optimized for VS code (vs) or spyder (sp) for live plot",
    )
    parser.add_argument(
        "--opti_uniformize_thkobs",
        type=str2bool,
        default=True,
        help="uniformize the density of thkobs",
    )
    parser.add_argument(
        "--sole_mask",
        type=str2bool,
        default=False,
        help="sole_mask",
    )
    parser.add_argument(
        "--regu_sole_mask",
        type=str2bool,
        default=False,
        help="sole_mask for regularization",
    )
    parser.add_argument(
        "--opti_retrain_iceflow_model",
        type=str2bool,
        default=True,
        help="Retrain the iceflow model simulatounously ?",
    )
    parser.add_argument(
       "--opti_to_regularize",
       type=str,
       default='topg',
       help="Field to regularize : topg or thk",
   )
    parser.add_argument(
       "--opti_include_low_speed_term",
       type=str2bool,
       default=False,
       help="opti_include_low_speed_term",
   ) 
    parser.add_argument(
        "--opti_infer_params",
        type=str2bool,
        default=False,
        help="infer slidingco and convexity weight from velocity observations. If False,Skip infer_params to compute slidingco but load useful variable for the optimization which are normally loaded by infer_param",
    )
    parser.add_argument(
        "--opti_tidewater_glacier",
        type=str2bool,
        default=False,
        help="Is the glacier you're trying to infer parameters for a tidewater type?",
    )
    parser.add_argument(
        "--opti_vol_std",
        type=float,
        default=0.1, #1000.0,
        help="Confidence/STD of the volume estimates from volume-area scaling",
    )
    parser.add_argument(
        "--opti_postprocess",
        type=str2bool,
        default=False,
        help="Your name is Samuel Cook and you need to do a second round of optimisation to post-process the first round",
    )
    parser.add_argument(
    "--opti_vol_factor",
    type=float,
    default=1,
    help="Modify the Volume area scaling law by multiplying it by this factor, ideally close to 1",
)
    parser.add_argument(
       "--opti_costvol",
       type=str2bool,
       default=False,
       help="Take volume into account in the cost function",
   ) 
    parser.add_argument(
       "--opti_load_sc_path",
       type=str,
       default='../data/slidingco.nc',
       help="Path to slidingco computed with infer_params and OGGMSHOP if opti_load_sc = 'field' ",
   )
    parser.add_argument(
       "--opti_load_sc",
       type=str,
       default='False',
       help="Load a scalar ('scalar'), a field ('field') or nothing ('False') to define slidingco as computed with infer_params and OGGMSHOP",
   )
    parser.add_argument(
     "--opti_sc_value",
     type=float,
     default=0.04,
     help="Scalar value of the slidingco infered by infer_params and OGGMSHOP if opti_load_sc = 'scalar' ",
 ) 
    parser.add_argument(
       "--opti_modify_sc",
       type=str2bool,
       default=False,
       help="Modify the uniform slidingco field to better represent velocity that are not given by the input field",
   ) 
    parser.add_argument(
     "--opti_sc_mod",
     type=float,
     default=0.0,
     help="Modify the scalar slidingco infered by infer_params and OGGM_shop by a given value",
 ) 
    parser.add_argument(
     "--opti_sc_high",
     type=float,
     default=0.,
     help="Slidingco value above the defined input velocity field. Defined as Sum wrt to the initial uniform sc field.",
 ) 
    parser.add_argument(
     "--opti_sc_low",
     type=float,
     default=0.,
     help="Slidingco value below the defined input velocity field. Defined as Sum wrt to the initial uniform sc field.",
 ) 
    parser.add_argument(
       "--opti_modify_arrhenius",
       type=str2bool,
       default=False,
       help="Modify the uniform Arrhenius field to better represent velocity that are not given by th input field",
   ) 
    parser.add_argument(
     "--opti_arrhenius_high",
     type=float,
     default=0.,
     help="Arrhenius value above the defined input velocity field. Defined as Sum wrt to the initial uniform field.",
 ) 
    parser.add_argument(
     "--opti_arrhenius_low",
     type=float,
     default=0.,
     help="Arrhenius value below the defined input velocity field. Defined as Sum wrt to the initial uniform field.",
 ) 
    parser.add_argument(
     "--opti_altitude_mask",
     type=str2bool,
     default=False,
     help="Modify the uniform Arrhenius field to better represent velocity that are not given by the input field",
 ) 
    parser.add_argument(
    "--opti_icecaps",
    type=str2bool,
    default=False,
    help="Useful for the volume area scaling law which is different for a glaicer and an icecap.",
) 
    parser.add_argument(
    "--opti_flowdir_mask",
    type=str2bool,
    default=False,
    help="If True, compute the flowdir (normalized velocity) on the icemask only (it extends a bit further due to the gaussian smoothing)",
) 
    parser.add_argument(
    "--opti_vol_factor_path",
    type=str,
    default=None,
    help="Modify the Volume area scaling law by multiplying it by this factor, ideally close to 1, frm the csv in the path",
)
    parser.add_argument(
        "--velthresh",
        type=float,
        default=10,
        help="Velocity slower than this treshold are not considered by the inversion process",
    )
    
    parser.add_argument(
    "--opti_mask_laplacian",
    type=str2bool,
    default=False,
    help="If True, compute laplacian on this part of the field"
) 
    parser.add_argument(
    "--opti_maskvel_laplacian",
    type=str2bool,
    default=False,
    help="If True, compute laplacian on this part of the field"
) 
def initialize(params, state):
    """
    This function does the data assimilation (inverse modelling) to optimize thk, slidingco ans usurf from data
    """

    try:
        initialize_iceflow(params, state)
        
        state.it = -1
        
        _update_iceflow_emulator(params, state)
    except:
        print('Problem with iceflow?')
        return

    ###### PERFORM CHECKS PRIOR OPTIMIZATIONS

    # make sure this condition is satisfied
    assert ("usurf" in params.opti_cost) == ("usurf" in params.opti_control)

    # make sure that there are lease some profiles in thkobs
    if params.opti_postprocess == False:
        if tf.reduce_all(tf.math.is_nan(state.thkobs)):
            if "thk" in params.opti_cost:
                params.opti_cost.remove("thk")

    ###### PREPARE DATA PRIOR OPTIMIZATIONS

    if "divfluxobs" in params.opti_cost:
        try:
            state.divfluxobs = state.smb - state.dhdt
        except:
            state.divfluxobs = state.smbobs - state.dhdt

    if hasattr(state, "thkinit"):
        state.thk = state.thkinit
        state.thk = tf.where(tf.math.is_nan(state.thk),0.0,state.thk)
        state.thkinit = tf.where(tf.math.is_nan(state.thkinit),0.0,state.thkinit)
        if tf.math.is_inf(tf.math.reduce_sum(state.thkinit)):
            state.thkinit = tf.where(state.icemask > 0.5, 10.0, state.thkinit)
            state.thk = tf.where(state.icemask > 0.5, 10.0, state.thk)
    elif params.opti_postprocess == True:
        state.thkinit = state.thk
    else:
        state.thk = tf.zeros_like(state.thk)

    if params.opti_init_zero_thk:
        state.thk = state.thk*0.0
        
    # this is a density matrix that will be used to weight the cost function
    if params.opti_postprocess == False:
        if params.opti_uniformize_thkobs:
            state.dens_thkobs = create_density_matrix(state.thkobs, kernel_size=25)
            state.dens_thkobs = tf.where(state.dens_thkobs>0, 1.0/state.dens_thkobs, 0.0)
            state.dens_thkobs = tf.where(tf.math.is_nan(state.thkobs),0.0,state.dens_thkobs)
            state.dens_thkobs = state.dens_thkobs / tf.reduce_mean(state.dens_thkobs[state.dens_thkobs>0])
        else:
            state.dens_thkobs = tf.ones_like(state.thkobs)
        
    if params.opti_postprocess == True:
        state.icemaskobs = state.icemask
        InputObs = Dataset(params.lncd_input_file[:-22]+"input_saved.nc", "r")
        for var in InputObs.variables:
           if var in ["uvelsurfobs", "vvelsurfobs"]:
               vars()[var] = np.squeeze(InputObs.variables[var]).astype("float32")
               vars()[var] = np.where(vars()[var] > 10**35, np.nan, vars()[var])
               vars(state)[var] = tf.Variable(vars()[var].astype("float32"), trainable=False)
        state.uvelsurf = state.uvelsurfobs
        state.vvelsurf = state.vvelsurfobs
        VelMag = getmag(state.uvelsurfobs, state.vvelsurfobs)
        VelMag = tf.where(tf.math.is_nan(VelMag),1e-6,VelMag)
        VelMag = tf.where(VelMag==0,1e-6,VelMag)
        VelMean = np.round(np.mean(VelMag[state.icemaskobs>0.5]),decimals=2)
        params.opti_nbitmax = 1000
        if VelMean == 0.0:
            NoVel = True
            params.opti_nbitmax = 1
        else:
            NoVel = False
            state.slidingco = tf.where(state.slidingco > 0, params.iflo_init_slidingco, params.iflo_init_slidingco)
    
    # force zero slidingco in the floating areas
    #state.slidingco = tf.where( state.icemaskobs == 2, 0.0, state.slidingco)
    
    # Force zero thickness and no velocities outside the ice mask
    try:
        state.uvelsurfobs = tf.where(state.icemaskobs < 0.5, np.nan, state.uvelsurfobs)
        state.vvelsurfobs = tf.where(state.icemaskobs < 0.5, np.nan, state.vvelsurfobs)
    except:
        pass
    state.thk = tf.where(state.icemaskobs < 0.5, 0.0, state.thk)
    if hasattr(state, 'uvelsurfobs'):
        print('Threshold applied to velocity')
        threshold = params.velthresh
        nanmask = tf.where(state.icemask>0.5,state.icemask,tf.constant(np.nan))
        velmag = getmag(state.uvelsurfobs, state.vvelsurfobs)
        state.uvelsurfobs = tf.where(velmag > threshold,state.uvelsurfobs,tf.constant(np.nan))
        state.vvelsurfobs = tf.where(velmag > threshold,state.vvelsurfobs,tf.constant(np.nan))
    else :
        pass
    _optimize(params, state)

def _optimize(params, state):
    # print(params.opti_infer_params)
    # this will infer values for slidingco and convexity weight based on the ice velocity and an empirical relationship from test glaciers with thickness profiles
    if params.opti_infer_params == True:
        print('INFER PARAM')
        #Because OGGM will index icemask from 0
        dummy = _infer_params(state, params)
        if tf.reduce_max(state.icemask).numpy() < 1:
            return
    else :
        print('No INFER PARAM')
        dummy = skip_infer_params(state, params)
    if (int(tf.__version__.split(".")[1]) <= 10) | (int(tf.__version__.split(".")[1]) >= 16) :
        optimizer = tf.keras.optimizers.Adam(learning_rate=params.opti_step_size)
        opti_retrain = tf.keras.optimizers.Adam(
            learning_rate=params.iflo_retrain_emulator_lr
        )
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=params.opti_step_size)
        opti_retrain = tf.keras.optimizers.legacy.Adam(
            learning_rate=params.iflo_retrain_emulator_lr
        )

    state.costs = []

    state.tcomp_optimize = []

    # this thing is outdated with using iflo_new_friction_param default as we use scaling of one.
    sc = {}
    sc["thk"] = params.opti_scaling_thk
    sc["usurf"] = params.opti_scaling_usurf
    sc["slidingco"] = params.opti_scaling_slidingco
    sc["arrhenius"] = params.opti_scaling_arrhenius
    Ny, Nx = state.thk.shape

    for f in params.opti_control:
        vars()[f] = tf.Variable(vars(state)[f] / sc[f])
    #compute a shrinked icemask to use when computing laplacian of the field
    from scipy.ndimage import binary_erosion
    
    def erode_mask(icemask, iterations=5):
        def erosion_fn(mask_np):
            return binary_erosion(mask_np, iterations=iterations).astype(np.float32)
    
        eroded_mask = tf.numpy_function(erosion_fn, [icemask], tf.float32)
        return eroded_mask
    eroded_icemask = erode_mask(state.icemask)
    
    def maskvel(state, params):
        mask_vel = state.vvelsurfobs/state.vvelsurfobs
        mask_vel = tf.where(mask_vel==1, mask_vel,0)
        return mask_vel
    # mask_vel = maskvel(state,params)
    # main loop
    for i in range(params.opti_nbitmax):
        with tf.GradientTape() as t, tf.GradientTape() as s:
            state.tcomp_optimize.append(time.time())
            
            if params.opti_step_size_decay < 1:
                optimizer.lr = params.opti_step_size * (params.opti_step_size_decay ** (i / 100))

            # is necessary to remember all operation to derive the gradients w.r.t. control variables
            for f in params.opti_control:
                t.watch(vars()[f])

            for f in params.opti_control:
                vars(state)[f] = vars()[f] * sc[f]

            fieldin = [vars(state)[f] for f in params.iflo_fieldin]

            X = fieldin_to_X(params, fieldin)

            # evalutae th ice flow emulator
            # evalutae th ice flow emulator                
            if params.iflo_multiple_window_size==0:
                Y = state.iceflow_model(X)
            else:
                Y = state.iceflow_model(tf.pad(X, state.PAD, "CONSTANT"))[:, :Ny, :Nx, :]

            U, V = Y_to_UV(params, Y)

            U = U[0]
            V = V[0]
            U = tf.where(state.thk > 0, U, 0)
            V = tf.where(state.thk > 0, V, 0)
            # this is strange, but it having state.U instead of U, slidingco is not more optimized ....
            state.uvelbase = U[0, :, :]
            state.vvelbase = V[0, :, :]
            state.ubar = tf.reduce_sum(U * state.vert_weight, axis=0)
            state.vbar = tf.reduce_sum(V * state.vert_weight, axis=0)
            state.uvelsurf = U[-1, :, :]
            state.vvelsurf = V[-1, :, :]

            if not params.opti_smooth_anisotropy_factor == 1:
                _compute_flow_direction_for_anisotropic_smoothing(state, params)

            cost = {} 
                 
            # misfit between surface velocity
            if "velsurf" in params.opti_cost:
                cost["velsurf"] = misfit_velsurf(params,state)

            # misfit between ice thickness profiles
            if "thk" in params.opti_cost:
                cost["thk"] = misfit_thk(params, state)

            # misfit between divergence of flux
            if ("divfluxfcz" in params.opti_cost):
                cost["divflux"] = cost_divflux(params, state, i)
            elif ("divfluxobs" in params.opti_cost):
                cost["divflux"] = cost_divflux(params, state, i)
 
            # misfit between top ice surfaces
            if "usurf" in params.opti_cost:
                cost["usurf"] = misfit_usurf(params, state) 

            # force zero thikness outisde the mask
            if "icemask" in params.opti_cost:
                cost["icemask"] = 10**10 * tf.math.reduce_mean( tf.where(state.icemaskobs > 0.5, 0.0, state.thk**2) )

            # Here one enforces non-negative ice thickness, and possibly zero-thickness in user-defined ice-free areas.
            if "thk" in params.opti_control:
                cost["thk_positive"] = 10**10 * tf.math.reduce_mean( tf.where(state.thk >= 0, 0.0, state.thk**2) )
                
            if params.opti_infer_params:
                cost["volume"] = cost_vol(params, state)
               
            elif params.opti_costvol :
                cost["volume"] = cost_vol(params, state)
            else:
                 cost["volume"]   = tf.Variable(0.0)
    
            # Here one adds a regularization terms for the bed toporgraphy to the cost function
            if "thk" in params.opti_control:
                cost["thk_regu"] = regu_thk(params, state)
                # cost["laplacian"] = smooth_thk(params,state,eroded_icemask,mask_vel)

            # Here one adds a regularization terms for slidingco to the cost function
            if "slidingco" in params.opti_control:
                cost["slid_regu"] = regu_slidingco(params, state)

            # Here one adds a regularization terms for arrhenius to the cost function
            if "arrhenius" in params.opti_control:
                cost["arrh_regu"] = regu_arrhenius(params, state) 
  
            cost_total = tf.reduce_sum(tf.convert_to_tensor(list(cost.values())))

            # Here one allow retraining of the ice flow emaultor
            if params.opti_retrain_iceflow_model:
               

                cost["glen"] = iceflow_energy_XY(params, X, Y)
                
                grads = s.gradient(cost["glen"], state.iceflow_model.trainable_variables)

                opti_retrain.apply_gradients(
                    zip(grads, state.iceflow_model.trainable_variables)
                )
            else:
                COST_GLEN = tf.Variable(0.0)

            print_costs(params, state, cost, i)

            #################

            var_to_opti = [ ]
            for f in params.opti_control:
                var_to_opti.append(vars()[f])

            # Compute gradient of COST w.r.t. X
            grads = tf.Variable(t.gradient(cost_total, var_to_opti))

            # this serve to restict the optimization of controls to the mask
            if params.sole_mask:
                for ii in range(grads.shape[0]):
                    if not "slidingco" == params.opti_control[ii]:
                        grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))
                    else:
                        grads[ii].assign(tf.where((state.icemaskobs == 1), grads[ii], 0))
            else:
                for ii in range(grads.shape[0]):
                    if not "slidingco" == params.opti_control[ii]:
                        grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))

            # One step of descent -> this will update input variable X
            optimizer.apply_gradients(
                zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
            )

            ###################

            # get back optimized variables in the pool of state.variables
            if "thk" in params.opti_control:
                state.thk = tf.where(state.icemaskobs > 0.5, state.thk, 0)
#                state.thk = tf.where(state.thk < 0.01, 0, state.thk)

            state.divflux = compute_divflux(
                state.ubar, state.vbar, state.thk, state.dx, state.dx, method=params.opti_divflux_method
            )

            #state.divflux = tf.where(ACT, state.divflux, 0.0)

            _compute_rms_std_optimization(state, i)

            state.tcomp_optimize[-1] -= time.time()
            state.tcomp_optimize[-1] *= -1

            if i % params.opti_output_freq == 0:
                if params.opti_plot2d:
                    _update_plot_inversion(params, state, i)
                if params.opti_save_iterat_in_ncdf:
                    _update_ncdf_optimize(params, state, i)

            # stopping criterion: stop if the cost no longer decrease
            # if i>params.opti_nbitmin:
            #     cost = [c[0] for c in costs]
            #     if np.mean(cost[-10:])>np.mean(cost[-20:-10]):
            #         break;

	# for final iteration
    i = params.opti_nbitmax

    print_costs(params, state, cost, i)

    if i % params.opti_output_freq == 0:
        if params.opti_plot2d:
            _update_plot_inversion(params, state, i)
        if params.opti_save_iterat_in_ncdf:
            _update_ncdf_optimize(params, state, i)

#    for f in params.opti_control:
#        vars(state)[f] = vars()[f] * sc[f]

    # now that the ice thickness is optimized, we can fix the bed once for all! (ONLY FOR GROUNDED ICE)
    state.topg = state.usurf - state.thk

    if not params.opti_save_result_in_ncdf=="":
        _output_ncdf_optimize_final(params, state)

    plot_cost_functions() 

    plt.close("all")

    save_rms_std(params, state)

    os.system(
        "echo rm " + "clean.sh" + " >> clean.sh"
    )
    # Flag so we can check if initialize was already called
    state.optimize_initializer_called = True
 
####################################


def update(params, state):
    pass


def finalize(params, state):
    if params.iflo_save_model:
        save_iceflow_model(params, state) 

####################################

def misfit_velsurf(params,state):
    velsurf    = tf.stack(getmag(state.uvelsurf,state.vvelsurf))
    velsurfobs = tf.stack(state.velsurfobs_mag)
    
    # velsurf    = tf.stack([state.uvelsurf,    state.vvelsurf],    axis=-1) 
    # velsurfobs = tf.stack([state.uvelsurfobs, state.vvelsurfobs], axis=-1)

    ACT = ~tf.math.is_nan(velsurfobs)

    cost = 0.5 * tf.reduce_mean(
           ( (velsurfobs[ACT] - velsurf[ACT]) / params.opti_velsurfobs_std  )** 2
    )
    
    if tf.math.is_nan(cost):
        cost = tf.Variable(0.0)

    if params.opti_include_low_speed_term:

        # This terms penalize the cost function when the velocity is low
        # Reference : Inversion of basal friction in Antarctica using exact and incompleteadjoints of a higher-order model
        # M. Morlighem, H. Seroussi, E. Larour, and E. Rignot, JGR, 2013
        cost = cost + 0.5 * 100 * tf.reduce_mean(
            tf.math.log( (tf.norm(velsurf[ACT],axis=-1)+1) / (tf.norm(velsurfobs[ACT],axis=-1)+1) )** 2
        )

    return cost


def misfit_thk(params,state):

    ACT = ~tf.math.is_nan(state.thkobs)

    return 0.5 * tf.reduce_mean( state.dens_thkobs[ACT] * 
        ((state.thkobs[ACT] - state.thk[ACT]) / params.opti_thkobs_std) ** 2
    )

def cost_divflux(params,state,i):

    divflux = compute_divflux(
        state.ubar, state.vbar, state.thk, state.dx, state.dx, method=params.opti_divflux_method
    )

    # divflux = tf.where(ACT, divflux, 0.0)

    if "divfluxfcz" in params.opti_cost:
        ACT = state.icemaskobs > 0.5
        if i % 10 == 0:
            # his does not need to be comptued any iteration as this is expensive
            state.res = stats.linregress(
                state.usurf[ACT], divflux[ACT]
            )  # this is a linear regression (usually that's enough)
        # or you may go for polynomial fit (more gl, but may leads to errors)
        #  weights = np.polyfit(state.usurf[ACT],divflux[ACT], 2)
        divfluxtar = tf.where(
            ACT, state.res.intercept + state.res.slope * state.usurf, 0.0
        )
    #   divfluxtar = tf.where(ACT, np.poly1d(weights)(state.usurf) , 0.0 )

    if "divfluxobs" in params.opti_cost:
        divfluxtar = state.divfluxobs

    ACT = state.icemaskobs > 0.5

    if ("divfluxobs" in params.opti_cost) | ("divfluxfcz" in params.opti_cost):
        COST_D = 0.5 * tf.reduce_mean(
            ((divfluxtar[ACT] - divflux[ACT]) / params.opti_divfluxobs_std) ** 2
        )

    if ("divfluxpen" in params.opti_cost):
        dddx = (divflux[:, 1:] - divflux[:, :-1])/state.dx
        dddy = (divflux[1:, :] - divflux[:-1, :])/state.dx
        COST_D = (params.opti_regu_param_div) * ( tf.nn.l2_loss(dddx) + tf.nn.l2_loss(dddy) )

    if params.opti_force_zero_sum_divflux:
            COST_D += 0.5 * 1000 * tf.reduce_mean(divflux[ACT] / params.opti_divfluxobs_std) ** 2

    if tf.math.is_nan(COST_D):
        COST_D = tf.Variable(0.0)
        divflux = tf.where(tf.math.is_nan(divflux), 0.0, divflux)
    return COST_D

def misfit_usurf(params,state):

    ACT = state.icemaskobs > 0.5

    return 0.5 * tf.reduce_mean(
        (
            (state.usurf[ACT] - state.usurfobs[ACT])
            / params.opti_usurfobs_std
        )
        ** 2
    )

def cost_vol(params,state):

    ACT = state.icemaskobs > 0.5
    
    ModVols = tf.experimental.numpy.copy(state.icemaskobs)
    
    ModVols = tf.where(ModVols>0.5,(tf.reduce_sum(tf.where(state.icemask>0.5,state.thk,0.0))*state.dx**2)/1e9,ModVols)
    
    
    cost = 0.5 * tf.reduce_mean(
            ( (state.volumes[ACT] - ModVols[ACT]) / state.volume_weights[ACT]  )** 2
    )
    #print(cost)
    
    # Modvols = tf.reduce_mean(ModVols[ACT])
    # volume_target = tf.reduce_mean(state.volumes[ACT])
    
    # tolerance =0.05
    # if abs(Modvols-volume_target)/volume_target <tolerance:
    #     cost = tf.Variable(0.0)
    # else :
    #     cost = tf.Variable(10.0**10)
    return cost

def regu_thk(params,state):
    from scipy.ndimage import binary_dilation
    areaicemask = tf.reduce_sum(tf.where(state.icemask>0.5,1.0,0.0))*state.dx**2

    # here we had factor 8*np.pi*0.04, which is equal to 1
    if params.opti_infer_params:
        # gamma = tf.zeros_like(state.thk)
        # gamma = state.convexity_weights * areaicemask**(params.opti_convexity_power-2.0)
        state.thk = tf.where(tf.math.is_nan(state.thk), 0, state.thk)
        # gamma = tf.where(tf.math.is_nan(gamma), 0, gamma)
        gamma = params.opti_convexity_weight * areaicemask**(params.opti_convexity_power-2.0)
    else:
        gamma = params.opti_convexity_weight * areaicemask**(params.opti_convexity_power-2.0)
        state.thk = tf.where(tf.math.is_nan(state.thk), 0, state.thk)


    state.topg = state.usurf - state.thk
    if params.opti_to_regularize == 'topg':
        field = state.usurf - state.thk
        # print(params.opti_to_regularize)
        # field = state.usurf - state.thk
    elif params.opti_to_regularize == 'thk':
        field = state.thk
        
     # Mask for missing values
    # fieldmask = tf.equal(field, -9999)
    # Shift values left (fill with last valid value)
    # def fill_missing_values(tensor):
    #     for i in range(tensor.shape[1] - 1):  # Iterate over columns
    #         tensor = tf.where(fieldmask, tf.roll(tensor, shift=-1, axis=1), tensor)
    #     return tensor
    
    # field = fill_missing_values(field)   
        # print(params.opti_to_regularize)
    # fig, axs = plt.subplots(1, 1, figsize=(8,16))
    # plt.imshow(state.flowdirx, cmap='jet',origin='lower')
    # axs.axis("equal")
    # plt.show()
     
    if params.opti_infer_params: #I do not understand the goal of this separation between infer_param and not bc the code is similar and the second case may work also with infer_param
        ACT = state.anisotropy_RD != 1.0
        ACT2 = state.anisotropy_RD == 1.0
        
        dbdx = (field[:, 1:] - field[:, :-1])/state.dx
        dbdx = (dbdx[1:, :] + dbdx[:-1, :]) / 2.0
        dbdy = (field[1:, :] - field[:-1, :])/state.dx
        dbdy = (dbdy[:, 1:] + dbdy[:, :-1]) / 2.0

        if params.regu_sole_mask:
            MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
            MASK = binary_dilation(MASK, iterations=5)
            dbdx = tf.where( MASK, dbdx, 0.0)
            dbdy = tf.where( MASK, dbdy, 0.0)

        
        CostRegH = (params.opti_regu_param_thk) * (tf.reduce_mean(
             (1.0/tf.math.sqrt(state.anisotropy_factor))
             * tf.nn.l2_loss((dbdx[ACT] * state.flowdirx[ACT] + dbdy[ACT] * state.flowdiry[ACT])))
             + tf.reduce_mean(tf.math.sqrt(state.anisotropy_factor)
             * tf.nn.l2_loss((dbdx[ACT] * state.flowdiry[ACT] - dbdy[ACT] * state.flowdirx[ACT])))
             + tf.nn.l2_loss(dbdx[ACT2]) + tf.nn.l2_loss(dbdy[ACT2])
             - gamma * tf.math.reduce_sum(state.thk)
        )
        REGU_H = CostRegH
    else:
        if params.opti_smooth_anisotropy_factor == 1:
            dbdx = (field[:, 1:] - field[:, :-1])/state.dx
            dbdy = (field[1:, :] - field[:-1, :])/state.dx
            dbdx = (dbdx[1:, :] + dbdx[:-1, :]) / 2.0
            dbdy = (dbdy[:, 1:] + dbdy[:, :-1]) / 2.0
            
            if params.regu_sole_mask:
                MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
                MASK = binary_dilation(MASK, iterations=5)
                dbdx = tf.where( MASK, dbdx, 0.0)
                dbdy = tf.where( MASK, dbdy, 0.0)
            #remove nan
            dbdx = tf.boolean_mask(dbdx, tf.math.is_finite(dbdx))
            dbdy = tf.boolean_mask(dbdy, tf.math.is_finite(dbdy))
            REGU_H = (params.opti_regu_param_thk) * (
                tf.nn.l2_loss(dbdx) + tf.nn.l2_loss(dbdy)
                - gamma * tf.math.reduce_sum(state.thk)
            )
            state.parallel_part = state.thk*0
            state.perpendicular_part =  state.thk*0
            
        else:
            dbdx = (field[:, 1:] - field[:, :-1])/state.dx
            dbdx = (dbdx[1:, :] + dbdx[:-1, :]) / 2.0
            dbdy = (field[1:, :] - field[:-1, :])/state.dx
            dbdy = (dbdy[:, 1:] + dbdy[:, :-1]) / 2.0
    
            if params.regu_sole_mask:
                MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
                MASK = binary_dilation(MASK, iterations=5)
                dbdx = tf.where( MASK, dbdx, 0.0)
                dbdy = tf.where( MASK, dbdy, 0.0)
                
                
                
            REGU_H = (params.opti_regu_param_thk) * (
                (1.0/np.sqrt(params.opti_smooth_anisotropy_factor))
                * tf.nn.l2_loss((dbdx * state.flowdirx + dbdy * state.flowdiry))
                + np.sqrt(params.opti_smooth_anisotropy_factor)
                * tf.nn.l2_loss((dbdx * state.flowdiry - dbdy * state.flowdirx))
                - gamma * tf.math.reduce_sum(state.thk)
            ) 
    
           #  #Prepare quantity to integrate and remove nan from it due to the filter on field above.
           #  parallel_part = tf.boolean_mask(dbdx * state.flowdirx + dbdy * state.flowdiry, tf.math.is_finite(dbdx * state.flowdirx + dbdy * state.flowdiry))
           #  perpendicular_part =tf.boolean_mask(dbdx * state.flowdiry - dbdy * state.flowdirx,tf.math.is_finite(dbdx * state.flowdiry - dbdy * state.flowdirx))
           #  #record them in state to naalyse them in postprocessing
           #  state.parallel_part = tf.pad(
           #      dbdx * state.flowdirx + dbdy * state.flowdiry,
           #      paddings=[[0, 1], [0, 1]],  # Add 1 row at the bottom and 1 column on the right
           #      mode='CONSTANT',  # Use constant padding
           #      constant_values=0  # Pad with zeros
           #  )
           #  state.perpendicular_part =  tf.pad(
           #      dbdx * state.flowdiry - dbdy * state.flowdirx,
           #      paddings=[[0, 1], [0, 1]],  # Add 1 row at the bottom and 1 column on the right
           #      mode='CONSTANT',  # Use constant padding
           #      constant_values=0  # Pad with zeros
           #  )
            
           #  REGU_H = (params.opti_regu_param_thk) * (
           #     tf.nn.l2_loss((parallel_part))
           #     + (params.opti_smooth_anisotropy_factor)
           #     * tf.nn.l2_loss((perpendicular_part))
           #     - gamma * tf.math.reduce_sum(state.thk)
           # )
            

    return REGU_H

def regu_slidingco(params,state):

#    if not hasattr(state, "flowdirx"):
    dadx = (state.slidingco[:, 1:] - state.slidingco[:, :-1])/state.dx
    dady = (state.slidingco[1:, :] - state.slidingco[:-1, :])/state.dx

    if params.sole_mask:                
        dadx = tf.where( (state.icemaskobs[:, 1:] == 1) & (state.icemaskobs[:, :-1] == 1) , dadx, 0.0)
        dady = tf.where( (state.icemaskobs[1:, :] == 1) & (state.icemaskobs[:-1, :] == 1) , dady, 0.0)
        
    if params.opti_infer_params:
        dadx = tf.where( (state.anisotropy_factor[:, 1:] == 1.0) & (state.anisotropy_factor[:, :-1] == 1.0) , 0.0, dadx)
        dady = tf.where( (state.anisotropy_factor[1:, :] == 1.0) & (state.anisotropy_factor[:-1, :] == 1.0) , 0.0, dady)

    REGU_S = (params.opti_regu_param_slidingco) * (
        tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
    )
    + 10**10 * tf.math.reduce_mean( tf.where(state.slidingco >= 0, 0.0, state.slidingco**2) ) 
    # this last line serve to enforce non-negative slidingco

#               else:
        # dadx = (state.slidingco[:, 1:] - state.slidingco[:, :-1])/state.dx
        # dadx = (dadx[1:, :] + dadx[:-1, :]) / 2.0
        # dady = (state.slidingco[1:, :] - state.slidingco[:-1, :])/state.dx
        # dady = (dady[:, 1:] + dady[:, :-1]) / 2.0

        # if params.sole_mask:
        #     MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
        #     dadx = tf.where( MASK, dadx, 0.0)
        #     dady = tf.where( MASK, dady, 0.0)

        # REGU_S = (params.opti_regu_param_slidingco) * (
        #     (1.0/np.sqrt(params.opti_smooth_anisotropy_factor))
        #     * tf.nn.l2_loss((dadx * state.flowdirx + dady * state.flowdiry))
        #     + np.sqrt(params.opti_smooth_anisotropy_factor)
        #     * tf.nn.l2_loss((dadx * state.flowdiry - dady * state.flowdirx))

    return REGU_S

def regu_arrhenius(params,state):

#    if not hasattr(state, "flowdirx"):
    dadx = (state.arrhenius[:, 1:] - state.arrhenius[:, :-1])/state.dx
    dady = (state.arrhenius[1:, :] - state.arrhenius[:-1, :])/state.dx

    if params.sole_mask:                
        dadx = tf.where( (state.icemaskobs[:, 1:] == 1) & (state.icemaskobs[:, :-1] == 1) , dadx, 0.0)
        dady = tf.where( (state.icemaskobs[1:, :] == 1) & (state.icemaskobs[:-1, :] == 1) , dady, 0.0)
        
    if params.opti_infer_params:
        dadx = tf.where( (state.anisotropy_factor[:, 1:] == 1.0) & (state.anisotropy_factor[:, :-1] == 1.0) , 0.0, dadx)
        dady = tf.where( (state.anisotropy_factor[1:, :] == 1.0) & (state.anisotropy_factor[:-1, :] == 1.0) , 0.0, dady)

    REGU_A = (params.opti_regu_param_arrhenius) * (
        tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
    )
    + 10**10 * tf.math.reduce_mean( tf.where(state.arrhenius >= 0, 0.0, state.arrhenius**2) ) 
    # this last line serve to enforce non-negative slidingco
    return REGU_A
def smooth_thk(params,state,eroded_icemask,maskvel):
    """this function compute the cost related the laplacian of the field to attempt to reduce numerical artefact"""
    
    #choose a mask where to compute it
    if params.opti_mask_laplacian == True and params.opti_maskvel_laplacian == False:
        choosenmask = eroded_icemask
    elif params.opti_maskvel_laplacian == True and params.opti_mask_laplacian == False:
        choosenmask = maskvel
    elif params.opti_maskvel_laplacian == True and params.opti_mask_laplacian == True:
         choosenmask = maskvel*eroded_icemask
    else :
        choosenmask = 1
    
    # Compute gradients along x and y using finite differences
   
    dy = (state.thk[1:, :] - state.thk[:-1, :]) / state.dx # y-gradient
    dx = (state.thk[:, 1:] - state.thk[:, :-1]) / state.dx  # x-gradient
    
    # Pad to maintain the same shape as input
    dy = tf.pad(dy, [[0, 1], [0, 0]])
    dx = tf.pad(dx, [[0, 0], [0, 1]])
    
    # Compute second derivatives
    d2y = (dy[1:, :] - dy[:-1, :]) / state.dx
    d2x = (dx[:, 1:] - dx[:, :-1]) / state.dx
    
    # Pad again to maintain shape
    d2y = tf.pad(d2y, [[0, 1], [0, 0]])
    d2x = tf.pad(d2x, [[0, 0], [0, 1]])
    
    # Compute Laplacian
    laplacian = d2x + d2y

  
    laplacian *=choosenmask
    state.laplacian = laplacian
    
    
    # laplacian = tf.where(state.icemask >0.5, laplacian, 0)
    cost = params.opti_regu_laplacian*tf.nn.l2_loss(laplacian)
  
    dy = (state.usurf[1:, :] - state.usurf[:-1, :]) / state.dx # y-gradient
    dx = (state.usurf[:, 1:] - state.usurf[:, :-1]) / state.dx  # x-gradient
    
    # Pad to maintain the same shape as input
    dy = tf.pad(dy, [[0, 1], [0, 0]])
    dx = tf.pad(dx, [[0, 0], [0, 1]])
    
    # Compute second derivatives
    d2y = (dy[1:, :] - dy[:-1, :]) / state.dx
    d2x = (dx[:, 1:] - dx[:, :-1]) / state.dx
    
    # Pad again to maintain shape
    d2y = tf.pad(d2y, [[0, 1], [0, 0]])
    d2x = tf.pad(d2x, [[0, 0], [0, 1]])
    
    # Compute Laplacian
    laplacian = d2x + d2y

    
    laplacian *=choosenmask
    
    cost += params.opti_regu_laplacian*tf.nn.l2_loss(laplacian)
    
  
    dy = (state.uvelsurf[1:, :] - state.uvelsurf[:-1, :]) / state.dx # y-gradient
    dx = (state.uvelsurf[:, 1:] - state.uvelsurf[:, :-1]) / state.dx  # x-gradient
    
    # Pad to maintain the same shape as input
    dy = tf.pad(dy, [[0, 1], [0, 0]])
    dx = tf.pad(dx, [[0, 0], [0, 1]])
    
    # Compute second derivatives
    d2y = (dy[1:, :] - dy[:-1, :]) / state.dx
    d2x = (dx[:, 1:] - dx[:, :-1]) / state.dx
    
    # Pad again to maintain shape
    d2y = tf.pad(d2y, [[0, 1], [0, 0]])
    d2x = tf.pad(d2x, [[0, 0], [0, 1]])
    
    # Compute Laplacian
    laplacian = d2x + d2y

    
    laplacian *=choosenmask
    
    cost += params.opti_regu_laplacian*tf.nn.l2_loss(laplacian)
    
    dy = (state.vvelsurf[1:, :] - state.vvelsurf[:-1, :]) / state.dx # y-gradient
    dx = (state.vvelsurf[:, 1:] - state.vvelsurf[:, :-1]) / state.dx  # x-gradient
   
    # Pad to maintain the same shape as input
    dy = tf.pad(dy, [[0, 1], [0, 0]])
    dx = tf.pad(dx, [[0, 0], [0, 1]])
    
    # Compute second derivatives
    d2y = (dy[1:, :] - dy[:-1, :]) / state.dx
    d2x = (dx[:, 1:] - dx[:, :-1]) / state.dx
    
    # Pad again to maintain shape
    d2y = tf.pad(d2y, [[0, 1], [0, 0]])
    d2x = tf.pad(d2x, [[0, 0], [0, 1]])
    
    # Compute Laplacian
    laplacian = d2x + d2y
    
    
    laplacian *=choosenmask
    
    cost += params.opti_regu_laplacian*tf.nn.l2_loss(laplacian)
    
    return cost
##################################

def print_costs(params, state, cost, i):

    vol = ( np.sum(state.thk) * (state.dx**2) / 10**9 ).numpy()
    # mean_slidingco = tf.math.reduce_mean(state.slidingco[state.icemaskobs > 0.5])

    f = open('costs.dat','a')

    def bound(x):
        return min(x, 9999999)

    keys = list(cost.keys()) 
    if i == 0:
        L = [f"{key:>8}" for key in ["it","vol (km³)"]] + [f"{key:>12}" for key in keys]
        print("Costs:     " + "   ".join(L))
        print("   ".join([f"{key:>12}" for key in keys]),file=f)
        os.system("echo rm costs.dat >> clean.sh")

    if i % params.opti_output_freq == 0:
        L = [datetime.datetime.now().strftime("%H:%M:%S"),f"{i:0>{8}}",f"{vol:>8.4f}"] \
          + [f"{bound(cost[key].numpy()):>12.4f}" for key in keys]
        print("   ".join(L))

    print("   ".join([f"{bound(cost[key].numpy()):>12.4f}" for key in keys]),file=f)
def save_costs(params, state):

    np.savetxt(
        "costs.dat",
        np.stack(state.costs),
        fmt="%.10f",
        header="        COST_U        COST_H      COST_D       COST_S       REGU_H       REGU_S          H>0    COST_VOL        Outlines          COST_TOTAL ",
    )


    os.system(
        "echo rm " + "costs.dat" + " >> clean.sh"
    )
    
def save_rms_std(params, state):

    np.savetxt(
        "rms_std.dat",
        np.stack(
            [
                state.rmsthk,
                state.stdthk,
                state.rmsvel,
                state.stdvel,
                state.rmsdiv,
                state.stddiv,
                state.rmsusurf,
                state.stdusurf,
            ],
            axis=-1,
        ),
        fmt="%.10f",
        header="        rmsthk      stdthk       rmsvel       stdvel       rmsdiv       stddiv       rmsusurf       stdusurf",
    )

    os.system(
        "echo rm " + "rms_std.dat" + " >> clean.sh"
    )

def create_density_matrix(data, kernel_size):
    # Convert data to binary mask (1 for valid data, 0 for NaN)
    binary_mask = tf.where(tf.math.is_nan(data), tf.zeros_like(data), tf.ones_like(data))

    # Create a kernel for convolution (all ones)
    kernel = tf.ones((kernel_size, kernel_size, 1, 1), dtype=binary_mask.dtype)

    # Apply convolution to count valid data points in the neighborhood
    density = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(binary_mask, 0), -1), 
                           kernel, strides=[1, 1, 1, 1], padding='SAME')

    # Remove the extra dimensions added for convolution
    density = tf.squeeze(density)

    return density

def _compute_rms_std_optimization(state, i):
    I = state.icemaskobs > 0.5

    if i == 0:
        state.rmsthk = []
        state.stdthk = []
        state.rmsvel = []
        state.stdvel = []
        state.rmsusurf = []
        state.stdusurf = []
        state.rmsdiv = []
        state.stddiv = []

    if hasattr(state, "thkobs"):
        ACT = ~tf.math.is_nan(state.thkobs)
        if np.sum(ACT) == 0:
            state.rmsthk.append(0)
            state.stdthk.append(0)
        else:
            state.rmsthk.append(np.nanmean(state.thk[ACT] - state.thkobs[ACT]))
            state.stdthk.append(np.nanstd(state.thk[ACT] - state.thkobs[ACT]))

    else:
        state.rmsthk.append(0)
        state.stdthk.append(0)

    if hasattr(state, "uvelsurfobs"):
        velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
        ACT = ~np.isnan(velsurfobs_mag)

        state.rmsvel.append(
            np.mean(velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()])
        )
        state.stdvel.append(
            np.std(velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()])
        )
        if np.isnan(state.rmsvel[-1]):
            state.rmsvel[-1] = 0
        if np.isnan(state.stdvel[-1]):
            state.stdvel[-1] = 0
    else:
        state.rmsvel.append(0)
        state.stdvel.append(0)

    if hasattr(state, "divfluxobs"):
        state.rmsdiv.append(np.mean(state.divfluxobs[I] - state.divflux[I]))
        state.stddiv.append(np.std(state.divfluxobs[I] - state.divflux[I]))
    else:
        state.rmsdiv.append(0)
        state.stddiv.append(0)

    if hasattr(state, "usurfobs"):
        state.rmsusurf.append(np.mean(state.usurf[I] - state.usurfobs[I]))
        state.stdusurf.append(np.std(state.usurf[I] - state.usurfobs[I]))
    else:
        state.rmsusurf.append(0)
        state.stdusurf.append(0)


def _update_ncdf_optimize(params, state, it):
    """
    Initialize and write the ncdf optimze file
    """

    if hasattr(state, "logger"):
        state.logger.info("Initialize  and write NCDF output Files")
        
    if "velbase_mag" in params.opti_vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    if "velsurf_mag" in params.opti_vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velsurfobs_mag" in params.opti_vars_to_save:
        state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    
    if "sliding_ratio" in params.opti_vars_to_save:
        state.sliding_ratio = tf.where(state.velsurf_mag > 10, state.velbase_mag / state.velsurf_mag, np.nan)
    
    if "velsurf_mag_noslid" in params.opti_vars_to_save:
        state.velsurf_mag_noslid = getmag(state.uvelsurf - state.uvelbase, state.vvelsurf -  state.vvelbase)
    # if "topg" in params.opti_vars_to_save:
    #     state.topg = state.usurf - state.thk
        
    if it == 0:
        nc = Dataset(
            "optimize.nc",
            "w",
            format="NETCDF4",
        )

        nc.createDimension("iterations", None)
        E = nc.createVariable("iterations", np.dtype("float32").char, ("iterations",))
        E.units = "None"
        E.long_name = "iterations"
        E.axis = "ITERATIONS"
        E[0] = it

        nc.createDimension("y", len(state.y))
        E = nc.createVariable("y", np.dtype("float32").char, ("y",))
        E.units = "m"
        E.long_name = "y"
        E.axis = "Y"
        E[:] = state.y.numpy()

        nc.createDimension("x", len(state.x))
        E = nc.createVariable("x", np.dtype("float32").char, ("x",))
        E.units = "m"
        E.long_name = "x"
        E.axis = "X"
        E[:] = state.x.numpy()

        for var in params.opti_vars_to_save:
            E = nc.createVariable(
                var, np.dtype("float32").char, ("iterations", "y", "x")
            )
            E[0, :, :] = vars(state)[var].numpy()

        nc.close()

        os.system( "echo rm " + "optimize.nc" + " >> clean.sh" )

    else:
        nc = Dataset("optimize.nc", "a", format="NETCDF4", )

        d = nc.variables["iterations"][:].shape[0]

        nc.variables["iterations"][d] = it

        for var in params.opti_vars_to_save:
            nc.variables[var][d, :, :] = vars(state)[var].numpy()

        nc.close()


def _output_ncdf_optimize_final(params, state):
    """
    Write final geology after optimizing
    """
    if params.opti_save_iterat_in_ncdf==False:
        if "velbase_mag" in params.opti_vars_to_save:
            state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

        if "velsurf_mag" in params.opti_vars_to_save:
            state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

        if "velsurfobs_mag" in params.opti_vars_to_save:
            state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
        
        if "sliding_ratio" in params.opti_vars_to_save:
            state.sliding_ratio = tf.where(state.velsurf_mag > 10, state.velbase_mag / state.velsurf_mag, np.nan)

    if params.opti_postprocess:
        nc = Dataset(
            params.lncd_input_file[:-3]+"_optimised.nc",
            "w",
            format="NETCDF4",
        )
    else:
        nc = Dataset(
            params.oggm_RGI_ID+"_optimised.nc",
            "w",
            format="NETCDF4",
        )

    nc.createDimension("y", len(state.y))
    E = nc.createVariable("y", np.dtype("float32").char, ("y",))
    E.units = "m"
    E.long_name = "y"
    E.axis = "Y"
    E[:] = state.y.numpy()

    nc.createDimension("x", len(state.x))
    E = nc.createVariable("x", np.dtype("float32").char, ("x",))
    E.units = "m"
    E.long_name = "x"
    E.axis = "X"
    E[:] = state.x.numpy()

    for v in params.opti_vars_to_save:
        if hasattr(state, v):
            E = nc.createVariable(v, np.dtype("float32").char, ("y", "x"))
            E.standard_name = v
            E[:] = vars(state)[v]

    nc.close()

    if params.opti_postprocess:
        os.system(
            "echo rm "
            + params.lncd_input_file[:-3]+"_optimised.nc"
            + " >> clean.sh"
        )
    else:
        os.system(
            "echo rm "
            + params.oggm_RGI_ID+"_optimised.nc"
            + " >> clean.sh"
        )


def _plot_cost_functions(params, state, costs):
    costs = np.stack(costs)

    for i in range(costs.shape[1]):
        costs[:, i] -= np.min(costs[:, i])
        costs[:, i] /= np.where(np.max(costs[:, i]) == 0, 1.0, np.max(costs[:, i]))

    fig = plt.figure(figsize=(10, 10))
    plt.plot(costs[:, 0], "-k", label="COST U")
    plt.plot(costs[:, 1], "-r", label="COST H")
    plt.plot(costs[:, 2], "-b", label="COST D")
    plt.plot(costs[:, 3], "-g", label="COST S")
    plt.plot(costs[:, 4], "--c", label="REGU H")
    plt.plot(costs[:, 5], "--m", label="REGU A")
    plt.ylim(0, 1)
    plt.legend()

    plt.savefig("convergence.png", pad_inches=0)
    plt.close("all")

    os.system(
        "echo rm "
        + "convergence.png"
        + " >> clean.sh"
    )


def _update_plot_inversion(params, state, i):
    """
    Plot thickness, velocity, mand slidingco"""

    if hasattr(state, "uvelsurfobs"):
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
    else:
        try:
            velsurfobs_mag = state.velsurfobs_mag
        except:
            velsurfobs_mag = np.zeros_like(state.thk.numpy())

    if hasattr(state, "usurfobs"):
        usurfobs = state.usurfobs
    else:
        usurfobs = np.zeros_like(state.thk.numpy())

    velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()
    velsurf_mag= tf.where(~tf.math.is_nan(velsurfobs_mag),velsurf_mag, tf.constant(np.nan))
    #########################################################

    if i == 0:
        if params.opti_editor_plot2d == "vs":
            plt.ion()  # enable interactive mode

        # state.fig = plt.figure()
        state.fig, state.axes = plt.subplots(2, 3,figsize=(10, 8))

        state.extent = [state.x[0], state.x[-1], state.y[0], state.y[-1]]

    #########################################################

    cmap = copy.copy(matplotlib.cm.jet)
    cmap.set_bad(color="white")

    ax1 = state.axes[0, 0]
    ACT = ~tf.math.is_nan(state.thkobs)
    im1 = ax1.imshow(
        np.ma.masked_where(state.thk == 0, state.thk),
        origin="lower",
        extent=state.extent,
        vmin=0,
        # vmax=np.quantile(state.thk, 0.98),
        vmax=tf.reduce_max(state.thkobs[ACT]),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, ax=ax1)
    ax1.set_title(
        "Ice thickness \n (RMS : "
        + str(int(state.rmsthk[-1]))
        + ", STD : "
        + str(int(state.stdthk[-1]))
        + ")",
        size=12,
    )
    ax1.axis("off")

    #########################################################

    ax2 = state.axes[0, 1]

    from matplotlib import colors

    im1 = ax2.imshow(
        state.slidingco,
        origin="lower",
#        norm=colors.LogNorm(),
        vmin=0.03,
        vmax=0.09,
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax2)
    ax2.set_title("Iteration " + str(i) + " \n Sliding coefficient", size=12)
    ax2.axis("off")

    ########################################################

    ax3 = state.axes[0, 2]

    im1 = ax3.imshow(
        state.usurf - usurfobs,
        origin="lower",
        extent=state.extent,
        vmin=-10,
        vmax=10,
        cmap="RdBu",
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax3)
    ax3.set_title(
        "Top surface adjustement \n (RMS : %5.1f , STD : %5.1f"
        % (state.rmsusurf[-1], state.stdusurf[-1])
        + ")",
        size=12,
    )
    ax3.axis("off")

    #########################################################

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(color="white")

    ax4 = state.axes[1, 0]

    # im1 = ax4.imshow(
    #     velsurf_mag, # np.ma.masked_where(state.thk == 0, velsurf_mag),
    #     origin="lower",
    #     extent=state.extent,
    #     norm=matplotlib.colors.LogNorm(vmin=1, vmax=200),
    #     cmap=cmap,
    # )
    im1 = ax4.imshow(
        velsurf_mag, # np.ma.masked_where(state.thk == 0, velsurf_mag),
        origin="lower",
        extent=state.extent,
        vmin=0, vmax=100,
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax4)
    ax4.set_title(
        "Modelled velocities \n (RMS : "
        + str(int(state.rmsvel[-1]))
        + ", STD : "
        + str(int(state.stdvel[-1]))
        + ")",
        size=12,
    )
    ax4.axis("off")

    ########################################################

    ax5 = state.axes[1, 1]
    # im1 = ax5.imshow(
    #     np.ma.masked_where(state.thk == 0, velsurfobs_mag),
    #     origin="lower",
    #     extent=state.extent,
    #     norm=matplotlib.colors.LogNorm(vmin=1, vmax=200),
    #     cmap=cmap,
    # )
    im1 = ax5.imshow(
        np.ma.masked_where(state.thk == 0, velsurfobs_mag),
        origin="lower",
        extent=state.extent,
        vmin=1, vmax=100,
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax5)
    ax5.set_title("Target \n Observed velocities", size=12)
    ax5.axis("off")

    #######################################################

    ax6 = state.axes[1, 2]
    im1 = ax6.imshow(
        state.divflux, # np.where(state.icemaskobs > 0.5, state.divflux,np.nan),
        origin="lower",
        extent=state.extent,
        vmin=-10,
        vmax=10,
        cmap="RdBu",
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax6)
    ax6.set_title(
        "Flux divergence \n (RMS : %5.1f , STD : %5.1f"
        % (state.rmsdiv[-1], state.stddiv[-1])
        + ")",
        size=12,
    )
    ax6.axis("off")

    #########################################################

    if params.opti_plot2d_live:
        if params.opti_editor_plot2d == "vs":
            state.fig.canvas.draw()  # re-drawing the figure
            state.fig.canvas.flush_events()  # to flush the GUI events
        else:
            from IPython.display import display, clear_output

            clear_output(wait=True)
            display(state.fig)
    else:
        plt.savefig("resu-opti-" + str(i).zfill(4) + ".png", bbox_inches="tight", pad_inches=0.2)

        os.system( "echo rm " + "*.png" + " >> clean.sh" )


def _update_plot_inversion_simple(params, state, i):
    """
    Plot thickness, velocity, mand slidingco"""

    if hasattr(state, "uvelsurfobs"):
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
    else:
        velsurfobs_mag = np.zeros_like(state.thk.numpy())

    if hasattr(state, "usurfobs"):
        usurfobs = state.usurfobs
    else:
        usurfobs = np.zeros_like(state.thk.numpy())

    velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()

    #########################################################

    if i == 0:
        if params.opti_editor_plot2d == "vs":
            plt.ion()  # enable interactive mode

        # state.fig = plt.figure()
        state.fig, state.axes = plt.subplots(1, 2)

        state.extent = [state.x[0], state.x[-1], state.y[0], state.y[-1]]

    #########################################################

    cmap = copy.copy(matplotlib.cm.jet)
    cmap.set_bad(color="white")

    ax1 = state.axes[0]
    ACT = ~tf.math.is_nan(state.thkobs)
    im1 = ax1.imshow(
        np.ma.masked_where(state.thk == 0, state.thk),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=tf.reduce_max(state.thkobs[ACT]),
        # vmax=np.quantile(state.thk, 0.98),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, ax=ax1)
    ax1.set_title(
        "Ice thickness \n (RMS : "
        + str(int(state.rmsthk[-1]))
        + ", STD : "
        + str(int(state.stdthk[-1]))
        + ")",
        size=16,
    )
    ax1.axis("off")

    #########################################################

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(color="white")

    ax4 = state.axes[1]

    im1 = ax4.imshow(
        np.ma.masked_where(state.thk == 0, velsurf_mag),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.nanmax(velsurfobs_mag),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax4)
    ax4.set_title(
        "Modelled velocities \n (RMS : "
        + str(int(state.rmsvel[-1]))
        + ", STD : "
        + str(int(state.stdvel[-1]))
        + ")",
        size=16,
    )
    ax4.axis("off")

    #########################################################

    if params.opti_plot2d_live:
        if params.opti_editor_plot2d == "vs":
            state.fig.canvas.draw()  # re-drawing the figure
            state.fig.canvas.flush_events()  # to flush the GUI events
        else:
            from IPython.display import display, clear_output

            clear_output(wait=True)
            display(state.fig)
    else:
        plt.savefig(
            "resu-opti-" + str(i).zfill(4) + ".png",
            pad_inches=0,
        )
        plt.close("all")

        os.system(
            "echo rm " + "*.png" + " >> clean.sh"
        )


def _compute_flow_direction_for_anisotropic_smoothing(state, params):
    uvelsurf = tf.where(tf.math.is_nan(state.uvelsurf), 0.0, state.uvelsurf)
    vvelsurf = tf.where(tf.math.is_nan(state.vvelsurf), 0.0, state.vvelsurf)
    if params.opti_flowdir_mask == True:
        uvelsurf /= getmag(uvelsurf,vvelsurf)
        vvelsurf /= getmag(uvelsurf,vvelsurf)
        uvelsurf*=state.icemask
        vvelsurf*=state.icemask
    state.flowdirx = (
        uvelsurf[1:, 1:] + uvelsurf[:-1, 1:] + uvelsurf[1:, :-1] + uvelsurf[:-1, :-1]
    ) / 4.0
    state.flowdiry = (
        vvelsurf[1:, 1:] + vvelsurf[:-1, 1:] + vvelsurf[1:, :-1] + vvelsurf[:-1, :-1]
    ) / 4.0
    
    state.anisotropy_RD = (
        state.anisotropy_factor[1:, 1:] + state.anisotropy_factor[:-1, 1:] + state.anisotropy_factor[1:, :-1] + state.anisotropy_factor[:-1, :-1]
    ) / 4.0
    state.anisotropy_RD = tf.where(state.anisotropy_RD > 0.5, 1.0, params.opti_smooth_anisotropy_factor)

    from scipy.ndimage import gaussian_filter

    state.flowdirx = gaussian_filter(state.flowdirx, 3, mode="constant")
    state.flowdiry = gaussian_filter(state.flowdiry, 3, mode="constant")
    
    # Same as gaussian filter above but for tensorflow is (NOT TESTED)
    # import tensorflow_addons as tfa
    # state.flowdirx = ( tfa.image.gaussian_filter2d( state.flowdirx , sigma=3, filter_shape=100, padding="CONSTANT") )

    if params.opti_flowdir_mask == True:
        pass
    else:
        state.flowdirx /= getmag(state.flowdirx, state.flowdiry)
        state.flowdiry /= getmag(state.flowdirx, state.flowdiry)
    state.flowdirx = tf.where(tf.math.is_nan(state.flowdirx), 0.0, state.flowdirx)
    state.flowdiry = tf.where(tf.math.is_nan(state.flowdiry), 0.0, state.flowdiry)
    
    if params.opti_infer_params:
       state.flowdirx = tf.where(state.anisotropy_RD==1.0, 0.0, state.flowdirx)
       state.flowdiry = tf.where(state.anisotropy_RD==1.0, 0.0, state.flowdiry)
    # state.flowdirx = tf.zeros_like(state.flowdirx)
    # state.flowdiry = tf.ones_like(state.flowdiry)

    # this is to plot the observed flow directions
    # fig, axs = plt.subplots(1, 1, figsize=(8,16))
    # plt.quiver(state.flowdirx,state.flowdiry)
    # axs.axis("equal")
    
def skip_infer_params(state, params):
    import pandas as pd
    #This function load all the usefull variable (volume, volume_weight, area, anysotropy_factor not related to the computation of the slidingco
    #in infer_param and use either a scalar value or a field on a nc file as input for slidingco.
    state.anisotropy_factor = tf.experimental.numpy.copy(state.icemaskobs)
    state.anisotropy_factor = tf.where(state.icemask > 0.5, params.opti_smooth_anisotropy_factor, params.opti_smooth_anisotropy_factor)
    TotalArea = tf.reduce_sum(tf.where(state.icemask > 0.5,1.0,0.0))*state.dx**2
    TotalArea = TotalArea/1e6 # in km**2
  
   
    #print(TotalArea/1e6)
    
    # state.convexity_weights = tf.experimental.numpy.copy(state.icemaskobs)
    state.volumes = tf.experimental.numpy.copy(state.icemaskobs)
    state.volume_weights = tf.experimental.numpy.copy(state.icemaskobs)
    state.volume_weights = tf.where(state.icemaskobs > 0, params.opti_vol_std, 0)
    

    #Get volume predictor
    if params.opti_vol_factor_path is not None:
        if params.opti_icecaps == True:
            print(os.getcwd())
            print(os.listdir("../data"))
            csv_vf = pd.read_csv(params.opti_vol_factor_path,index_col =0)
            vol_factor = csv_vf.max().values#params.opti_vol_factor
            if vol_factor > 1.5:
                vol_factor = 1.5
            elif vol_factor < 0.75:
                vol_factor = 0.75
            state.volumes = tf.where(state.icemaskobs > 0.5, 0.034*TotalArea**1.25*vol_factor, state.volumes)
            # state.volume_weights = tf.where(state.icemaskobs > 0.5, 1.0, state.volume_weights)
            print('Area (km^2) : ',np.round(TotalArea.numpy(),2),' and volume (km^3) : ',np.round(0.034*TotalArea.numpy()**1.25*vol_factor,2),' and vol_factor : ', str(vol_factor))
            state.vf = state.icemask*vol_factor
        else:
            csv_vf = pd.read_csv(params.opti_vol_factor_path,index_col =0)
            vol_factor = csv_vf.max().values#params.opti_vol_factor
            if vol_factor >1.5:
                vol_factor = 1.5
            elif vol_factor < 0.75:
                vol_factor = 0.75
            VolBasins =  0.034*TotalArea**1.375*vol_factor
            state.volumes = tf.where(state.icemaskobs > 0.5, VolBasins, state.volumes) #Make sure to put into km3!
            print('Area (km^2) : ',np.round(TotalArea.numpy(),2),'volume (km^3) : ',np.round(0.034*TotalArea.numpy()**1.375*vol_factor,2),' and vol_factor : ', str(vol_factor))
            state.vf = state.icemask*vol_factor
    else:
        pass
    if params.opti_load_sc =='False':
        SC1D = tf.reduce_mean(state.slidingco)
    elif params.opti_load_sc == 'scalar':
        if params.opti_modify_sc == True:
            sc = params.opti_sc_value + params.opti_sc_mod
            SC1D = tf.where(state.icemaskobs >=0, sc ,state.icemaskobs)
        else :
            sc = params.opti_sc_value
            SC1D = tf.where(state.icemaskobs >=0, sc ,state.icemaskobs)
        
    elif params.opti_load_sc == 'field':
        # nc = Dataset(
        #     os.path.join(params.opti_load_sc_path), "r", format="NETCDF4"
        # )
        # SC1D = tf.Variable(np.squeeze(nc.variables["slidingco"]).astype("float32"))
    
        nc = Dataset(
           os.path.join(params.lncd_input_file), "r", format="NETCDF4"
       )
        SC1D = tf.Variable(np.squeeze(nc.variables["infered_sc"]).astype("float32"))
    
    else :
        print('INVALID INPUT : opti_load_sc must be chosen amongst the 3 following values : False, scalar or field')
    
    #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)            
    maxslidingco = 0.1
    minslidingco = 0.01
    
    state.slidingco = tf.where(state.icemaskobs >= 0, SC1D.numpy(), state.slidingco)
    state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
    state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco) 
    
    if params.opti_altitude_mask == True:
        #Modify the uniform fields of slidingco and arrhenius to better match dynamics
        # Step 1: Create a boolean mask for inside the glacier outline
        inside_glacier = state.icemaskobs > 0.5
        # Step 2: Create a boolean mask for where velocity information is available
        has_velocity = tf.math.is_finite(state.uvelsurfobs)
        outside_velocity_mask = ~has_velocity
        # Step 3: Find the highest and lowest point of the velocity mask
        highest_velocity_point = tf.reduce_max(tf.where(has_velocity, state.usurfobs, tf.fill(tf.shape(state.usurfobs), -10.))) # -50
        lowest_velocity_point = tf.reduce_min(tf.where(has_velocity, state.usurfobs, tf.fill(tf.shape(state.usurfobs), 10000.)))# +50
        # Step 4: Create a mask for points inside the glacier, outside the velocity mask, and above the highest point/below lowest point
        above_highest_point = state.usurfobs > highest_velocity_point
        below_lowest_point = state.usurfobs < lowest_velocity_point
        
        high_mask = inside_glacier & outside_velocity_mask & above_highest_point
        low_mask = inside_glacier & outside_velocity_mask & below_lowest_point
        # Step 5: Replace the values in state.slidingco/arrhenius where the combined mask is True
        if params.opti_modify_sc == True:
            state.slidingco = tf.where(high_mask, SC1D + params.opti_sc_high, state.slidingco)
            state.slidingco = tf.where(low_mask,  SC1D + params.opti_sc_low, state.slidingco) 
        if params.opti_modify_arrhenius == True:
            state.arrhenius = tf.where(high_mask, tf.reduce_mean(state.arrhenius) + params.opti_arrhenius_high, state.arrhenius)
            state.arrhenius = tf.where(low_mask,  tf.reduce_mean(state.arrhenius) + params.opti_arrhenius_low, state.arrhenius) 
   
def _infer_params(state, params):
    #This function allows for both parameters to be specified as varying 2D fields (you could compute them pixel-wise from VelMag by swapping in VelMag for VelPerc).
    #This is probably not a good idea, because the values of both parameters do not depend solely on the velocity at that point. But feel free to try! If you do
    #want to do that, you'll also need to un-comment the code block for smoothing and then converting the smoothed weights back to tensors (you may also want to
    #visualise the figures!), and set up a state.convexity_weights field to act as the 2D array
    import scipy
   
    #Get list of G entities in each C/get multi-valued ice mask
    #Loop over all Gs to construct parameter rasters
    
    state.anisotropy_factor = tf.experimental.numpy.copy(state.icemaskobs)
    state.anisotropy_factor = tf.where(state.icemask > 0.5, params.opti_smooth_anisotropy_factor, params.opti_smooth_anisotropy_factor)
    # params.opti_divfluxobs_std = 1.0
    # params.opti_usurfobs_std = 0.3
    # params.opti_regu_param_thk = 1.0
    
    percthreshold = 99
    dynthreshold = 95
    NumGlaciers = int(tf.reduce_max(state.icemask).numpy())
    #print(NumGlaciers)
    
    TotalArea = tf.reduce_sum(tf.where(state.icemask > 0.5,1.0,0.0))*state.dx**2
    TotalArea = TotalArea/1e6
    TotalVolume = 0.0
    VolBasins = 0.0
    #print(TotalArea/1e6)
    
    # state.convexity_weights = tf.experimental.numpy.copy(state.icemaskobs)
    state.volumes = tf.experimental.numpy.copy(state.icemaskobs)
    state.volume_weights = tf.experimental.numpy.copy(state.icemaskobs)
    state.volume_weights = tf.where(state.icemaskobs > 0, params.opti_vol_std, 0.0)
    
    #Get some initial information
    VelMag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    VelMag = tf.where(tf.math.is_nan(VelMag),1e-6,VelMag)
    VelMag = tf.where(VelMag==0,1e-6,VelMag)

    AnnualTemp = tf.reduce_mean(state.air_temp, axis=0)
    # print('Annual Temp : ', AnnualTemp.numpy())
   
    
    #Start of G loop
    for i in range(1,NumGlaciers+1):
        
        #Get area predictor
        Area = tf.reduce_sum(tf.where(state.icemask==i,1.0,0.0))*state.dx**2
        Area = np.round(Area.numpy()/1000**2, decimals=1)
        #print('Area is: ', Area)
        #print('Predicted volume is: ', np.exp(np.log(Area)*1.359622487))
        #Work out nominal volume target based on volume-area scaling - only has much of an effect if no other observations
        if (NumGlaciers/TotalArea < 0.1) & (NumGlaciers > 4) & (TotalArea > 100): #if ice caps
            state.volumes = tf.where(state.icemaskobs > 0.5, 0.034*TotalArea**1.25*params.opti_vol_factor, state.volumes)
            #print(0.034*TotalArea**1.25)
            state.volume_weights = tf.where(state.icemaskobs > 0.5, 1.0, state.volume_weights)
            # params.opti_divfluxobs_std = 10.0
            # params.opti_usurfobs_std = 3.0
            # params.opti_regu_param_thk = 1.0
        else:
            VolBasins = VolBasins + 0.034*Area**1.375*params.opti_vol_factor
            state.volumes = tf.where(state.icemaskobs > 0.5, VolBasins, state.volumes) #Make sure to put into km3!

        # print('Area (km^2) : ',Area,' and volume (km^3) : ',np.round(0.034*Area**1.375,2))
        # print('Volume: ',Area,0.034*Area**1.375)
        if Area <= 0.0:
            continue
        
        #Get velocity predictors
        VelMean = np.round(np.mean(VelMag[state.icemaskobs==i]),decimals=2)
        # print("Mean velocity is: ", VelMean)
        VelPerc = np.round(np.percentile(VelMag[state.icemaskobs==i], percthreshold))
        VelThresh = np.percentile(VelMag[state.icemaskobs==i], dynthreshold)
        #print("Threshold velocity is: ", VelThresh)
        print('Velmean : ', VelMean)
        if VelMean == 0.0:
            #With volume-area scaling (on the assumption these will all be very small)
            state.slidingco = tf.where(state.icemaskobs == i, 0.1, state.slidingco)
            #state.volume_weights = tf.where(state.icemaskobs == i, 0.1, state.volume_weights)
            state.anisotropy_factor = tf.where(state.icemaskobs == i, 1.0, state.anisotropy_factor)
            state.uvelsurfobs = tf.where(state.icemaskobs == i, np.nan, state.uvelsurfobs)
            state.vvelsurfobs = tf.where(state.icemaskobs == i, np.nan, state.vvelsurfobs)
           
            #Record the infered slidingco
            print('Infered Slidingco : ',0.1)
           
            
              #record slidngco
            print('Recording Slidingco')
            var_info = ["Infered Sliding Coefficient using T and slope", "?"]
            nc = Dataset(
                os.path.join("data/slidingco.nc"), "w", format="NETCDF4"
            )
            nc_input = Dataset(os.path.join(params.oggm_RGI_ID, "gridded_data.nc"), "r+")
            x = np.squeeze(nc_input.variables["x"]).astype("float32")
            y = np.flip(np.squeeze(nc_input.variables["y"]).astype("float32"))
            
            nc.createDimension("y", len(y))
            yn = nc.createVariable("y", np.dtype("float32").char, ("y",))
            yn.units = "m"
            yn.long_name = "y"
            yn.standard_name = "y"
            yn.axis = "Y"
            yn[:] = y
            
            nc.createDimension("x", len(x))
            xn = nc.createVariable("x", np.dtype("float32").char, ("x",))
            xn.units = "m"
            xn.long_name = "x"
            xn.standard_name = "x"
            xn.axis = "X"
            xn[:] = x
            
            E = nc.createVariable("slidingco", np.dtype("float32").char, ("y", "x"))
            E.long_name = var_info[0]
            E.units = var_info[1]
            E.standard_name = "slidingco"
            E[:] = state.slidingco
            
            nc.close()
           
            continue
        if VelThresh < 10.0:
            state.anisotropy_factor = tf.where(state.icemaskobs == i, 1.0, state.anisotropy_factor)
            
        #Get average annual air temperature across whole domain
        AvgTemp = tf.reduce_mean(AnnualTemp[state.icemaskobs==i]).numpy()
        AvgTemp = np.round(abs(AvgTemp), decimals=1)
        print('Average Temp and NumGlaciers : -',AvgTemp,'degree celsius', i,'/', NumGlaciers)
        
        #Get thickness predictors (this uses Millan et al. (2022) thickness estimates just to give some idea of the expected ice thicknesses)
        MaxThk = tf.math.reduce_max(state.thkinit[state.icemask==i])
        MeanThk = tf.math.reduce_mean(state.thkinit[state.icemask==i])
        MaxThk = tf.math.round(MaxThk)
        MeanThk = tf.math.round(MeanThk)
        #print('Max and mean thickness are: ',MaxThk, MeanThk)
        
        #Get slope field
       
        AvgSlope = np.round(tf.reduce_max(state.slopes[state.icemaskobs==i]).numpy(), decimals=1)
        print("Average Slope is: ", AvgSlope)
        
        Tidewater = params.opti_tidewater_glacier
        
        #Do regressions
        if hasattr(state, "tidewatermask"):
            if tf.reduce_max(state.tidewatermask[state.icemask == i]).numpy() == 1:
                Tidewater = True
            
        if Tidewater == True:            
            VelMean1DSC = tf.math.log(VelMean)*0.779646843
            #Slope1DSC = tf.math.log(AvgSlope)*-0.075281289
            
            LogMeanThkSC = tf.math.log(MeanThk)*2.113449074
            LogTempSC = np.log(AvgTemp)*-0.305778358
            LogMaxThkSC = tf.math.log(MaxThk)*-1.45777084
            LogAreaSC = np.log(Area)*0.220126846
            LogMaxVelSC = np.log(VelPerc)*-1.003939161
            #LogMeanVelSC = tf.math.log(VelMag)*0.779646843
            #LogMeanVelSC = tf.where(tf.math.is_nan(LogMeanVelSC),0,LogMeanVelSC)
                    
            #These regressions are purely empirical and are based on a selection of 37 glaciers from around the world with thickness measurements,
            #where IGM inversions were performed and the best parameters chosen
            SC1D = tf.math.exp(LogMeanThkSC + LogTempSC + LogMaxThkSC + LogAreaSC + LogMaxVelSC + VelMean1DSC - 3.157484542)
            #print(CW1D.numpy(),SC1D.numpy())
            
            #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)
            maxslidingco = 0.1
            minslidingco = 0.01
            print('Infered Slidingco : ',np.mean(SC1D.numpy()))
            state.slidingco = tf.where(state.icemaskobs == i, SC1D.numpy(), state.slidingco)
            state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
            state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)      
        else:
            #Set up various constants            
            VelMean1DSC = VelMean*-0.002076554
            Slope1DSC = AvgSlope*0.001521555
            
            #MeanVelSC = VelMag*-0.002076554
            #SlopeSC = Slopes*0.001521555
            AreaSC = Area*-0.00003986
            MaxThkSC = MaxThk*0.0000186659
            #MeanThkSC = MeanThk*0.000193306
            MaxVelSC = VelPerc*0.000330179
            #TempSC = AvgTemp*-0.000686353 
                    
            #These regressions are purely empirical and are based on a selection of 37 glaciers from around the world with thickness measurements,
            #where IGM inversions were performed and the best parameters chosen
            
            #This fills the area outside the ice mask with the correct average inferred parameter values (so the smoothing works properly)
            SC1D = VelMean1DSC + Slope1DSC + AreaSC + MaxThkSC + MaxVelSC + 0.018228361
            print('Infered Slidingco : ',np.mean(SC1D.numpy()))
            
            #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)            
            maxslidingco = 0.1
            minslidingco = 0.01
            
            state.slidingco = tf.where(state.icemaskobs == i, SC1D.numpy(), state.slidingco)
            state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
            state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)  
        
          #record slidngco
    print('Recording Slidingco')
    var_info = ["Infered Sliding Coefficient using T and slope", "?"]
    nc = Dataset(
        os.path.join("data/slidingco.nc"), "w", format="NETCDF4"
    )
    nc_input = Dataset(os.path.join(params.oggm_RGI_ID, "gridded_data.nc"), "r+")
    x = np.squeeze(nc_input.variables["x"]).astype("float32")
    y = np.flip(np.squeeze(nc_input.variables["y"]).astype("float32"))
    
    nc.createDimension("y", len(y))
    yn = nc.createVariable("y", np.dtype("float32").char, ("y",))
    yn.units = "m"
    yn.long_name = "y"
    yn.standard_name = "y"
    yn.axis = "Y"
    yn[:] = y
    
    nc.createDimension("x", len(x))
    xn = nc.createVariable("x", np.dtype("float32").char, ("x",))
    xn.units = "m"
    xn.long_name = "x"
    xn.standard_name = "x"
    xn.axis = "X"
    xn[:] = x
    
    E = nc.createVariable("slidingco", np.dtype("float32").char, ("y", "x"))
    E.long_name = var_info[0]
    E.units = var_info[1]
    E.standard_name = "slidingco"
    E[:] = state.slidingco
    
    nc.close()
    #End of G loop
    #To plot weights if required
    # VolWeights = state.volume_weights.numpy()
    # VolumesNumpy = state.volumes.numpy()
    # fig = plt.figure(2, figsize=(8, 7),dpi=200) 
    # plt.subplot(2, 1, 1)
    # plt.imshow(VolWeights, cmap='jet',origin='lower')
    # plt.colorbar(label='volumeweights')
    # plt.title('volumeweights') 
    # plt.xlabel('Distance, km') 
    # plt.ylabel('Distance, km') 

    # plt.subplot(2, 1, 2)
    # plt.imshow(VolumesNumpy, cmap='jet',origin='lower')
    # plt.colorbar(label='volumes')
    # plt.title('volumes') 
    # plt.xlabel('Distance, km') 
    # plt.ylabel('Distance, km') 
    # plt.show()