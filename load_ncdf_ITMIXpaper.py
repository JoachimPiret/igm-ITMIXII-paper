#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Modif wrt load_ncdf.py :
        - Regrid slidingco from infer_param on itmix grid (this is commented for now)
        - remove velsurf with magnitude < 10m/y from the input field since it is not dynamically significant (also redone in optimizeCookJo)
        - smooth the velocity observations using a gaussian filter to avoid irregularity in usurf when assimilating
"""

import numpy as np
import os
import datetime, time
import tensorflow as tf
from netCDF4 import Dataset
from scipy.interpolate import RectBivariateSpline
import xarray as xr
from igm.modules.utils import *
from scipy.ndimage import gaussian_filter


def params(parser):
    parser.add_argument(
        "--lncd_input_file",
        type=str,
        default="input.nc",
        help="NetCDF input data file",
    )
    parser.add_argument(
        "--lncd_method_coarsen",
        type=str,
        default="skipping",
        help="Method for coarsening the data from NetCDF file: skipping or cubic_spline",
    )
    parser.add_argument(
        "--lncd_coarsen",
        type=int,
        default=1,
        help="Coarsen the data from NetCDF file by a certain (integer) number: 2 would be twice coarser ignore data each 2 grid points",
    )
    parser.add_argument(
        "--lncd_crop",
        type=str2bool,
        default="False",
        help="Crop the data from NetCDF file with given top/down/left/right bounds",
    )
    parser.add_argument(
        "--lncd_xmin",
        type=float,
        help="X left coordinate for cropping the NetCDF data",
        default=-(10**20),
    )
    parser.add_argument(
        "--lncd_xmax",
        type=float,
        help="X right coordinate for cropping the NetCDF data",
        default=10**20,
    )
    parser.add_argument(
        "--lncd_ymin",
        type=float,
        help="Y bottom coordinate fro cropping the NetCDF data",
        default=-(10**20),
    )
    parser.add_argument(
        "--lncd_ymax",
        type=float,
        help="Y top coordinate for cropping the NetCDF data",
        default=10**20,
    )
    parser.add_argument(
        "--sc_input_file",
        type=str,
        default="data/slidingco.nc",
        help="NetCDF input data file",
    )
    parser.add_argument(
        "--velthresh",
        type=float,
        default=10,
        help="Velocity slower than this treshold are not considered by the inversion process",
    )


def initialize(params, state):
    if hasattr(state, "logger"):
        state.logger.info("LOAD NCDF file")

    nc = Dataset(params.lncd_input_file, "r")

    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.squeeze(nc.variables["y"]).astype("float32")

    # make sure the grid has same cell spacing in x and y
    # assert abs(x[1] - x[0]) == abs(y[1] - y[0])

    # load any field contained in the ncdf file, replace missing entries by nan

    if "time" in nc.variables:
        TIME = np.squeeze(nc.variables["time"]).astype("float32")
        I = np.where(TIME == params.time_start)[0][0]
        istheretime = True
    else:
        istheretime = False

    for var in nc.variables:
        if not var in ["x", "y", "z", "time"]:
            if istheretime:
                vars()[var] = np.squeeze(nc.variables[var][I]).astype("float32")
            else:
                vars()[var] = np.squeeze(nc.variables[var]).astype("float32")
            vars()[var] = np.where(vars()[var] > 10**35, np.nan, vars()[var])
            # vars()[var] = np.where(vars()[var] == -9999, np.nan, vars()[var])

    # Apply Gaussian filter only to specific variables
    # for var in ["vvelsurfobs", "uvelsurfobs"]:
    #     if var in vars():  # Check if the variable exists
    #         vars()[var] = gaussian_filter(vars()[var], sigma=5, mode="constant")

    # coarsen if requested
    if params.lncd_coarsen > 1:
        xx = x[:: params.lncd_coarsen]
        yy = y[:: params.lncd_coarsen]
         
        if params.lncd_method_coarsen == "skipping":            
            for var in nc.variables:
                if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                    vars()[var] = vars()[var][
                        :: params.lncd_coarsen, :: params.lncd_coarsen
                    ]
        elif params.lncd_method_coarsen == "cubic_spline":
            for var in nc.variables:
                if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                                    
                    interp_spline = RectBivariateSpline(y, x, vars()[var])
                    vars()[var] = interp_spline(yy, xx)
        x = xx
        y = yy



    # crop if requested
    if params.lncd_crop:
        i0 = max(0, int((params.lncd_xmin - x[0]) / (x[1] - x[0])))
        i1 = min(int((params.lncd_xmax - x[0]) / (x[1] - x[0])), x.shape[0] - 1)
        i1 = max(i0 + 1, i1)
        j0 = max(0, int((params.lncd_ymin - y[0]) / (y[1] - y[0])))
        j1 = min(int((params.lncd_ymax - y[0]) / (y[1] - y[0])), y.shape[0] - 1)
        j1 = max(j0 + 1, j1)
        #        i0,i1 = int((params.lncd_xmin-x[0])/(x[1]-x[0])),int((params.lncd_xmax-x[0])/(x[1]-x[0]))
        #        j0,j1 = int((params.lncd_ymin-y[0])/(y[1]-y[0])),int((params.lncd_ymax-y[0])/(y[1]-y[0]))
        for var in nc.variables:
            if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                vars()[var] = vars()[var][j0:j1, i0:i1]
        y = y[j0:j1]
        x = x[i0:i1]

    # transform from numpy to tensorflow
    for var in nc.variables:
        if not var in ["z", "time"]:
            if var in ["x", "y"]:
                vars(state)[var] = tf.constant(vars()[var].astype("float32"))
            else:
                vars(state)[var] = tf.Variable(vars()[var].astype("float32"))
    # Filter velocity with magnitude < 10 m/y
    if hasattr(state, 'uvelsurfobs'):
        print('Threshold applied to velocity')
        threshold = params.velthresh
        nanmask = tf.where(state.icemask>0.5,state.icemask,tf.constant(np.nan))
        velmag = getmag(state.uvelsurfobs, state.vvelsurfobs)
        state.uvelsurfobs = tf.where(velmag > threshold,state.uvelsurfobs,tf.constant(np.nan))*nanmask
        state.vvelsurfobs = tf.where(velmag > threshold,state.vvelsurfobs,tf.constant(np.nan))*nanmask
        state.velsurfobs_mag = tf.where(velmag > threshold,velmag,tf.constant(np.nan))*nanmask
    else :
        pass
    
    fieldmask = tf.equal(state.usurf, -9999)
    # Shift values left (fill with last valid value)
    def fill_missing_values(tensor):
        for i in range(tensor.shape[1] - 1):  # Iterate over columns
            tensor = tf.where(fieldmask, tf.roll(tensor, shift=-1, axis=1), tensor)
        return tensor
    
    state.usurf =(fill_missing_values(state.usurf))
    state.usurfobs = (fill_missing_values(state.usurfobs))
    # #Regrid slidingco
    # nc_sc = Dataset(params.sc_input_file, "r")
    # x_sc = np.squeeze(nc_sc.variables["x"]).astype("float32")
    # y_sc = np.squeeze(nc_sc.variables["y"]).astype("float32")
    # if len(x_sc) != len(x) or len(y_sc) != len(y):
    #     print('Regridding of slidingco')
    #     good_grid_x = xr.open_dataset(params.lncd_input_file).x
    #     good_grid_y = xr.open_dataset(params.lncd_input_file).y
    #     old_grid_sc = xr.open_dataset(params.sc_input_file).slidingco
        
        
    #     good_grid_sc = old_grid_sc.interp(x =good_grid_x, y =good_grid_y, method = 'nearest')
    #     os.remove(params.sc_input_file)
    #     good_grid_sc.to_netcdf(params.sc_input_file)
    # else :
    #     print('No regridding of slidingco')
        
    nc.close()

    complete_data(state)


def update(params, state):
    pass


def finalize(params, state):
    pass

