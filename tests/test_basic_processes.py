# coding: utf-8

# Copyright (c) 2019-2022 CS GROUP - France, University of Toulouse (France)
#
# This file is part of the sw1dto2d Package. sw1dto2d is a package to convert
# 1D Shallow water results to 2D. It is designed to compute 2D geographic 
# representation of the results of 1D shallow water models.
#
# This software is governed by the CeCILL license under French law and abiding 
# by the rules of distribution of free software. You can use, modify and/or 
# redistribute the software under the terms of the CeCILL license as circulated 
# by CEA, CNRS and INRIA at the following URL: "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy, modify 
# and redistribute granted by the license, users are provided only with a 
# limited warranty and the software's author, the holder of the economic 
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with 
# loading, using, modifying and/or developing or reproducing the software by 
# the user in light of its specific status of free software, that may mean that 
# it is complicated to manipulate, and that also therefore means that it is 
# reserved for developers and experienced professionals having in-depth 
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling 
# the security of their systems and/or data to be ensured and, more generally, 
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had 
# knowledge of the CeCILL license and that you accept its terms.

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from sw1dto2d.sw1dto2d import SW1Dto2D

def test_xs_coords():
    """ Test computation of cross-sections coordinates on centerline
    """
    
    # Create curvilinear abscissae
    xs_elem = np.linspace(0.0, 10.0, 11, endpoint=True)
    xs = np.repeat(xs_elem, 10)

    # Create centerline
    lon = np.linspace(-0.1, 0.1, 51, endpoint=True)
    lat = np.linspace(43.0, 43.50, 51, endpoint=True)
    coords = [(x, y) for x,y in zip(lon, lat)]
    centerline = LineString(coords)
    
    # Create arrays of wse and widths
    H = np.random.uniform(0.0, 10.0, (10 * 11))
    W = np.random.uniform(10.0, 100.0, (10 * 11))
    
    # Create output data
    output_data = pd.DataFrame()
    output_data["xs"] = xs
    output_data["W"] = W
    output_data["H"] = H

    # Instanciate SW1Dto2D object
    # sw1dto2d = SW1Dto2D(xs, H, W, centerline)
    sw1dto2d = SW1Dto2D(
        model_output_1d=output_data,
        curvilinear_abscissa_key="xs",
        heights_key="H",
        widths_key="W",
        centerline=centerline
    )

    # Compute alphas
    xs_alpha = sw1dto2d._compute_xs_alpha(xs, enforce_length=True)
    cl_alpha = sw1dto2d._compute_cl_alpha()
    
    # Compute cross-sections coordinates on centerline
    coords = sw1dto2d._compute_xs_coords(xs_alpha, cl_alpha)
    
    # Test computed coordinates
    xs_lon = np.linspace(-0.1, 0.1, 11, endpoint=True)
    xs_lat = np.linspace(43.0, 43.50, 11, endpoint=True)
    assert(np.allclose(coords, np.stack((xs_lon, xs_lat), axis=-1)))

def test_xs_normals():
    """ Test computation of cross-sections normals
    """
    
    # Create curvilinear abscissae
    xs_elem = np.linspace(0.0, 10.0, 11, endpoint=True)
    xs = np.repeat(xs_elem, 10)

    # Create centerline
    lon = np.linspace(-0.1, 0.1, 51, endpoint=True)
    lat = np.linspace(43.0, 43.50, 51, endpoint=True)
    coords = [(x, y) for x,y in zip(lon, lat)]
    centerline = LineString(coords)
    
    # Create arrays of wse and widths
    H = np.random.uniform(0.0, 10.0, (10 * 11))
    W = np.random.uniform(10.0, 100.0, (10 * 11))
    
    # Create output data
    output_data = pd.DataFrame()
    output_data["xs"] = xs
    output_data["W"] = W
    output_data["H"] = H

    # Instanciate SW1Dto2D object
    # sw1dto2d = SW1Dto2D(xs, H, W, centerline)
    sw1dto2d = SW1Dto2D(
        model_output_1d=output_data,
        curvilinear_abscissa_key="xs",
        heights_key="H",
        widths_key="W",
        centerline=centerline
    )

    # Compute alphas
    xs_alpha = sw1dto2d._compute_xs_alpha(xs, enforce_length=True)
    cl_alpha = sw1dto2d._compute_cl_alpha()
    
    # Compute cross-sections normals
    xs_normals = sw1dto2d._compute_xs_normals(xs_alpha, cl_alpha)
    
    # Test computed normals
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]
    norm2 = np.sqrt(dlon**2 + dlat**2)
    nx = dlat / norm2
    ny = -dlon / norm2
    expected_normals = np.repeat([[nx, ny]], 11, axis=0)
    assert(np.allclose(xs_normals, expected_normals))
