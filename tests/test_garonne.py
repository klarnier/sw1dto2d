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

import fiona
from fiona.crs import from_epsg
import numpy as np
import os
import pandas as pd
from shapely.geometry import LineString, mapping, Polygon, shape

from sw1dto2d.sw1dto2d import SW1Dto2D

def export_to_shp(fname, geo, sw1dto2d):
    """ Export a geometry or a list of geometries to a ESRI/Shapefile
    
    Parameters
    ----------
        fname (str): Path of the shp output file
        geo (shapely.geometry.BaseGeometry or list): Geometry or list of geometries
    """

    if isinstance(geo, list):
        schema = {"geometry": "LineString",
                  "properties": {("xs", "float")}}
        fout =  fiona.open(fname, "w", driver="ESRI Shapefile", crs=from_epsg(4326), schema=schema)
        for index, line in enumerate(geo):
            feature = {"geometry": mapping(line),
                       "properties": {"xs": sw1dto2d.curv_abscissa[index]}}
            fout.write(feature)
        fout.close()
    elif isinstance(geo, Polygon):
        schema = {"geometry": "Polygon",
                  "properties": {("ID", "int")}}
        fout =  fiona.open(fname, "w", driver="ESRI Shapefile", crs=from_epsg(4326), schema=schema)
        feature = {"geometry": mapping(geo),
                   "properties": {"ID": 0}}
        fout.write(feature)
        fout.close()

def test_garonne():
    """Test garonne cross-section lines and points reconstruction."""

    # Create output directory if necessary
    if not os.path.isdir("out"):
        os.mkdir("out")
            
    # Load centerline  
    centerline_path = os.path.join(
        os.path.dirname(__file__),
        "data/garonne/centerline.shp"
    )  
    layer = fiona.open(centerline_path, "r")
    centerline = shape(layer[0]["geometry"])

    # Load Results
    results_path = os.path.join(
        os.path.dirname(__file__),
        "data/garonne/Garonne2019_results_selection.csv"
    )
    results = pd.read_csv(results_path, sep=";")
    # xs = results["xs"].unique()
    # W = results["W"].values
    # W = W.reshape((W.size//xs.size, xs.size))
    # H = results["H"].values
    # H = H.reshape((W.size//xs.size, xs.size))

    # Instanciate SW1Dto2D object
    # sw1dto2d = SW1Dto2D(xs, H, W, centerline)
    sw1dto2d = SW1Dto2D(
        model_output_1d=results,
        curvilinear_abscissa_key="xs",
        heights_key="H",
        widths_key="W",
        centerline=centerline
    )

    # Compute cross-sections parameters without normals optimization
    sw1dto2d.compute_xs_parameters(dx=50, optimize_normals=False)

    # Export cross-sections cut lines (with maximum width as argument it=None for SW1Dto2D.compute_xs_geometry)
    lines = sw1dto2d.compute_xs_cutlines()
    export_to_shp("out/Garonne_cutlines_raw_normals.shp", lines, sw1dto2d)

    # Compute cross-sections parameters with normals optimization
    sw1dto2d.compute_xs_parameters(dx=50, optimize_normals=True)

    # Export cross-sections cut lines (with maximum width as argument it=None for SW1Dto2D.compute_xs_geometry)
    lines = sw1dto2d.compute_xs_cutlines()
    export_to_shp("out/Garonne_cutlines_opt_normals.shp", lines, sw1dto2d)

    # Export maximum water mask (argument it=None for SW1Dto2D.compute_water_mask)
    poly = sw1dto2d.compute_water_mask()
    export_to_shp("out/Garonne_max_water_mask.shp", poly, sw1dto2d)

    # Export water mask at 10th time (argument it=10 for SW1Dto2D.compute_water_mask)
    poly = sw1dto2d.compute_water_mask(it=10)
    export_to_shp("out/Garonne_water_mask_it10.shp", poly, sw1dto2d)
