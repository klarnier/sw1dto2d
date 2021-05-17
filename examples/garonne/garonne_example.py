import fiona
from fiona.crs import from_epsg
import numpy as np
import pandas as pd
from shapely.geometry import LineString, mapping, Polygon, shape

import dassflow1d
from sw1dto2d import SW1Dto2D

def export_to_shp(fname, geo):
    if isinstance(geo, list):
        schema = {"geometry": "LineString",
                  "properties": {("xs", "float")}}
        fout =  fiona.open(fname, "w", driver="ESRI Shapefile", crs=from_epsg(4326), schema=schema)
        for index, line in enumerate(lines):
            feature = {"geometry": mapping(line),
                       "properties": {"xs": sw1dto2d.xs[index]}}
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
        

# Load centerline    
layer = fiona.open("centerline.shp", "r")
centerline = shape(layer[0]["geometry"])

# Load Results
results = pd.read_csv("Garonne2019_results_selection.csv", sep=";")
xs = results["xs"].unique()
W = results["W"].values
W = W.reshape((W.size//xs.size, xs.size))
H = results["H"].values
H = H.reshape((W.size//xs.size, xs.size))

# Instanciate SW1Dto2D object
sw1dto2d = SW1Dto2D(xs, H, W, centerline)

# Compute cross-sections parameters without normals optimization
sw1dto2d.compute_xs_parameters(dx=50, optimize_normals=False)

# Export cross-sections cut lines (with maximum width as argument it=None for SW1Dto2D.compute_xs_geometry)
lines = sw1dto2d.compute_xs_cutlines()
export_to_shp("Garonne_cutlines_raw_normals.shp", lines)

# Compute cross-sections parameters with normals optimization
sw1dto2d.compute_xs_parameters(dx=50, optimize_normals=True)

# Export cross-sections cut lines (with maximum width as argument it=None for SW1Dto2D.compute_xs_geometry)
lines = sw1dto2d.compute_xs_cutlines()
export_to_shp("Garonne_cutlines_opt_normals.shp", lines)

# Export maximum water mask (argument it=None for SW1Dto2D.compute_water_mask)
poly = sw1dto2d.compute_water_mask()
export_to_shp("Garonne_max_water_mask.shp", poly)

# Export water mask at 10th time (argument it=10 for SW1Dto2D.compute_water_mask)
poly = sw1dto2d.compute_water_mask(it=10)
export_to_shp("Garonne_water_mask_it10.shp", poly)
