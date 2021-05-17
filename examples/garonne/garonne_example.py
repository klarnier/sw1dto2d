import fiona
from fiona.crs import from_epsg
import numpy as np
import pandas as pd
from shapely.geometry import LineString, mapping, shape

import dassflow1d
from sw1dto2d import SW1Dto2D

def export_to_shp(fname, lines):
    schema = {'geometry': 'LineString',
            'properties': {('xs', 'float')}}
    fout =  fiona.open(fname, 'w', driver="ESRI Shapefile", crs=from_epsg(4326), schema=schema)
    for index, line in enumerate(lines):
        feature = {"geometry": mapping(line),
                "properties": {"xs": sw1dto2d.xs[index]}}
        fout.write(feature)
    fout.close()

# Load centerline    
layer = fiona.open("centerline_single_WGS84.shp", "r")
centerline = shape(layer[0]["geometry"])

# Load Results
results = pd.read_csv("Garonne2019_results.csv", sep=";")
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
lines = sw1dto2d.compute_xs_geometry()
fname = 'Garonne_cross_sections_%i.shp' % 0
export_to_shp(fname, lines)

# Compute cross-sections parameters with normals optimization
sw1dto2d.compute_xs_parameters(dx=50, optimize_normals=True)

# Export cross-sections cut lines (with maximum width as argument it=None for SW1Dto2D.compute_xs_geometry)
lines = sw1dto2d.compute_xs_geometry()
fname = 'Garonne_cross_sections_%i.shp' % 1
export_to_shp(fname, lines)
