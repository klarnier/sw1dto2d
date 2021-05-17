import numpy as np
from shapely.geometry import LineString

from sw1dto2d import SW1Dto2D

def test_xs_coords():
    """
    Test computation of cross-sections coordinates on centerline
    """
    
    # Create curvilinear abscissae
    xs = np.linspace(0.0, 10.0, 11, endpoint=True)

    # Create centerline
    lon = np.linspace(-0.1, 0.1, 51, endpoint=True)
    lat = np.linspace(43.0, 43.50, 51, endpoint=True)
    coords = [(x, y) for x,y in zip(lon, lat)]
    centerline = LineString(coords)
    
    # Create arrays of wse and widths
    H = np.random.uniform(0.0, 10.0, (10, 11))
    W = np.random.uniform(10.0, 100.0, (10, 11))
    
    # Instanciate SW1Dto2D object
    sw1dto2d = SW1Dto2D(xs, H, W, centerline)

    # Compute alphas
    xs_alpha = sw1dto2d.compute_xs_alpha(xs, enforce_length=True)
    cl_alpha = sw1dto2d.compute_cl_alpha()
    
    # Compute cross-sections coordinates on centerline
    coords = sw1dto2d.compute_xs_coords(xs_alpha, cl_alpha)
    
    # Test computed coordinates
    xs_lon = np.linspace(-0.1, 0.1, 11, endpoint=True)
    xs_lat = np.linspace(43.0, 43.50, 11, endpoint=True)
    assert(np.allclose(coords, np.stack((xs_lon, xs_lat), axis=-1)))

def test_xs_normals():
    """
    Test computation of cross-sections normals
    """
    
    # Create curvilinear abscissae
    xs = np.linspace(0.0, 10.0, 11, endpoint=True)

    # Create centerline
    lon = np.linspace(-0.1, 0.1, 51, endpoint=True)
    lat = np.linspace(43.0, 43.50, 51, endpoint=True)
    coords = [(x, y) for x,y in zip(lon, lat)]
    centerline = LineString(coords)
    
    # Create arrays of wse and widths
    H = np.random.uniform(0.0, 10.0, (10, 11))
    W = np.random.uniform(10.0, 100.0, (10, 11))
    
    # Instanciate SW1Dto2D object
    sw1dto2d = SW1Dto2D(xs, H, W, centerline)

    # Compute alphas
    xs_alpha = sw1dto2d.compute_xs_alpha(xs, enforce_length=True)
    cl_alpha = sw1dto2d.compute_cl_alpha()
    
    # Compute cross-sections normals
    xs_normals = sw1dto2d.compute_xs_normals(xs_alpha, cl_alpha)
    
    # Test computed normals
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]
    norm2 = np.sqrt(dlon**2 + dlat**2)
    nx = dlat / norm2
    ny = -dlon / norm2
    expected_normals = np.repeat([[nx, ny]], 11, axis=0)
    assert(np.allclose(xs_normals, expected_normals))
    
if __name__ == "__main__":
    test_xs_coords()
    test_xs_normals()
