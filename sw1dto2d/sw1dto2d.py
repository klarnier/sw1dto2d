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


"""SW1Dto2D object for river cross-section lines and points reconstruction from centerline."""


import logging
from typing import Dict, List, Tuple

import geopy.distance
import numpy as np
import pandas as pd
from pyproj.transformer import Transformer
from shapely.geometry import LineString, Point, Polygon


class SW1Dto2D:
    """Builds river cross-section data from river centerline data."""

    def __init__(
        self,
        model_output_1d: pd.DataFrame,
        curvilinear_abscissa_key: str,
        heights_key: str,
        widths_key: str,
        centerline: LineString,
    ):
        """
        Constructor.

        Parameters
        ----------
            model_output_1d : 1D-Hydraulics output data
            curvilinear_abscissa_key : river center line curvilinear abscissa key in model_output_1d data
            heights_key : heights key in model_output_1d data
            widths_key : widths key in model_output_1d data
            centerline : river center line geometry
        """

        # results = pd.read_csv(model_output_1d, sep=";")
        results = model_output_1d

        self._curv_abscissa_base = results[curvilinear_abscissa_key].unique()
        self._curv_abscissa = results[curvilinear_abscissa_key].unique()
        self._curv_absc_coords = None
        self._curv_absc_normals = None

        widths = results[widths_key].values
        self._widths = widths.reshape((widths.size // self._curv_abscissa.size, self._curv_abscissa.size))

        heights = results[heights_key].values
        self._heights = heights.reshape((widths.size // self._curv_abscissa.size, self._curv_abscissa.size))

        self._centerline = centerline

    @property
    def curv_abscissa(self):
        """Get curv_abscissa parameter."""

        return self._curv_abscissa

    def compute_xs_parameters(self, dx: float = None, optimize_normals: bool = True, enforce_length: bool = True):
        """
        Compute cross-sections parameters (coordinates and normals)

        Parameters
        ----------
            dx : Resampling spacing (in curvilinear axis)
            optimize_normals : True to optimize normals to prevent cutlines instersections
            enforce_length : True to enforce length of the centerline to the total distance computed from xs
        """

        logging.info("-" * 60)
        logging.info("Compute cross-sections parameters (coordinates and normals)")
        logging.info("-" * 60)

        if self._centerline is None:
            raise RuntimeError("centerline is not set yet")

        if dx is not None:
            L = np.abs(self._curv_abscissa_base[-1] - self._curv_abscissa_base[0])
            n = int(np.round(L / dx))
            self._curv_abscissa = np.linspace(
                self._curv_abscissa_base[0], self._curv_abscissa_base[-1], n + 1, endpoint=True
            )

        # Compute alpha (ratio of length) of the cross-sections on the centerline
        logging.info("Compute alpha values for the cross-sections")
        xs_alpha = self._compute_xs_alpha(enforce_length)

        # Compute alpha (ratio of length) of the points on the centerline
        logging.info("Compute alpha values for the centerline points")
        cl_alpha = self._compute_cl_alpha()

        # Compute positions of the cross-sections on the centerline
        logging.info("Compute cross-sections centers coordinates")
        xs_coords = self._compute_xs_coords(xs_alpha, cl_alpha)

        # Compute (raw) normals of the cross-sections on the centerline
        logging.info("Compute raw normals")
        xs_normals = self._compute_xs_normals(xs_alpha, cl_alpha)
        logging.info("")

        # Optimize normals
        if optimize_normals is True:
            # xs_normals, intersect_flag = self._optimize_xs_normals(xs_coords, xs_normals)
            xs_normals, _ = self._optimize_xs_normals(xs_coords, xs_normals)

        self._curv_absc_normals = xs_normals
        self._curv_absc_coords = xs_coords

    def _compute_xs_alpha(self, xs: np.ndarray, enforce_length: bool = True) -> np.ndarray:
        """
        Compute alpha (ratio of length) of the cross-sections on the centerline

        Parameters
        ----------
            xp : array of curvilinear abscissae
            enforce_length : True to enforce length of the centerline to the total distance computed from xs

        Return
        ------
            xs_alpha : array of alpha
        """

        if enforce_length:
            xs_alpha = (self._curv_abscissa[:] - self._curv_abscissa[0]) / (
                self._curv_abscissa[-1] - self._curv_abscissa[0]
            )
        else:
            raise NotImplementedError("Computation of positions with enforce_length=False is not implemented yet")

        return xs_alpha

    def _compute_cl_alpha(self) -> np.ndarray:
        """
        Compute alpha (ratio of length) of the points of the centerline

        Return
        ------
            cl_alpha : array of alpha values
        """

        coords = np.array(self._centerline.coords)
        d_x = coords[1:, 0] - coords[0:-1, 0]
        d_y = coords[1:, 1] - coords[0:-1, 1]
        dist = np.cumsum(np.sqrt(d_x**2 + d_y**2))
        dist = np.insert(dist, 0, 0.0)
        cl_alpha = dist / dist[-1]

        return cl_alpha

    def _compute_xs_coords(self, xs_alpha: np.ndarray, cl_alpha: np.ndarray) -> np.ndarray:
        """
        Compute positions of the cross-sections on the centerline

        Parameters
        ----------
            xs_alpha : array of alpha (ratio of length) values for the cross-sections
            cl_alpha : array of alpha (ratio of length) values for the points of the centerline

        Return
        ------
            result : array of coordinates
        """

        coords = np.array(self._centerline.coords)
        x = np.interp(xs_alpha, cl_alpha, coords[:, 0])
        y = np.interp(xs_alpha, cl_alpha, coords[:, 1])

        return np.stack((x, y), axis=-1)

    def _compute_xs_normals(self, xs_alpha: np.ndarray, cl_alpha: np.ndarray) -> np.ndarray:
        """
        Compute normals of the cross-sections on the centerline

        Parameters
        ----------
            xs_alpha : array of alpha (ratio of length) values for the cross-sections
            cl_alpha : array of alpha (ratio of length) values for the points of the centerline

        Return
        ------
            result : array of normals
        """

        coords = np.array(self._centerline.coords)
        dx = coords[2:, 0] - coords[0:-2, 0]
        dy = coords[2:, 1] - coords[0:-2, 1]
        dx = np.insert(dx, 0, coords[1, 0] - coords[0, 0])
        dx = np.append(dx, coords[-1, 0] - coords[-2, 0])
        dy = np.insert(dy, 0, coords[1, 1] - coords[0, 1])
        dy = np.append(dy, coords[-1, 1] - coords[-2, 1])
        norm = np.sqrt(dx**2 + dy**2)

        cl_nx = dy / norm
        cl_ny = -dx / norm

        xs_nx = np.interp(xs_alpha, cl_alpha, cl_nx)
        xs_ny = np.interp(xs_alpha, cl_alpha, cl_ny)

        return np.stack((xs_nx, xs_ny), axis=-1)

    def _optimize_xs_normals(  # noqa: C901
        self, xs_coords: np.ndarray, xs_normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize normals by filtering to prevent intersections

        Parameters
        ----------
            xs_coords : array of coordinates of the cross-sections
            xs_normals : array of normals of the cross-sections

        Return
        ------
            result : tuple (xs_normals, intersect_flag) where xs_normals is a array of normals and intersect_flag an
                array of intersection flag
        """

        logging.info("-" * 60)
        logging.info("Optimization of normals to prevent intersections")
        logging.info("-" * 60)

        # Compute normals angles
        angles = np.arctan2(xs_normals[:, 0], xs_normals[:, 1]) * 180.0 / np.pi
        # angles_base = angles.copy()

        # Compute max widths
        Wmax = np.max(self._widths, axis=0)

        if self._curv_abscissa_base.size != self._curv_abscissa.size:
            Wmax = np.interp(self._curv_abscissa, self._curv_abscissa_base, Wmax)

        nintersect = 1
        while nintersect > 0:
            # Compute cross-sections lines
            xs_lines = []

            for ix in range(0, xs_coords.shape[0]):
                Wdemi = 0.5 * Wmax[ix]
                bearing1 = angles[ix]

                # TODO correct bearing !!! 0=North, 90=east, ect :/
                if angles[ix] > 180.0:
                    bearing2 = angles[ix] - 180.0
                else:
                    bearing2 = angles[ix] + 180.0

                center = (xs_coords[ix, 1], xs_coords[ix, 0])
                point1 = geopy.distance.distance(meters=Wdemi).destination(center, bearing=bearing1)
                point2 = geopy.distance.distance(meters=Wdemi).destination(center, bearing=bearing2)
                xs_lines.append(LineString([(point1.longitude, point1.latitude), (point2.longitude, point2.latitude)]))

            # Flag intersecting lines
            intersect_flag = np.zeros(xs_coords.shape[0], dtype=int)

            for ix1 in range(0, xs_coords.shape[0]):
                for ix2 in range(ix1 + 1, xs_coords.shape[0]):
                    if xs_lines[ix1].intersects(xs_lines[ix2]):
                        intersect_flag[ix1] += 1
                        intersect_flag[ix2] += 1
            nintersect = np.sum(intersect_flag)

            logging.info(f"Number of intersections: {nintersect}")

            if nintersect == 0:
                continue

            indices = np.argwhere(intersect_flag > 0).flatten()
            seq = indices[1:] - indices[:-1]
            seq = np.insert(seq, 0, 2)
            seq_start = np.argwhere(seq > 1).flatten()

            logging.info(f"Number of intersecting ranges: {len(seq_start)}")

            for i in range(0, len(seq_start)):
                start = indices[seq_start[i]]

                if i < len(seq_start) - 1:
                    end = indices[seq_start[i + 1] - 1]
                else:
                    end = indices[-1]

                logging.info("Processing intersecting range: %i-%i" % (start, end))

                nintersect2 = 1
                while nintersect2 > 0:
                    new_angles = angles.copy()
                    xr = self._curv_abscissa[start : end + 1]
                    xb = [self._curv_abscissa[start], self._curv_abscissa[(start + end) // 2], self._curv_abscissa[end]]

                    if xr[-1] < xr[0]:
                        xr = xr[0] - xr
                        xb = xb[0] - xb

                    anglesb = [angles[start], angles[(start + end) // 2], angles[end]]
                    new_angles[start : end + 1] = np.interp(xr, xb, anglesb)

                    # Compute new lines
                    new_lines = xs_lines.copy()
                    for ix in range(start, end + 1):
                        Wdemi = 0.5 * Wmax[ix]
                        bearing1 = new_angles[ix]

                        if new_angles[ix] > 180.0:
                            bearing2 = new_angles[ix] - 180.0
                        else:
                            bearing2 = new_angles[ix] + 180.0

                        center = (xs_coords[ix, 1], xs_coords[ix, 0])
                        point1 = geopy.distance.distance(meters=Wdemi).destination(center, bearing=bearing1)
                        point2 = geopy.distance.distance(meters=Wdemi).destination(center, bearing=bearing2)
                        new_lines[ix] = LineString(
                            [(point1.longitude, point1.latitude), (point2.longitude, point2.latitude)]
                        )

                    # Check for remaining intersections
                    nintersect2 = 0
                    for ix1 in range(max(0, start - 2), min(end + 3, xs_coords.shape[0])):
                        for ix2 in range(ix1 + 1, min(end + 3, xs_coords.shape[0])):
                            if new_lines[ix1].intersects(new_lines[ix2]):
                                nintersect2 += 1

                    if nintersect2 > 0:
                        start = max(0, start - 1)
                        end = min(end + 1, xs_coords.shape[0] - 1)

                logging.info("Intersecting range filtered with range[%i-%i]" % (start, end))

                xs_lines[:] = new_lines[:]
                angles[:] = new_angles[:]
                del new_lines

                # Recompute normals
                xs_normals[:, 0] = np.sin(angles * np.pi / 180.0)
                xs_normals[:, 1] = np.cos(angles * np.pi / 180.0)

        logging.info("")

        return xs_normals, intersect_flag

    def compute_xs_cutlines(self, it: int = None, extend: int = None) -> List[LineString]:
        """
        Compute cutlines of the cross-sections on the centerline

        Parameters
        ----------
            it : index of time occurence (in the H and W arrays)
            extend :

        Return
        ------
            xs_lines : list of LineStrings objects for the cutlines
        """

        xs_coords = self._curv_absc_coords
        xs_normals = self._curv_absc_normals
        angles = np.arctan2(xs_normals[:, 0], xs_normals[:, 1]) * 180.0 / np.pi

        # Compute widths
        if it is None or it == "max":
            Wit = np.max(self._widths, axis=0)
        elif it == "min":
            Wit = np.min(self._widths, axis=0)
        elif it == "mean":
            Wit = np.mean(self._widths, axis=0)
        else:
            Wit = self._widths[it, :]

        if self._curv_abscissa_base.size != self._curv_abscissa.size:
            Wit = np.interp(self._curv_abscissa, self._curv_abscissa_base, Wit)

        if extend is not None:
            Wit = Wit + extend

        xs_lines = []
        for ix in range(0, xs_coords.shape[0]):
            Wdemi = 0.5 * Wit[ix]
            bearing1 = angles[ix]

            if angles[ix] > 180.0:
                bearing2 = angles[ix] - 180.0
            else:
                bearing2 = angles[ix] + 180.0

            center = (xs_coords[ix, 1], xs_coords[ix, 0])
            point1 = geopy.distance.distance(meters=Wdemi).destination(center, bearing=bearing1)
            point2 = geopy.distance.distance(meters=Wdemi).destination(center, bearing=bearing2)
            xs_lines.append(LineString([(point1.longitude, point1.latitude), (point2.longitude, point2.latitude)]))

        return xs_lines

    def compute_water_mask(self, it: int = None):
        """Compute watermask

        Parameters
        ----------
            it : index of time occurence (in the H and W arrays)

        Return
        ------
            polygon: Polygon object of the water mask
        """

        xs_coords = self._curv_absc_coords
        xs_normals = self._curv_absc_normals
        angles = np.arctan2(xs_normals[:, 0], xs_normals[:, 1]) * 180.0 / np.pi

        # Compute widths
        if it is None or it == "max":
            Wit = np.max(self._widths, axis=0)
        elif it == "min":
            Wit = np.min(self._widths, axis=0)
        elif it == "mean":
            Wit = np.mean(self._widths, axis=0)
        else:
            Wit = self._widths[it, :]

        if self._curv_abscissa_base.size != self._curv_abscissa.size:
            Wit = np.interp(self._curv_abscissa, self._curv_abscissa_base, Wit)

        left_points = []
        right_points = []
        for ix in range(0, xs_coords.shape[0]):
            Wdemi = 0.5 * Wit[ix]
            bearing1 = angles[ix]

            if angles[ix] > 180.0:
                bearing2 = angles[ix] - 180.0
            else:
                bearing2 = angles[ix] + 180.0

            center = (xs_coords[ix, 1], xs_coords[ix, 0])
            point1 = geopy.distance.distance(meters=Wdemi).destination(center, bearing=bearing1)
            point2 = geopy.distance.distance(meters=Wdemi).destination(center, bearing=bearing2)
            left_points.append((point2.longitude, point2.latitude))
            right_points.insert(0, (point1.longitude, point1.latitude))

        polygon = Polygon(right_points + left_points)

        return polygon

    def compute_xs_points(
        self,
        main_channel: int = None,
        overbanks: int = None,
        extend: int = None,
        overbank_slope: float = 0.0001,
        epsg: int = 4326,
    ) -> Tuple[List[Point], List[Dict]]:
        """
        Compute cutlines of the cross-sections on the centerline

        Parameters
        ----------
            main_channel : number of points in the main channel cross-section part
            overbanks : number of points in each overbank part
            extend : TODO
            overbank_slope : overbank slope
            epsg : EPSG code

        Return
        ------
            result : tuple (points, attributes) with points is a list of LineStrings objects for the cutlines,
                and attributes the corresponding points attributes
        """

        xs_coords = self._curv_absc_coords
        xs_normals = self._curv_absc_normals
        angles = np.arctan2(xs_normals[:, 0], xs_normals[:, 1]) * 180.0 / np.pi

        Wit = np.max(self._widths, axis=0)
        if self._curv_abscissa_base.size != self._curv_abscissa.size:
            Wit = np.interp(self._curv_abscissa, self._curv_abscissa_base, Wit)

        # Create pyporj transformer
        transformer = Transformer.from_crs(4326, epsg, always_xy=True)

        points = []
        attributes = []
        for ix in range(0, xs_coords.shape[0]):
            # index0 = len(points)

            # Compute left overbank points
            Hmax = np.max(self._heights[:, ix])
            alpha_widths = np.linspace(1.0, 0.0, overbanks, endpoint=False)
            for index in range(0, overbanks):
                dist = -(0.5 * Wit[ix] + alpha_widths[index] * extend)
                bearing1 = angles[ix]
                center = (xs_coords[ix, 1], xs_coords[ix, 0])
                point = geopy.distance.distance(meters=dist).destination(center, bearing=bearing1)
                points.append(Point(point.longitude, point.latitude))
                x, y = transformer.transform(point.longitude, point.latitude)
                z = Hmax + alpha_widths[index] * overbank_slope * extend
                attributes.append(
                    {
                        "xs_id": ix,
                        "xsnd_id": index,
                        "abs": self._curv_abscissa[ix],
                        "loc": "LOB",
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )

            # Interpolate (H,W) tuples sorted with increasing H for the main channel
            Hx = self._heights[:, ix]
            Wx = self._widths[:, ix]
            index_sorted = np.argsort(Hx)
            Hx = Hx[index_sorted]
            Wx = Wx[index_sorted]

            # Compute main channel points
            alpha_widths = np.linspace(-0.5, 0.5, main_channel, endpoint=True)
            Wp = 2.0 * np.abs(alpha_widths * Wit[ix])
            Hp = np.interp(Wp, Wx, Hx)
            for index in range(0, main_channel):
                dist = alpha_widths[index] * Wit[ix]
                bearing1 = angles[ix]
                center = (xs_coords[ix, 1], xs_coords[ix, 0])
                point = geopy.distance.distance(meters=dist).destination(center, bearing=bearing1)
                points.append(Point(point.longitude, point.latitude))
                x, y = transformer.transform(point.longitude, point.latitude)
                z = Hp[index]
                attributes.append(
                    {
                        "xs_id": ix,
                        "xsnd_id": index + overbanks,
                        "abs": self._curv_abscissa[ix],
                        "loc": "MC",
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )

            # Compute right overbank points
            Hmax = np.max(self._heights[:, ix])
            alpha_widths = np.linspace(1.0, 0.0, overbanks, endpoint=False)
            alpha_widths = 1.0 - alpha_widths
            for index in range(0, overbanks):
                dist = 0.5 * Wit[ix] + alpha_widths[index] * extend
                bearing1 = angles[ix]
                center = (xs_coords[ix, 1], xs_coords[ix, 0])
                point = geopy.distance.distance(meters=dist).destination(center, bearing=bearing1)
                points.append(Point(point.longitude, point.latitude))
                x, y = transformer.transform(point.longitude, point.latitude)
                z = Hmax + alpha_widths[index] * overbank_slope * extend
                attributes.append(
                    {
                        "xs_id": ix,
                        "xsnd_id": index + main_channel + overbanks,
                        "abs": self._curv_abscissa[ix],
                        "loc": "ROB",
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )

        return points, attributes
