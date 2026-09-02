"""Evidence-preserving geospatial terrain mapping for OpenPRISM.

This module turns timestamped camera evidence plus a calibrated vehicle pose
into a local, north-up 2.5-D orthomosaic.  It intentionally has no GIS or
computer-vision dependency beyond NumPy, making the geometry auditable and
suitable for an embedded reference implementation.

Coordinate conventions
----------------------
* Geodetic coordinates use the WGS84 ellipsoid and ellipsoidal height.
* Local coordinates are ENU: ``(+east, +north, +up)`` in metres.
* Camera rays use the OpenCV pinhole convention: ``(+right, +down, +forward)``.
* A quaternion is Hamilton ``(w, x, y, z)`` and actively rotates a vector from
  the frame named on the right to the frame named on the left.  Consequently,
  ``enu_from_body`` maps body vectors into ENU, and ``body_from_camera`` maps
  camera vectors into the vehicle body frame.

The mapper is deliberately conservative.  A frame is not integrated unless
capture time, calibration identity, pose identity, and safe ground projection
are all known.  This prevents a plausible-looking map from silently encoding
unlocated or poorly located pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, radians, sin, sqrt
from types import MappingProxyType
from typing import Any, Mapping
import threading

import numpy as np


WGS84_SEMI_MAJOR_M = 6_378_137.0
WGS84_FLATTENING = 1.0 / 298.257_223_563
WGS84_SEMI_MINOR_M = WGS84_SEMI_MAJOR_M * (1.0 - WGS84_FLATTENING)
WGS84_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)


def _readonly(value: np.ndarray, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly(value)
    return value


def _finite_scalar(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class GeodeticCoordinate:
    """A WGS84 latitude, longitude, and ellipsoidal height.

    ``ellipsoid_height_m`` is not mean-sea-level altitude.  A geoid correction
    must be applied before constructing this contract when a flight controller
    reports an orthometric/MSL altitude.
    """

    latitude_deg: float
    longitude_deg: float
    ellipsoid_height_m: float
    datum: str = "WGS84"

    def __post_init__(self) -> None:
        latitude = _finite_scalar("latitude_deg", self.latitude_deg)
        longitude = _finite_scalar("longitude_deg", self.longitude_deg)
        height = _finite_scalar("ellipsoid_height_m", self.ellipsoid_height_m)
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("latitude_deg must be within [-90, 90]")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("longitude_deg must be within [-180, 180]")
        if self.datum != "WGS84":
            raise ValueError("the dependency-light reference supports datum='WGS84' only")
        object.__setattr__(self, "latitude_deg", latitude)
        object.__setattr__(self, "longitude_deg", longitude)
        object.__setattr__(self, "ellipsoid_height_m", height)


@dataclass(frozen=True, slots=True)
class LocalENUFrame:
    """Definition of one local tangent plane with an explicit WGS84 origin."""

    origin: GeodeticCoordinate
    frame_id: str = "map_enu"

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id is required")

    @property
    def datum(self) -> str:
        return self.origin.datum

    def to_enu(self, coordinate: GeodeticCoordinate) -> np.ndarray:
        return geodetic_to_enu(coordinate, self)

    def to_geodetic(self, enu_m: np.ndarray) -> GeodeticCoordinate:
        return enu_to_geodetic(enu_m, self)


def _geodetic_to_ecef(coordinate: GeodeticCoordinate) -> np.ndarray:
    latitude = radians(coordinate.latitude_deg)
    longitude = radians(coordinate.longitude_deg)
    sin_latitude = sin(latitude)
    prime_vertical = WGS84_SEMI_MAJOR_M / sqrt(
        1.0 - WGS84_ECCENTRICITY_SQUARED * sin_latitude * sin_latitude
    )
    radius = prime_vertical + coordinate.ellipsoid_height_m
    return np.array(
        [
            radius * cos(latitude) * cos(longitude),
            radius * cos(latitude) * sin(longitude),
            (
                prime_vertical * (1.0 - WGS84_ECCENTRICITY_SQUARED)
                + coordinate.ellipsoid_height_m
            )
            * sin(latitude),
        ],
        dtype=np.float64,
    )


def _ecef_to_geodetic(ecef_m: np.ndarray) -> GeodeticCoordinate:
    x, y, z = np.asarray(ecef_m, dtype=np.float64)
    horizontal = float(np.hypot(x, y))
    longitude = atan2(float(y), float(x))

    # Bowring's closed-form latitude is sub-millimetre accurate for terrestrial
    # coordinates and avoids an iterative dependency.
    if horizontal < 1e-9:
        latitude = np.pi / 2.0 if z >= 0.0 else -np.pi / 2.0
        height = abs(float(z)) - WGS84_SEMI_MINOR_M
    else:
        second_eccentricity_squared = (
            WGS84_SEMI_MAJOR_M**2 - WGS84_SEMI_MINOR_M**2
        ) / WGS84_SEMI_MINOR_M**2
        theta = atan2(
            float(z) * WGS84_SEMI_MAJOR_M,
            horizontal * WGS84_SEMI_MINOR_M,
        )
        latitude = atan2(
            float(z)
            + second_eccentricity_squared
            * WGS84_SEMI_MINOR_M
            * sin(theta) ** 3,
            horizontal
            - WGS84_ECCENTRICITY_SQUARED
            * WGS84_SEMI_MAJOR_M
            * cos(theta) ** 3,
        )
        sin_latitude = sin(latitude)
        prime_vertical = WGS84_SEMI_MAJOR_M / sqrt(
            1.0 - WGS84_ECCENTRICITY_SQUARED * sin_latitude * sin_latitude
        )
        height = horizontal / cos(latitude) - prime_vertical

    longitude_deg = float(np.degrees(longitude))
    # atan2 already returns the requested interval, but avoid a negative zero in
    # metadata and tests.
    if abs(longitude_deg) < 1e-14:
        longitude_deg = 0.0
    return GeodeticCoordinate(
        latitude_deg=float(np.degrees(latitude)),
        longitude_deg=longitude_deg,
        ellipsoid_height_m=float(height),
    )


def _ecef_to_enu_rotation(origin: GeodeticCoordinate) -> np.ndarray:
    latitude = radians(origin.latitude_deg)
    longitude = radians(origin.longitude_deg)
    return np.array(
        [
            [-sin(longitude), cos(longitude), 0.0],
            [
                -sin(latitude) * cos(longitude),
                -sin(latitude) * sin(longitude),
                cos(latitude),
            ],
            [
                cos(latitude) * cos(longitude),
                cos(latitude) * sin(longitude),
                sin(latitude),
            ],
        ],
        dtype=np.float64,
    )


def geodetic_to_enu(
    coordinate: GeodeticCoordinate,
    frame: LocalENUFrame,
) -> np.ndarray:
    """Convert WGS84 geodetic position to metres in an explicit ENU frame."""

    if coordinate.datum != frame.datum:
        raise ValueError("coordinate and ENU origin must use the same datum")
    delta_ecef = _geodetic_to_ecef(coordinate) - _geodetic_to_ecef(frame.origin)
    return _ecef_to_enu_rotation(frame.origin) @ delta_ecef


def enu_to_geodetic(
    enu_m: np.ndarray,
    frame: LocalENUFrame,
) -> GeodeticCoordinate:
    """Convert a three-element local ENU position back to WGS84 geodetic."""

    local = np.asarray(enu_m, dtype=np.float64)
    if local.shape != (3,) or not np.all(np.isfinite(local)):
        raise ValueError("enu_m must be a finite three-element vector")
    ecef = _geodetic_to_ecef(frame.origin) + _ecef_to_enu_rotation(frame.origin).T @ local
    return _ecef_to_geodetic(ecef)


@dataclass(frozen=True, slots=True)
class Quaternion:
    """Unit Hamilton quaternion ``(w, x, y, z)`` for active rotations.

    If a field is called ``a_from_b``, its matrix left-multiplies a vector
    expressed in B and returns the same vector expressed in A.
    """

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        values = np.array([self.w, self.x, self.y, self.z], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("quaternion components must be finite")
        norm = float(np.linalg.norm(values))
        if norm < 1e-12:
            raise ValueError("a zero quaternion cannot define a rotation")
        values /= norm
        # Canonical sign makes equivalent rotations serialize identically.
        if values[0] < 0.0:
            values *= -1.0
        for name, value in zip(("w", "x", "y", "z"), values, strict=True):
            object.__setattr__(self, name, float(value))

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle_rad: float) -> "Quaternion":
        direction = np.asarray(axis, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("axis must be a finite three-element vector")
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            raise ValueError("rotation axis must be non-zero")
        half_angle = 0.5 * _finite_scalar("angle_rad", angle_rad)
        vector = direction / norm * sin(half_angle)
        return cls(cos(half_angle), *vector)

    @classmethod
    def from_rotation_matrix(cls, matrix: np.ndarray) -> "Quaternion":
        rotation = np.asarray(matrix, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation matrix must be finite and 3x3")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-8
        ):
            raise ValueError("rotation matrix must be right-handed and orthonormal")

        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (rotation[2, 1] - rotation[1, 2]) / scale
            y = (rotation[0, 2] - rotation[2, 0]) / scale
            z = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            index = int(np.argmax(np.diag(rotation)))
            if index == 0:
                scale = sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                w = (rotation[2, 1] - rotation[1, 2]) / scale
                x = 0.25 * scale
                y = (rotation[0, 1] + rotation[1, 0]) / scale
                z = (rotation[0, 2] + rotation[2, 0]) / scale
            elif index == 1:
                scale = sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                w = (rotation[0, 2] - rotation[2, 0]) / scale
                x = (rotation[0, 1] + rotation[1, 0]) / scale
                y = 0.25 * scale
                z = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                w = (rotation[1, 0] - rotation[0, 1]) / scale
                x = (rotation[0, 2] + rotation[2, 0]) / scale
                y = (rotation[1, 2] + rotation[2, 1]) / scale
                z = 0.25 * scale
        return cls(w, x, y, z)

    def as_rotation_matrix(self) -> np.ndarray:
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float64)
        if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("vectors must be finite with final dimension three")
        return values @ self.as_rotation_matrix().T


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Calibrated, distortion-corrected pinhole camera intrinsics in pixels."""

    width: int
    height: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    calibration_id: str | None
    calibration_rms_px: float = 0.0
    distortion_model: str = "rectified_pinhole"

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            raw = getattr(self, name)
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError(f"{name} must be a positive integer")
            try:
                numeric_dimension = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a positive integer") from error
            if (
                not np.isfinite(numeric_dimension)
                or not numeric_dimension.is_integer()
                or numeric_dimension <= 0.0
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(numeric_dimension))
        for name in ("fx_px", "fy_px"):
            value = _finite_scalar(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("cx_px", "cy_px"):
            object.__setattr__(
                self, name, _finite_scalar(name, getattr(self, name))
            )
        rms = _finite_scalar("calibration_rms_px", self.calibration_rms_px)
        if rms < 0.0:
            raise ValueError("calibration_rms_px must be non-negative")
        if self.distortion_model != "rectified_pinhole":
            raise ValueError("input imagery must be rectified to the pinhole model")
        if self.calibration_id is not None:
            calibration_id = str(self.calibration_id).strip()
            if not calibration_id:
                raise ValueError("calibration_id cannot be blank")
            object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "calibration_rms_px", rms)

    @property
    def focal_mean_px(self) -> float:
        return sqrt(self.fx_px * self.fy_px)


@dataclass(frozen=True, slots=True)
class VehicleCameraPose:
    """A timestamped camera pose derived from a vehicle navigation solution.

    Body axes are whatever the producer declares; transform names remove the
    ambiguity.  ``camera_position_body_m`` is the body-origin-to-camera-center
    lever arm expressed in body axes.  Standard deviations are 1-sigma values.
    """

    position: GeodeticCoordinate
    enu_from_body: Quaternion
    body_from_camera: Quaternion
    timestamp_tai_ns: int | None
    pose_id: str | None
    camera_position_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_sigma_enu_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    attitude_sigma_rad: float = 0.0
    velocity_enu_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_rate_body_radps: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in (
            "camera_position_body_m",
            "position_sigma_enu_m",
            "velocity_enu_mps",
            "angular_rate_body_radps",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values")
            if name == "position_sigma_enu_m" and any(value < 0.0 for value in values):
                raise ValueError("position standard deviations must be non-negative")
            object.__setattr__(self, name, values)
        attitude_sigma = _finite_scalar("attitude_sigma_rad", self.attitude_sigma_rad)
        if attitude_sigma < 0.0:
            raise ValueError("attitude_sigma_rad must be non-negative")
        object.__setattr__(self, "attitude_sigma_rad", attitude_sigma)
        if self.timestamp_tai_ns is not None and int(self.timestamp_tai_ns) < 0:
            raise ValueError("timestamp_tai_ns must be non-negative when supplied")
        if self.timestamp_tai_ns is not None:
            object.__setattr__(self, "timestamp_tai_ns", int(self.timestamp_tai_ns))

    def camera_origin_enu(self, frame: LocalENUFrame) -> np.ndarray:
        body_origin = geodetic_to_enu(self.position, frame)
        lever_arm = self.enu_from_body.rotate(np.asarray(self.camera_position_body_m))
        return body_origin + lever_arm

    def camera_to_enu_rotation(self) -> np.ndarray:
        return self.enu_from_body.as_rotation_matrix() @ self.body_from_camera.as_rotation_matrix()


@dataclass(frozen=True, slots=True)
class GroundProjection:
    """Projected rays; invalid intersections contain NaNs and a false mask."""

    camera_origin_enu_m: np.ndarray
    rays_enu: np.ndarray
    points_enu_m: np.ndarray
    slant_range_m: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.camera_origin_enu_m, dtype=np.float64)
        rays = np.asarray(self.rays_enu, dtype=np.float64)
        points = np.asarray(self.points_enu_m, dtype=np.float64)
        ranges = np.asarray(self.slant_range_m, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        if origin.shape != (3,) or rays.ndim != 2 or rays.shape[1] != 3:
            raise ValueError("ground projection geometry is malformed")
        if points.shape != rays.shape or ranges.shape != (len(rays),) or valid.shape != ranges.shape:
            raise ValueError("ground projection arrays must share their sample dimension")
        object.__setattr__(self, "camera_origin_enu_m", _readonly(origin))
        object.__setattr__(self, "rays_enu", _readonly(rays))
        object.__setattr__(self, "points_enu_m", _readonly(points))
        object.__setattr__(self, "slant_range_m", _readonly(ranges))
        object.__setattr__(self, "valid", _readonly(valid))


def intersect_ground_plane(
    camera_origin_enu_m: np.ndarray,
    rays_enu: np.ndarray,
    *,
    ground_elevation_enu_m: float = 0.0,
    min_downward_cosine: float = 0.05,
    max_slant_range_m: float | None = None,
) -> GroundProjection:
    """Intersect ENU rays with a horizontal ground plane safely.

    Upward rays, rays behind the camera, and rays closer to the horizon than
    ``min_downward_cosine`` are invalid.  This guard is essential because a tiny
    attitude error near the horizon creates an unbounded ground error.
    """

    origin = np.asarray(camera_origin_enu_m, dtype=np.float64)
    rays = np.asarray(rays_enu, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("camera origin must be a finite three-element ENU vector")
    if rays.ndim == 1:
        rays = rays[None, :]
    if rays.ndim != 2 or rays.shape[1] != 3 or not np.all(np.isfinite(rays)):
        raise ValueError("rays_enu must be a finite Nx3 array")
    ground = _finite_scalar("ground_elevation_enu_m", ground_elevation_enu_m)
    threshold = _finite_scalar("min_downward_cosine", min_downward_cosine)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("min_downward_cosine must be within (0, 1]")
    if max_slant_range_m is not None and max_slant_range_m <= 0.0:
        raise ValueError("max_slant_range_m must be positive when supplied")

    norms = np.linalg.norm(rays, axis=1)
    nonzero = norms > 1e-12
    unit_rays = np.full_like(rays, np.nan)
    unit_rays[nonzero] = rays[nonzero] / norms[nonzero, None]
    downward = -unit_rays[:, 2]
    valid = nonzero & np.isfinite(downward) & (downward >= threshold)

    ranges = np.full(len(rays), np.nan, dtype=np.float64)
    ranges[valid] = (origin[2] - ground) / downward[valid]
    valid &= ranges > 0.0
    if max_slant_range_m is not None:
        valid &= ranges <= float(max_slant_range_m)

    points = np.full_like(unit_rays, np.nan)
    points[valid] = origin + ranges[valid, None] * unit_rays[valid]
    ranges[~valid] = np.nan
    return GroundProjection(origin, unit_rays, points, ranges, valid)


def project_pixels_to_ground(
    pixels_uv: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: VehicleCameraPose,
    frame: LocalENUFrame,
    *,
    ground_elevation_enu_m: float = 0.0,
    min_downward_cosine: float = 0.05,
    max_slant_range_m: float | None = None,
) -> GroundProjection:
    """Project image pixels through camera -> body -> ENU onto the ground."""

    pixels = np.asarray(pixels_uv, dtype=np.float64)
    if pixels.ndim == 1:
        pixels = pixels[None, :]
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.all(np.isfinite(pixels)):
        raise ValueError("pixels_uv must be a finite Nx2 array")

    rays_camera = np.column_stack(
        (
            (pixels[:, 0] - intrinsics.cx_px) / intrinsics.fx_px,
            (pixels[:, 1] - intrinsics.cy_px) / intrinsics.fy_px,
            np.ones(len(pixels), dtype=np.float64),
        )
    )
    rays_enu = rays_camera @ pose.camera_to_enu_rotation().T
    projection = intersect_ground_plane(
        pose.camera_origin_enu(frame),
        rays_enu,
        ground_elevation_enu_m=ground_elevation_enu_m,
        min_downward_cosine=min_downward_cosine,
        max_slant_range_m=max_slant_range_m,
    )
    in_sensor = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= intrinsics.width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= intrinsics.height - 1)
    )
    if np.all(in_sensor):
        return projection
    valid = projection.valid & in_sensor
    points = np.array(projection.points_enu_m, copy=True)
    ranges = np.array(projection.slant_range_m, copy=True)
    points[~valid] = np.nan
    ranges[~valid] = np.nan
    return GroundProjection(
        projection.camera_origin_enu_m,
        projection.rays_enu,
        points,
        ranges,
        valid,
    )


def projected_ground_uncertainty(
    projection: GroundProjection,
    intrinsics: CameraIntrinsics,
    pose: VehicleCameraPose,
    *,
    frame_timestamp_tai_ns: int,
    pixel_sigma_px: np.ndarray | float = 0.0,
    ground_elevation_enu_m: np.ndarray | float = 0.0,
    config: OrthoMosaicConfig | None = None,
) -> np.ndarray:
    """Conservative first-order 1-sigma horizontal projection uncertainty.

    The estimate combines horizontal/vertical position error, timing-induced
    displacement, attitude error at slant range, and calibration/pixel error.
    When a height-field ``config`` is supplied, the same first-visible-surface
    intersection is also evaluated at one-sigma pose/ray perturbations.  This
    catches discontinuous visibility changes at ridge silhouettes and tangent
    rays, where a planar Jacobian can otherwise report false precision.

    It is a gating metric rather than a replacement for a full covariance
    propagation performed by a production navigation filter.
    """

    if pose.timestamp_tai_ns is None:
        raise ValueError("pose time is required for projected uncertainty")
    pixel_sigma = np.asarray(pixel_sigma_px, dtype=np.float64)
    if pixel_sigma.ndim == 0:
        pixel_sigma = np.full(len(projection.valid), float(pixel_sigma), dtype=np.float64)
    if pixel_sigma.shape != (len(projection.valid),):
        raise ValueError("pixel_sigma_px must be scalar or match the number of rays")
    if not np.all(np.isfinite(pixel_sigma)) or np.any(pixel_sigma < 0.0):
        raise ValueError("pixel uncertainty must be finite and non-negative")
    elevation = np.asarray(ground_elevation_enu_m, dtype=np.float64)
    if elevation.ndim == 0:
        elevation = np.full(
            len(projection.valid), float(elevation), dtype=np.float64
        )
    if elevation.shape != (len(projection.valid),):
        raise ValueError(
            "ground_elevation_enu_m must be scalar or match the number of rays"
        )
    if np.any(projection.valid & ~np.isfinite(elevation)):
        raise ValueError("ground elevation must be finite for every valid ray")

    surface_sigma = np.zeros(len(projection.valid), dtype=np.float64)
    if config is not None and config.surface_elevation_enu_m is not None:
        surface_sigma, surface_sigma_valid = _sample_surface_elevation_sigma(
            projection.points_enu_m[:, 0],
            projection.points_enu_m[:, 1],
            config,
        )
        if np.any(projection.valid & ~surface_sigma_valid):
            raise ValueError(
                "surface elevation uncertainty is unknown for a valid intersection"
            )

    time_offset_s = abs(int(frame_timestamp_tai_ns) - pose.timestamp_tai_ns) * 1e-9
    sigma_east, sigma_north, sigma_up = pose.position_sigma_enu_m
    horizontal_pose_sigma = np.hypot(sigma_east, sigma_north)
    horizontal_speed = np.hypot(pose.velocity_enu_mps[0], pose.velocity_enu_mps[1])
    horizontal_pose_sigma = np.hypot(horizontal_pose_sigma, horizontal_speed * time_offset_s)
    angular_rate = float(np.linalg.norm(pose.angular_rate_body_radps))
    angular_sigma = np.hypot(pose.attitude_sigma_rad, angular_rate * time_offset_s)

    result = np.full(len(projection.valid), np.inf, dtype=np.float64)
    valid = projection.valid
    if np.any(valid):
        horizontal_range = np.linalg.norm(
            projection.points_enu_m[valid, :2] - projection.camera_origin_enu_m[:2],
            axis=1,
        )
        vertical_separation = np.abs(
            projection.camera_origin_enu_m[2] - elevation[valid]
        )
        vertical_leverage = horizontal_range / np.maximum(
            vertical_separation, 1e-9
        )
        image_sigma = np.hypot(pixel_sigma[valid], intrinsics.calibration_rms_px)
        image_angular_sigma = image_sigma / intrinsics.focal_mean_px
        vertical_sigma = np.hypot(sigma_up, surface_sigma[valid])
        result[valid] = np.sqrt(
            horizontal_pose_sigma**2
            + (vertical_leverage * vertical_sigma) ** 2
            + (projection.slant_range_m[valid] * angular_sigma) ** 2
            + (projection.slant_range_m[valid] * image_angular_sigma) ** 2
        )
        if config is not None and config.surface_elevation_enu_m is not None:
            if config.shape != np.asarray(config.surface_elevation_enu_m).shape:
                raise ValueError("height-field config geometry is malformed")
            terrain_uncertainty = _terrain_projection_probe_uncertainty(
                projection,
                config,
                sigma_east_m=float(sigma_east),
                sigma_north_m=float(sigma_north),
                sigma_up_m=float(
                    np.hypot(
                        sigma_up,
                        config.surface_elevation_sigma_m
                        if np.ndim(config.surface_elevation_sigma_m) == 0
                        else 0.0,
                    )
                ),
                attitude_sigma_rad=np.hypot(
                    angular_sigma,
                    np.hypot(pixel_sigma, intrinsics.calibration_rms_px)
                    / intrinsics.focal_mean_px,
                ),
            )
            result[valid] = np.maximum(result[valid], terrain_uncertainty[valid])
    return result


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    source = np.asarray(rgb)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 array")
    if np.issubdtype(source.dtype, np.integer):
        maximum = float(np.iinfo(source.dtype).max)
        result = source.astype(np.float32) / maximum
    else:
        result = source.astype(np.float32)
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("rgb values must be finite and normalize to [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class MappingFrame:
    """One image and its optional mapping prerequisites.

    Optional timing/calibration/pose fields are intentional: malformed incoming
    packets can be represented and rejected with an auditable reason instead of
    raising before they reach the mapping safety gate.
    """

    frame_id: str
    timestamp_tai_ns: int | None
    rgb: np.ndarray
    intrinsics: CameraIntrinsics | None
    pose: VehicleCameraPose | None
    thermal_normalized: np.ndarray | None = None
    semantic_class: np.ndarray | None = None
    validity: np.ndarray | None = None
    support: np.ndarray | float = 1.0
    pixel_uncertainty_px: np.ndarray | float = 0.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id is required")
        if self.timestamp_tai_ns is not None:
            if (
                isinstance(self.timestamp_tai_ns, (bool, np.bool_))
                or int(self.timestamp_tai_ns) != self.timestamp_tai_ns
            ):
                raise ValueError("timestamp_tai_ns must be an integer")
            timestamp = int(self.timestamp_tai_ns)
            if timestamp < 0:
                raise ValueError("timestamp_tai_ns must be non-negative when supplied")
            object.__setattr__(self, "timestamp_tai_ns", timestamp)

        rgb = _normalize_rgb(self.rgb)
        height, width = rgb.shape[:2]
        object.__setattr__(self, "rgb", _readonly(rgb))

        if self.thermal_normalized is not None:
            thermal = np.asarray(self.thermal_normalized, dtype=np.float32)
            if thermal.shape != (height, width):
                raise ValueError("thermal_normalized must match RGB geometry")
            finite = np.isfinite(thermal)
            if np.any((thermal[finite] < 0.0) | (thermal[finite] > 1.0)):
                raise ValueError("finite thermal values must be normalized to [0, 1]")
            object.__setattr__(self, "thermal_normalized", _readonly(thermal))

        if self.semantic_class is not None:
            supplied_semantic = np.asarray(self.semantic_class)
            if (
                np.issubdtype(supplied_semantic.dtype, np.bool_)
                or not np.issubdtype(supplied_semantic.dtype, np.integer)
            ):
                raise ValueError("semantic_class values must use an integer dtype")
            integer_limits = np.iinfo(np.int32)
            if supplied_semantic.size and (
                np.min(supplied_semantic) < integer_limits.min
                or np.max(supplied_semantic) > integer_limits.max
            ):
                raise ValueError("semantic_class values must fit signed int32")
            semantic = supplied_semantic.astype(np.int32, copy=False)
            if semantic.shape != (height, width):
                raise ValueError("semantic_class must match RGB geometry")
            object.__setattr__(self, "semantic_class", _readonly(semantic))

        validity = (
            np.ones((height, width), dtype=bool)
            if self.validity is None
            else np.asarray(self.validity, dtype=bool)
        )
        if validity.shape != (height, width):
            raise ValueError("validity must match RGB geometry")
        object.__setattr__(self, "validity", _readonly(validity))

        for name in ("support", "pixel_uncertainty_px"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.ndim == 0:
                value = np.full((height, width), float(value), dtype=np.float32)
            if value.shape != (height, width) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite, scalar, or match RGB geometry")
            if np.any(value < 0.0):
                raise ValueError(f"{name} must be non-negative")
            if name == "support" and np.any(value > 1.0):
                raise ValueError("support must be within [0, 1]")
            object.__setattr__(self, name, _readonly(value))

        object.__setattr__(self, "provenance", _freeze(dict(self.provenance)))


@dataclass(frozen=True, slots=True)
class OrthoMosaicConfig:
    """Geometry and safety policy for a north-up local orthomosaic."""

    east_min_m: float
    east_max_m: float
    north_min_m: float
    north_max_m: float
    resolution_m: float
    ground_elevation_enu_m: float = 0.0
    surface_elevation_enu_m: np.ndarray | None = None
    surface_model_id: str | None = None
    surface_intersection_tolerance_m: float = 0.05
    surface_intersection_max_iterations: int = 20
    surface_ray_step_m: float | None = None
    surface_ray_max_steps: int = 8_192
    surface_ray_max_work: int = 1_000_000
    min_downward_cosine: float = 0.1
    max_slant_range_m: float = 2_000.0
    max_projected_uncertainty_m: float = 5.0
    frame_uncertainty_percentile: float = 95.0
    max_pose_time_offset_ns: int = 100_000_000
    max_cells: int = 1_048_576
    max_semantic_classes: int = 256
    max_peak_memory_bytes: int = 536_870_912
    max_sampled_pixels_per_frame: int = 131_072
    surface_elevation_sigma_m: np.ndarray | float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "east_min_m",
            "east_max_m",
            "north_min_m",
            "north_max_m",
            "resolution_m",
            "ground_elevation_enu_m",
            "surface_intersection_tolerance_m",
            "min_downward_cosine",
            "max_slant_range_m",
            "max_projected_uncertainty_m",
            "frame_uncertainty_percentile",
        ):
            object.__setattr__(self, name, _finite_scalar(name, getattr(self, name)))
        if self.east_max_m <= self.east_min_m or self.north_max_m <= self.north_min_m:
            raise ValueError("mosaic bounds must have positive area")
        if self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive")
        if self.surface_intersection_tolerance_m <= 0.0:
            raise ValueError("surface_intersection_tolerance_m must be positive")
        if self.surface_ray_step_m is not None:
            ray_step = _finite_scalar("surface_ray_step_m", self.surface_ray_step_m)
            if ray_step <= 0.0:
                raise ValueError("surface_ray_step_m must be positive")
            object.__setattr__(self, "surface_ray_step_m", ray_step)
        iterations = int(self.surface_intersection_max_iterations)
        if (
            isinstance(self.surface_intersection_max_iterations, (bool, np.bool_))
            or iterations != self.surface_intersection_max_iterations
            or iterations <= 0
        ):
            raise ValueError("surface_intersection_max_iterations must be positive")
        object.__setattr__(self, "surface_intersection_max_iterations", iterations)
        ray_max_steps = int(self.surface_ray_max_steps)
        if (
            isinstance(self.surface_ray_max_steps, (bool, np.bool_))
            or ray_max_steps != self.surface_ray_max_steps
            or ray_max_steps <= 0
        ):
            raise ValueError("surface_ray_max_steps must be positive")
        object.__setattr__(self, "surface_ray_max_steps", ray_max_steps)
        if not 0.0 < self.min_downward_cosine <= 1.0:
            raise ValueError("min_downward_cosine must be within (0, 1]")
        if self.max_slant_range_m <= 0.0 or self.max_projected_uncertainty_m <= 0.0:
            raise ValueError("range and uncertainty limits must be positive")
        if not 0.0 < self.frame_uncertainty_percentile <= 100.0:
            raise ValueError("frame uncertainty percentile must be within (0, 100]")
        if (
            isinstance(self.max_pose_time_offset_ns, (bool, np.bool_))
            or int(self.max_pose_time_offset_ns) != self.max_pose_time_offset_ns
            or int(self.max_pose_time_offset_ns) < 0
        ):
            raise ValueError("max_pose_time_offset_ns must be non-negative")
        object.__setattr__(self, "max_pose_time_offset_ns", int(self.max_pose_time_offset_ns))

        for name in (
            "max_cells",
            "max_semantic_classes",
            "max_peak_memory_bytes",
            "max_sampled_pixels_per_frame",
            "surface_ray_max_work",
        ):
            value = getattr(self, name)
            try:
                integer_value = int(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a positive integer") from error
            if (
                isinstance(value, (bool, np.bool_))
                or integer_value != value
                or integer_value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, integer_value)

        width_exact = (self.east_max_m - self.east_min_m) / self.resolution_m
        height_exact = (self.north_max_m - self.north_min_m) / self.resolution_m
        if not np.isfinite(width_exact) or not np.isfinite(height_exact):
            raise ValueError("mosaic grid dimensions must be finite")
        if not np.isclose(width_exact, round(width_exact), atol=1e-9) or not np.isclose(
            height_exact, round(height_exact), atol=1e-9
        ):
            raise ValueError("map spans must be integer multiples of resolution_m")
        width = int(round(width_exact))
        height = int(round(height_exact))
        if width * height > self.max_cells:
            raise ValueError(
                f"mosaic requires {width * height} cells, exceeding max_cells={self.max_cells}"
            )
        baseline_bytes = self.estimated_peak_memory_bytes(semantic_class_count=0)
        if baseline_bytes > self.max_peak_memory_bytes:
            raise ValueError(
                "mosaic's conservative peak-memory estimate "
                f"({baseline_bytes} bytes) exceeds max_peak_memory_bytes="
                f"{self.max_peak_memory_bytes}"
            )
        if self.surface_elevation_enu_m is not None:
            surface = np.asarray(self.surface_elevation_enu_m, dtype=np.float64)
            expected_shape = (height, width)
            if surface.shape != expected_shape:
                raise ValueError(
                    "surface_elevation_enu_m must match the north-up map grid shape"
                )
            if not np.any(np.isfinite(surface)):
                raise ValueError("surface_elevation_enu_m has no finite elevations")
            if not self.surface_model_id or not self.surface_model_id.strip():
                raise ValueError("a surface_model_id is required with a height field")
            object.__setattr__(self, "surface_elevation_enu_m", _readonly(surface))
            object.__setattr__(self, "surface_model_id", self.surface_model_id.strip())
        elif self.surface_model_id is not None:
            raise ValueError("surface_model_id requires surface_elevation_enu_m")

        surface_sigma = np.asarray(self.surface_elevation_sigma_m, dtype=np.float64)
        if surface_sigma.ndim == 0:
            sigma_value = float(surface_sigma)
            if not np.isfinite(sigma_value) or sigma_value < 0.0:
                raise ValueError("surface_elevation_sigma_m must be finite and non-negative")
            object.__setattr__(self, "surface_elevation_sigma_m", sigma_value)
        else:
            if self.surface_elevation_enu_m is None:
                raise ValueError(
                    "an array surface_elevation_sigma_m requires a height field"
                )
            if surface_sigma.shape != (height, width):
                raise ValueError(
                    "surface_elevation_sigma_m must be scalar or match the map grid shape"
                )
            if not np.all(np.isfinite(surface_sigma)) or np.any(surface_sigma < 0.0):
                raise ValueError(
                    "surface_elevation_sigma_m must be finite and non-negative"
                )
            object.__setattr__(
                self, "surface_elevation_sigma_m", _readonly(surface_sigma)
            )

    @property
    def width(self) -> int:
        return int(round((self.east_max_m - self.east_min_m) / self.resolution_m))

    @property
    def height(self) -> int:
        return int(round((self.north_max_m - self.north_min_m) / self.resolution_m))

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def estimated_peak_memory_bytes(self, semantic_class_count: int) -> int:
        """Return the conservative dense-grid peak used by allocation gates."""

        class_count = int(semantic_class_count)
        if class_count < 0:
            raise ValueError("semantic_class_count must be non-negative")
        # Fixed arrays, a full splat scratch grid, and an owned snapshot are
        # budgeted at 192 bytes/cell. Each active semantic class adds its
        # float64 accumulator and float32 exported evidence layer. The second
        # term reserves vectorized projection/splat scratch for the largest
        # permitted source frame so grid and frame caps form one peak budget.
        grid_bytes = self.width * self.height * (192 + 12 * class_count)
        frame_scratch_bytes = self.max_sampled_pixels_per_frame * 384
        return grid_bytes + frame_scratch_bytes


def _sample_surface_elevation(
    east_m: np.ndarray,
    north_m: np.ndarray,
    config: OrthoMosaicConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample a north-up height field without filling unknown cells."""

    if config.surface_elevation_enu_m is None:
        raise ValueError("surface_elevation_enu_m is required")
    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    if east.shape != north.shape:
        raise ValueError("east_m and north_m must have the same shape")
    column = (east - config.east_min_m) / config.resolution_m - 0.5
    row = (config.north_max_m - north) / config.resolution_m - 0.5
    inside = (
        np.isfinite(column)
        & np.isfinite(row)
        & (column >= 0.0)
        & (column <= config.width - 1)
        & (row >= 0.0)
        & (row <= config.height - 1)
    )
    safe_column = np.clip(np.where(inside, column, 0.0), 0.0, config.width - 1)
    safe_row = np.clip(np.where(inside, row, 0.0), 0.0, config.height - 1)
    column_zero = np.floor(safe_column).astype(np.int64)
    row_zero = np.floor(safe_row).astype(np.int64)
    column_one = np.minimum(column_zero + 1, config.width - 1)
    row_one = np.minimum(row_zero + 1, config.height - 1)
    delta_column = safe_column - column_zero
    delta_row = safe_row - row_zero
    neighbours = (
        (row_zero, column_zero, (1.0 - delta_row) * (1.0 - delta_column)),
        (row_zero, column_one, (1.0 - delta_row) * delta_column),
        (row_one, column_zero, delta_row * (1.0 - delta_column)),
        (row_one, column_one, delta_row * delta_column),
    )
    values = np.zeros(east.shape, dtype=np.float64)
    available_weight = np.zeros(east.shape, dtype=np.float64)
    surface = np.asarray(config.surface_elevation_enu_m)
    for sample_row, sample_column, weight in neighbours:
        elevation = surface[sample_row, sample_column]
        finite = np.isfinite(elevation)
        values += np.where(finite, elevation, 0.0) * weight
        available_weight += finite * weight
    valid = inside & (available_weight >= 1.0 - 1e-9)
    result = np.full(east.shape, np.nan, dtype=np.float64)
    result[valid] = values[valid] / available_weight[valid]
    return result, valid


def sample_surface_elevation(
    east_m: np.ndarray | float,
    north_m: np.ndarray | float,
    config: OrthoMosaicConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample the configured DEM with an explicit validity mask.

    Unknown cells and coordinates outside the grid return ``NaN`` and false;
    they are never extrapolated or replaced by the flat-ground elevation.
    Returned arrays are owned and read-only so callers cannot mutate mapper
    state through this query API.
    """

    values, valid = _sample_surface_elevation(
        np.asarray(east_m, dtype=np.float64),
        np.asarray(north_m, dtype=np.float64),
        config,
    )
    return _readonly(values), _readonly(valid)


def _sample_surface_elevation_sigma(
    east_m: np.ndarray,
    north_m: np.ndarray,
    config: OrthoMosaicConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample declared 1-sigma DEM elevation error at projected points."""

    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    if east.shape != north.shape:
        raise ValueError("east_m and north_m must have the same shape")
    finite_coordinates = np.isfinite(east) & np.isfinite(north)
    surface_sigma = np.asarray(config.surface_elevation_sigma_m, dtype=np.float64)
    if surface_sigma.ndim == 0:
        result = np.full(east.shape, np.nan, dtype=np.float64)
        result[finite_coordinates] = float(surface_sigma)
        return result, finite_coordinates

    column = (east - config.east_min_m) / config.resolution_m - 0.5
    row = (config.north_max_m - north) / config.resolution_m - 0.5
    inside = (
        finite_coordinates
        & (column >= 0.0)
        & (column <= config.width - 1)
        & (row >= 0.0)
        & (row <= config.height - 1)
    )
    safe_column = np.clip(np.where(inside, column, 0.0), 0.0, config.width - 1)
    safe_row = np.clip(np.where(inside, row, 0.0), 0.0, config.height - 1)
    column_zero = np.floor(safe_column).astype(np.int64)
    row_zero = np.floor(safe_row).astype(np.int64)
    column_one = np.minimum(column_zero + 1, config.width - 1)
    row_one = np.minimum(row_zero + 1, config.height - 1)
    delta_column = safe_column - column_zero
    delta_row = safe_row - row_zero
    result = np.full(east.shape, np.nan, dtype=np.float64)
    interpolated = (
        surface_sigma[row_zero, column_zero]
        * (1.0 - delta_row)
        * (1.0 - delta_column)
        + surface_sigma[row_zero, column_one]
        * (1.0 - delta_row)
        * delta_column
        + surface_sigma[row_one, column_zero]
        * delta_row
        * (1.0 - delta_column)
        + surface_sigma[row_one, column_one] * delta_row * delta_column
    )
    result[inside] = interpolated[inside]
    return result, inside


def intersect_height_field(
    camera_origin_enu_m: np.ndarray,
    rays_enu: np.ndarray,
    config: OrthoMosaicConfig,
) -> GroundProjection:
    """Intersect rays with an explicit 2.5-D elevation field.

    A bounded nearest-hit march brackets the first visible crossing and then
    refines it by bisection.  Any ray that leaves the known DEM, exceeds the
    work/range gates, or fails to converge is rejected rather than snapped to
    a plausible cell.
    Survey products should still use multi-view reconstruction and bundle
    adjustment instead of this latency-oriented approximation.
    """

    if config.surface_elevation_enu_m is None:
        raise ValueError("surface_elevation_enu_m is required")
    origin = np.asarray(camera_origin_enu_m, dtype=np.float64)
    rays = np.asarray(rays_enu, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("camera origin must be a finite three-element ENU vector")
    if rays.ndim == 1:
        rays = rays[None, :]
    if rays.ndim != 2 or rays.shape[1] != 3 or not np.all(np.isfinite(rays)):
        raise ValueError("rays_enu must be a finite Nx3 array")

    norms = np.linalg.norm(rays, axis=1)
    unit_rays = np.full_like(rays, np.nan)
    nonzero = norms > 1e-12
    unit_rays[nonzero] = rays[nonzero] / norms[nonzero, None]
    downward = -unit_rays[:, 2]
    valid = nonzero & np.isfinite(downward) & (
        downward >= config.min_downward_cosine
    )

    finite_surface = np.asarray(config.surface_elevation_enu_m)[
        np.isfinite(config.surface_elevation_enu_m)
    ]
    minimum_height = float(np.min(finite_surface))
    # March from the camera rather than jumping directly to the highest known
    # elevation. The shortcut would skip intervening nodata, which cannot be
    # assumed transparent because it may conceal an occluding surface.
    start_range = np.zeros(len(rays), dtype=np.float64)
    end_range = np.full(len(rays), np.nan, dtype=np.float64)
    end_range[valid] = np.minimum(
        config.max_slant_range_m,
        (origin[2] - minimum_height) / downward[valid],
    )
    valid &= np.isfinite(end_range) & (end_range >= start_range)

    step_m = (
        config.surface_ray_step_m
        if config.surface_ray_step_m is not None
        else max(
            config.resolution_m * 0.5,
            config.surface_intersection_tolerance_m,
        )
    )
    march_span = np.where(
        valid, np.maximum(end_range - start_range, 0.0), 0.0
    )
    required_steps = np.ceil(march_span / step_m).astype(np.int64)
    valid &= required_steps <= config.surface_ray_max_steps
    required_work = int(np.sum(required_steps[valid] + 1, dtype=np.int64))
    if required_work > config.surface_ray_max_work:
        # One call is bounded; the uncertainty routine makes at most fourteen
        # additional calls, so frame-level terrain work is bounded as well.
        valid[:] = False

    previous_range = np.full(len(rays), np.nan, dtype=np.float64)
    previous_residual = np.full(len(rays), np.nan, dtype=np.float64)
    previous_previous_range = np.full(len(rays), np.nan, dtype=np.float64)
    previous_previous_residual = np.full(len(rays), np.nan, dtype=np.float64)
    bracket_low = np.full(len(rays), np.nan, dtype=np.float64)
    bracket_high = np.full(len(rays), np.nan, dtype=np.float64)
    maximum_steps = int(np.max(required_steps[valid], initial=0)) + 1
    for step_index in range(maximum_steps):
        active = valid & ~np.isfinite(bracket_high) & (step_index <= required_steps)
        indexes = np.flatnonzero(active)
        if len(indexes) == 0:
            continue
        sample_range = np.minimum(
            start_range[indexes] + step_index * step_m,
            end_range[indexes],
        )
        sample_points = origin + sample_range[:, None] * unit_rays[indexes]
        elevation, known = _sample_surface_elevation(
            sample_points[:, 0], sample_points[:, 1], config
        )
        residual = sample_points[:, 2] - elevation

        # Unknown terrain before the first hit is a possible occluder, not an
        # empty tunnel. Permanently invalidate the ray instead of resuming on a
        # farther known patch and manufacturing line of sight through nodata.
        unknown_indexes = indexes[~known]
        valid[unknown_indexes] = False
        previous_range[unknown_indexes] = np.nan
        previous_residual[unknown_indexes] = np.nan
        known_indexes = indexes[known]
        known_range = sample_range[known]
        known_residual = residual[known]
        if len(known_indexes) == 0:
            continue

        exact = np.abs(known_residual) <= config.surface_intersection_tolerance_m
        bracket_low[known_indexes[exact]] = known_range[exact]
        bracket_high[known_indexes[exact]] = known_range[exact]
        remaining_indexes = known_indexes[~exact]
        remaining_range = known_range[~exact]
        remaining_residual = known_residual[~exact]
        if len(remaining_indexes) == 0:
            continue
        has_previous = np.isfinite(previous_range[remaining_indexes])
        crossed = (
            has_previous
            & (previous_residual[remaining_indexes] > 0.0)
            & (remaining_residual < 0.0)
        )
        crossed_indexes = remaining_indexes[crossed]
        bracket_low[crossed_indexes] = previous_range[crossed_indexes]
        bracket_high[crossed_indexes] = remaining_range[crossed]

        # A ridge can be tangent to a ray: the residual touches zero and rises
        # again without ever changing sign.  Fixed-step sign tests miss that
        # physically valid first hit and may incorrectly select terrain behind
        # it.  When three positive samples bracket a local minimum, refine the
        # minimum.  If it dips below the surface, retain the *entry* bracket;
        # if it merely touches, retain the tangent point.
        positive = (remaining_residual > 0.0) & ~crossed
        positive_indexes = remaining_indexes[positive]
        if len(positive_indexes):
            local_minimum = (
                np.isfinite(previous_previous_range[positive_indexes])
                & (previous_residual[positive_indexes]
                   <= previous_previous_residual[positive_indexes])
                & (previous_residual[positive_indexes]
                   <= remaining_residual[positive])
            )
            tangent_indexes = positive_indexes[local_minimum]
            if len(tangent_indexes):
                low = previous_previous_range[tangent_indexes].copy()
                high = remaining_range[positive][local_minimum].copy()
                known_minimum = np.ones(len(tangent_indexes), dtype=bool)
                for _ in range(config.surface_intersection_max_iterations):
                    span = high - low
                    first = low + span / 3.0
                    second = high - span / 3.0
                    first_points = origin + first[:, None] * unit_rays[tangent_indexes]
                    second_points = origin + second[:, None] * unit_rays[tangent_indexes]
                    first_height, first_known = _sample_surface_elevation(
                        first_points[:, 0], first_points[:, 1], config
                    )
                    second_height, second_known = _sample_surface_elevation(
                        second_points[:, 0], second_points[:, 1], config
                    )
                    known_minimum &= first_known & second_known
                    first_residual = first_points[:, 2] - first_height
                    second_residual = second_points[:, 2] - second_height
                    choose_left = first_residual <= second_residual
                    high = np.where(choose_left, second, high)
                    low = np.where(choose_left, low, first)

                minimum_range = 0.5 * (low + high)
                minimum_points = (
                    origin + minimum_range[:, None] * unit_rays[tangent_indexes]
                )
                minimum_height, minimum_known = _sample_surface_elevation(
                    minimum_points[:, 0], minimum_points[:, 1], config
                )
                minimum_residual = minimum_points[:, 2] - minimum_height
                usable = known_minimum & minimum_known & (
                    minimum_residual <= config.surface_intersection_tolerance_m
                )
                touching = usable & (
                    minimum_residual >= -config.surface_intersection_tolerance_m
                )
                touching_indexes = tangent_indexes[touching]
                bracket_low[touching_indexes] = minimum_range[touching]
                bracket_high[touching_indexes] = minimum_range[touching]

                penetrated = usable & ~touching
                penetrated_indexes = tangent_indexes[penetrated]
                bracket_low[penetrated_indexes] = previous_previous_range[
                    penetrated_indexes
                ]
                bracket_high[penetrated_indexes] = minimum_range[penetrated]

        update = (remaining_residual > 0.0) & ~np.isfinite(
            bracket_high[remaining_indexes]
        )
        update_indexes = remaining_indexes[update]
        previous_previous_range[update_indexes] = previous_range[update_indexes]
        previous_previous_residual[update_indexes] = previous_residual[update_indexes]
        previous_range[update_indexes] = remaining_range[update]
        previous_residual[update_indexes] = remaining_residual[update]

    bracketed = valid & np.isfinite(bracket_high)
    valid &= bracketed
    refine = np.flatnonzero(valid & (bracket_high > bracket_low))
    for _ in range(config.surface_intersection_max_iterations):
        if len(refine) == 0:
            break
        middle = 0.5 * (bracket_low[refine] + bracket_high[refine])
        sample_points = origin + middle[:, None] * unit_rays[refine]
        elevation, known = _sample_surface_elevation(
            sample_points[:, 0], sample_points[:, 1], config
        )
        valid[refine[~known]] = False
        refine = refine[known]
        middle = middle[known]
        if len(refine) == 0:
            break
        residual = sample_points[known, 2] - elevation[known]
        above = residual > 0.0
        bracket_low[refine[above]] = middle[above]
        bracket_high[refine[~above]] = middle[~above]

    slant_range = 0.5 * (bracket_low + bracket_high)
    points = np.full_like(unit_rays, np.nan)
    active = np.flatnonzero(valid)
    if len(active):
        candidates = origin + slant_range[active, None] * unit_rays[active]
        final_height, known = _sample_surface_elevation(
            candidates[:, 0], candidates[:, 1], config
        )
        residual = np.abs(candidates[:, 2] - final_height)
        converged = known & (
            residual <= config.surface_intersection_tolerance_m
        )
        valid[active[~converged]] = False
        points[active[converged]] = candidates[converged]
    slant_range[~valid] = np.nan
    return GroundProjection(origin, unit_rays, points, slant_range, valid)


def project_pixels_to_height_field(
    pixels_uv: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: VehicleCameraPose,
    frame: LocalENUFrame,
    config: OrthoMosaicConfig,
) -> GroundProjection:
    """Project image pixels through camera -> body -> ENU onto a height field."""

    pixels = np.asarray(pixels_uv, dtype=np.float64)
    if pixels.ndim == 1:
        pixels = pixels[None, :]
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.all(np.isfinite(pixels)):
        raise ValueError("pixels_uv must be a finite Nx2 array")
    rays_camera = np.column_stack(
        (
            (pixels[:, 0] - intrinsics.cx_px) / intrinsics.fx_px,
            (pixels[:, 1] - intrinsics.cy_px) / intrinsics.fy_px,
            np.ones(len(pixels), dtype=np.float64),
        )
    )
    rays_enu = rays_camera @ pose.camera_to_enu_rotation().T
    projection = intersect_height_field(
        pose.camera_origin_enu(frame), rays_enu, config
    )
    in_sensor = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= intrinsics.width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= intrinsics.height - 1)
    )
    if np.all(in_sensor):
        return projection
    valid = projection.valid & in_sensor
    points = np.array(projection.points_enu_m, copy=True)
    ranges = np.array(projection.slant_range_m, copy=True)
    points[~valid] = np.nan
    ranges[~valid] = np.nan
    return GroundProjection(
        projection.camera_origin_enu_m,
        projection.rays_enu,
        points,
        ranges,
        valid,
    )


def _rotate_rays_about_axes(
    rays: np.ndarray,
    axes: np.ndarray,
    angle_rad: np.ndarray,
) -> np.ndarray:
    """Vectorized Rodrigues rotation for paired ENU rays and axes."""

    cosine = np.cos(angle_rad)[:, None]
    sine = np.sin(angle_rad)[:, None]
    return (
        rays * cosine
        + np.cross(axes, rays) * sine
        + axes * np.sum(axes * rays, axis=1)[:, None] * (1.0 - cosine)
    )


def _terrain_projection_probe_uncertainty(
    nominal: GroundProjection,
    config: OrthoMosaicConfig,
    *,
    sigma_east_m: float,
    sigma_north_m: float,
    sigma_up_m: float,
    attitude_sigma_rad: np.ndarray,
) -> np.ndarray:
    """Bound terrain projection error with one-sigma visibility probes.

    A height field makes the image-to-map transform piecewise continuous: an
    arbitrarily small pose change can move the first visible hit from one ridge
    to another.  Local planar covariance propagation cannot represent that
    branch change.  These probes therefore return infinity whenever a nominally
    valid ray loses its intersection, and otherwise retain the largest observed
    horizontal displacement across translation and angular perturbations.
    """

    ray_count = len(nominal.valid)
    angular_sigma = np.asarray(attitude_sigma_rad, dtype=np.float64)
    if angular_sigma.shape != (ray_count,):
        raise ValueError("attitude_sigma_rad must match the number of rays")
    if not np.all(np.isfinite(angular_sigma)) or np.any(angular_sigma < 0.0):
        raise ValueError("attitude_sigma_rad must be finite and non-negative")

    result = np.full(ray_count, np.inf, dtype=np.float64)
    nominal_valid = np.asarray(nominal.valid, dtype=bool)
    result[nominal_valid] = 0.0

    def incorporate(perturbed: GroundProjection) -> None:
        lost = nominal_valid & ~perturbed.valid
        result[lost] = np.inf
        comparable = nominal_valid & perturbed.valid & np.isfinite(result)
        if np.any(comparable):
            displacement = np.linalg.norm(
                perturbed.points_enu_m[comparable, :2]
                - nominal.points_enu_m[comparable, :2],
                axis=1,
            )
            result[comparable] = np.maximum(result[comparable], displacement)

    origin = nominal.camera_origin_enu_m
    for axis_index, sigma in enumerate(
        (sigma_east_m, sigma_north_m, sigma_up_m)
    ):
        sigma = _finite_scalar("terrain pose probe sigma", sigma)
        if sigma < 0.0:
            raise ValueError("terrain pose probe sigma must be non-negative")
        if sigma == 0.0:
            continue
        for sign in (-1.0, 1.0):
            perturbed_origin = np.array(origin, copy=True)
            perturbed_origin[axis_index] += sign * sigma
            incorporate(
                intersect_height_field(perturbed_origin, nominal.rays_enu, config)
            )

    angular_active = nominal_valid & (angular_sigma > 0.0)
    if np.any(angular_active):
        rays = np.asarray(nominal.rays_enu, dtype=np.float64)
        # Two orthogonal axes span every first-order pointing-error direction.
        # Diagonal combinations additionally catch ridges not aligned with ENU.
        first_axis = np.cross(rays, np.array([0.0, 0.0, 1.0]))
        degenerate = np.linalg.norm(first_axis, axis=1) < 1e-9
        first_axis[degenerate] = np.cross(
            rays[degenerate], np.array([1.0, 0.0, 0.0])
        )
        first_axis /= np.linalg.norm(first_axis, axis=1)[:, None]
        second_axis = np.cross(rays, first_axis)
        second_axis /= np.linalg.norm(second_axis, axis=1)[:, None]
        inverse_sqrt_two = 1.0 / sqrt(2.0)
        probe_axes = (
            first_axis,
            second_axis,
            (first_axis + second_axis) * inverse_sqrt_two,
            (first_axis - second_axis) * inverse_sqrt_two,
        )
        for axes in probe_axes:
            for sign in (-1.0, 1.0):
                perturbed_rays = _rotate_rays_about_axes(
                    rays, axes, sign * angular_sigma
                )
                incorporate(intersect_height_field(origin, perturbed_rays, config))

    return result


@dataclass(frozen=True, slots=True)
class FrameIntegrationResult:
    frame_id: str
    accepted: bool
    reason: str
    sampled_pixels: int = 0
    safe_intersections: int = 0
    integrated_samples: int = 0
    mean_projected_uncertainty_m: float | None = None
    max_projected_uncertainty_m: float | None = None


@dataclass(frozen=True, slots=True)
class MosaicSnapshot:
    """Immutable array snapshot of the accumulated 2.5-D evidence map."""

    rgb: np.ndarray
    thermal_normalized: np.ndarray
    height_enu_m: np.ndarray
    semantic_class: np.ndarray
    semantic_confidence: np.ndarray
    coverage_count: np.ndarray
    support: np.ndarray
    weight_sum: np.ndarray
    projected_uncertainty_m: np.ndarray
    valid: np.ndarray
    dominant_source_index: np.ndarray
    class_evidence: Mapping[int, np.ndarray]
    source_ids: tuple[str, ...]
    provenance: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]
    thermal_weight_sum: np.ndarray | None = None
    semantic_weight_sum: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "rgb",
            "thermal_normalized",
            "height_enu_m",
            "semantic_class",
            "semantic_confidence",
            "coverage_count",
            "support",
            "weight_sum",
            "projected_uncertainty_m",
            "valid",
            "dominant_source_index",
        ):
            object.__setattr__(self, name, _readonly(np.asarray(getattr(self, name))))
        object.__setattr__(
            self,
            "class_evidence",
            MappingProxyType(
                {
                    int(class_id): _readonly(np.asarray(evidence, dtype=np.float32))
                    for class_id, evidence in sorted(self.class_evidence.items())
                }
            ),
        )
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(self, "provenance", _freeze(dict(self.provenance)))
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))
        thermal_weight = self.thermal_weight_sum
        if thermal_weight is None:
            thermal_weight = np.isfinite(self.thermal_normalized).astype(np.float32)
        semantic_weight = self.semantic_weight_sum
        if semantic_weight is None:
            semantic_weight = np.zeros(self.semantic_class.shape, dtype=np.float32)
            for evidence in self.class_evidence.values():
                semantic_weight += evidence
        for name, value in (
            ("thermal_weight_sum", thermal_weight),
            ("semantic_weight_sum", semantic_weight),
        ):
            array = np.asarray(value, dtype=np.float32)
            if array.shape != self.weight_sum.shape:
                raise ValueError(f"{name} must match weight_sum geometry")
            if not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, _readonly(array))

    @property
    def rgb_weight_sum(self) -> np.ndarray:
        """RGB evidence weight; alias of the legacy aggregate ``weight_sum`` layer."""

        return self.weight_sum

    @property
    def rgb_support(self) -> np.ndarray:
        """Bounded RGB support; alias of the legacy aggregate ``support`` layer."""

        return self.support

    @property
    def thermal_support(self) -> np.ndarray:
        return _readonly((-np.expm1(-self.thermal_weight_sum)).astype(np.float32))

    @property
    def semantic_support(self) -> np.ndarray:
        return _readonly((-np.expm1(-self.semantic_weight_sum)).astype(np.float32))

    @property
    def array_layers(self) -> Mapping[str, np.ndarray]:
        """Named immutable arrays suitable for an AI model or renderer."""

        layers = {
                "rgb": self.rgb,
                "thermal_normalized": self.thermal_normalized,
                "height_enu_m": self.height_enu_m,
                "semantic_class": self.semantic_class,
                "semantic_confidence": self.semantic_confidence,
                "coverage_count": self.coverage_count,
                "support": self.support,
                "weight_sum": self.weight_sum,
                "projected_uncertainty_m": self.projected_uncertainty_m,
                "valid": self.valid,
                "dominant_source_index": self.dominant_source_index,
                "rgb_weight_sum": self.rgb_weight_sum,
                "rgb_support": self.rgb_support,
                "thermal_weight_sum": self.thermal_weight_sum,
                "thermal_support": self.thermal_support,
                "semantic_weight_sum": self.semantic_weight_sum,
                "semantic_support": self.semantic_support,
            }
        for class_id, evidence in self.class_evidence.items():
            layers[f"semantic_class_evidence_{class_id}"] = evidence
        return MappingProxyType(layers)


class OrthoMosaicGrid:
    """Deterministic uncertainty-aware weighted-splat orthomosaic accumulator."""

    def __init__(self, frame: LocalENUFrame, config: OrthoMosaicConfig) -> None:
        self.frame = frame
        self.config = config
        shape = config.shape
        self._rgb_sum = np.zeros((*shape, 3), dtype=np.float64)
        self._thermal_sum = np.zeros(shape, dtype=np.float64)
        self._thermal_weight = np.zeros(shape, dtype=np.float64)
        self._height_sum = np.zeros(shape, dtype=np.float64)
        self._uncertainty_sum = np.zeros(shape, dtype=np.float64)
        self._weight_sum = np.zeros(shape, dtype=np.float64)
        self._coverage_count = np.zeros(shape, dtype=np.uint32)
        self._semantic_evidence: dict[int, np.ndarray] = {}
        self._dominant_source_weight = np.zeros(shape, dtype=np.float64)
        self._dominant_source_id = np.full(shape, "", dtype=object)
        self._frame_provenance: dict[str, Mapping[str, Any]] = {}
        self._accepted_results: list[FrameIntegrationResult] = []
        self._rejected_results: list[FrameIntegrationResult] = []
        self._lock = threading.RLock()
        self._integration_lock = threading.Lock()

    def _reject(self, frame_id: str, reason: str, **counts: Any) -> FrameIntegrationResult:
        result = FrameIntegrationResult(frame_id=frame_id, accepted=False, reason=reason, **counts)
        with self._lock:
            self._rejected_results.append(result)
        return result

    def integrate(self, source: MappingFrame, *, sample_stride: int = 1) -> FrameIntegrationResult:
        """Safely integrate one frame, returning an auditable accept/reject result."""

        # Projection is intentionally serialized with commit.  Besides protecting
        # NumPy accumulators, this makes the duplicate-id check and frame commit one
        # atomic operation, so concurrent delivery cannot count a frame twice.
        with self._integration_lock:
            return self._integrate_serial(source, sample_stride=sample_stride)

    def _integrate_serial(
        self, source: MappingFrame, *, sample_stride: int = 1
    ) -> FrameIntegrationResult:
        """Integrate while ``_integration_lock`` is held."""

        if int(sample_stride) != sample_stride or int(sample_stride) <= 0:
            raise ValueError("sample_stride must be a positive integer")
        stride = int(sample_stride)
        if source.timestamp_tai_ns is None:
            return self._reject(source.frame_id, "unknown_capture_time")
        if source.intrinsics is None or not source.intrinsics.calibration_id:
            return self._reject(source.frame_id, "unknown_camera_calibration")
        if source.pose is None or not source.pose.pose_id:
            return self._reject(source.frame_id, "unknown_vehicle_camera_pose")
        if source.pose.timestamp_tai_ns is None:
            return self._reject(source.frame_id, "unknown_pose_time")
        if source.frame_id in self._frame_provenance:
            return self._reject(source.frame_id, "duplicate_frame_id")

        intrinsics = source.intrinsics
        pose = source.pose
        if source.rgb.shape[:2] != (intrinsics.height, intrinsics.width):
            return self._reject(source.frame_id, "calibration_geometry_mismatch")
        time_offset = abs(source.timestamp_tai_ns - pose.timestamp_tai_ns)
        if time_offset > self.config.max_pose_time_offset_ns:
            return self._reject(source.frame_id, "pose_time_offset_exceeds_limit")

        sampled_count = (
            (intrinsics.height + stride - 1) // stride
        ) * ((intrinsics.width + stride - 1) // stride)
        if sampled_count > self.config.max_sampled_pixels_per_frame:
            return self._reject(
                source.frame_id,
                "sampled_pixel_limit_exceeded",
                sampled_pixels=sampled_count,
            )
        rows = np.arange(0, intrinsics.height, stride, dtype=np.int64)
        columns = np.arange(0, intrinsics.width, stride, dtype=np.int64)
        grid_u, grid_v = np.meshgrid(columns, rows)
        sampled_rows = grid_v.ravel()
        sampled_columns = grid_u.ravel()
        pixels = np.column_stack((sampled_columns, sampled_rows)).astype(np.float64)
        sampled_count = len(pixels)

        if self.config.surface_elevation_enu_m is None:
            projection = project_pixels_to_ground(
                pixels,
                intrinsics,
                pose,
                self.frame,
                ground_elevation_enu_m=self.config.ground_elevation_enu_m,
                min_downward_cosine=self.config.min_downward_cosine,
                max_slant_range_m=self.config.max_slant_range_m,
            )
        else:
            projection = project_pixels_to_height_field(
                pixels,
                intrinsics,
                pose,
                self.frame,
                self.config,
            )
        safe_count = int(np.count_nonzero(projection.valid))
        if safe_count == 0:
            return self._reject(
                source.frame_id,
                "no_safe_ground_intersections",
                sampled_pixels=sampled_count,
            )

        sampled_pixel_sigma = np.asarray(source.pixel_uncertainty_px)[
            sampled_rows, sampled_columns
        ]
        uncertainty = projected_ground_uncertainty(
            projection,
            intrinsics,
            pose,
            frame_timestamp_tai_ns=source.timestamp_tai_ns,
            pixel_sigma_px=sampled_pixel_sigma,
            ground_elevation_enu_m=(
                projection.points_enu_m[:, 2]
                if self.config.surface_elevation_enu_m is not None
                else self.config.ground_elevation_enu_m
            ),
            config=(
                self.config
                if self.config.surface_elevation_enu_m is not None
                else None
            ),
        )
        safe_uncertainty = uncertainty[projection.valid]
        frame_uncertainty = float(
            np.percentile(safe_uncertainty, self.config.frame_uncertainty_percentile)
        )
        maximum_uncertainty = float(np.max(safe_uncertainty))
        mean_uncertainty = float(np.mean(safe_uncertainty))
        result_counts = {
            "sampled_pixels": sampled_count,
            "safe_intersections": safe_count,
            "mean_projected_uncertainty_m": mean_uncertainty,
            "max_projected_uncertainty_m": maximum_uncertainty,
        }
        if frame_uncertainty > self.config.max_projected_uncertainty_m:
            return self._reject(
                source.frame_id,
                "projected_uncertainty_exceeds_limit",
                **result_counts,
            )

        sampled_validity = np.asarray(source.validity)[sampled_rows, sampled_columns]
        sampled_support = np.asarray(source.support)[sampled_rows, sampled_columns]
        candidate = (
            projection.valid
            & sampled_validity
            & (sampled_support > 0.0)
            & (uncertainty <= self.config.max_projected_uncertainty_m)
        )
        if not np.any(candidate):
            return self._reject(
                source.frame_id,
                "no_supported_samples_after_uncertainty_gate",
                **result_counts,
            )

        source_indexes = np.flatnonzero(candidate)
        east = projection.points_enu_m[source_indexes, 0]
        north = projection.points_enu_m[source_indexes, 1]
        continuous_column = (
            (east - self.config.east_min_m) / self.config.resolution_m - 0.5
        )
        continuous_row = (
            (self.config.north_max_m - north) / self.config.resolution_m - 0.5
        )
        column_zero = np.floor(continuous_column).astype(np.int64)
        row_zero = np.floor(continuous_row).astype(np.int64)
        delta_column = continuous_column - column_zero
        delta_row = continuous_row - row_zero

        target_rows = np.concatenate((row_zero, row_zero, row_zero + 1, row_zero + 1))
        target_columns = np.concatenate(
            (column_zero, column_zero + 1, column_zero, column_zero + 1)
        )
        bilinear = np.concatenate(
            (
                (1.0 - delta_row) * (1.0 - delta_column),
                (1.0 - delta_row) * delta_column,
                delta_row * (1.0 - delta_column),
                delta_row * delta_column,
            )
        )
        repeated_source_indexes = np.tile(source_indexes, 4)
        inside = (
            (target_rows >= 0)
            & (target_rows < self.config.height)
            & (target_columns >= 0)
            & (target_columns < self.config.width)
            & (bilinear > 0.0)
        )
        if not np.any(inside):
            return self._reject(source.frame_id, "projected_footprint_outside_map", **result_counts)

        target_rows = target_rows[inside]
        target_columns = target_columns[inside]
        repeated_source_indexes = repeated_source_indexes[inside]
        bilinear = bilinear[inside]
        incidence = np.clip(-projection.rays_enu[repeated_source_indexes, 2], 0.0, 1.0)
        uncertainty_weight = 1.0 / (
            1.0
            + (
                uncertainty[repeated_source_indexes] / self.config.resolution_m
            )
            ** 2
        )
        contribution_weight = (
            bilinear
            * sampled_support[repeated_source_indexes]
            * incidence
            * uncertainty_weight
        )
        positive = contribution_weight > 0.0
        target_rows = target_rows[positive]
        target_columns = target_columns[positive]
        repeated_source_indexes = repeated_source_indexes[positive]
        contribution_weight = contribution_weight[positive]
        if len(contribution_weight) == 0:
            return self._reject(source.frame_id, "zero_weight_after_splat", **result_counts)

        source_rgb = np.asarray(source.rgb)[sampled_rows, sampled_columns]
        source_thermal = (
            None
            if source.thermal_normalized is None
            else np.asarray(source.thermal_normalized)[sampled_rows, sampled_columns]
        )
        source_semantic = (
            None
            if source.semantic_class is None
            else np.asarray(source.semantic_class)[sampled_rows, sampled_columns]
        )
        semantic_class_ids: tuple[int, ...] = ()
        if source_semantic is not None:
            semantic_values = source_semantic[repeated_source_indexes]
            semantic_class_ids = tuple(
                sorted(int(value) for value in np.unique(semantic_values) if value >= 0)
            )
            new_class_count = sum(
                class_id not in self._semantic_evidence for class_id in semantic_class_ids
            )
            if len(self._semantic_evidence) + new_class_count > self.config.max_semantic_classes:
                return self._reject(
                    source.frame_id,
                    "semantic_class_limit_exceeded",
                    **result_counts,
                )
            projected_class_count = len(self._semantic_evidence) + new_class_count
            projected_peak_bytes = self.config.estimated_peak_memory_bytes(
                projected_class_count
            )
            if projected_peak_bytes > self.config.max_peak_memory_bytes:
                return self._reject(
                    source.frame_id,
                    "semantic_memory_budget_exceeded",
                    **result_counts,
                )

        local_frame_weight = np.zeros(self.config.shape, dtype=np.float64)
        np.add.at(
            local_frame_weight,
            (target_rows, target_columns),
            contribution_weight,
        )

        with self._lock:
            # Mutations begin only after all frame-level safety checks pass.
            np.add.at(self._weight_sum, (target_rows, target_columns), contribution_weight)
            np.add.at(
                self._height_sum,
                (target_rows, target_columns),
                projection.points_enu_m[repeated_source_indexes, 2] * contribution_weight,
            )
            np.add.at(
                self._uncertainty_sum,
                (target_rows, target_columns),
                uncertainty[repeated_source_indexes] * contribution_weight,
            )
            np.add.at(self._coverage_count, (target_rows, target_columns), 1)
            for channel in range(3):
                np.add.at(
                    self._rgb_sum[..., channel],
                    (target_rows, target_columns),
                    source_rgb[repeated_source_indexes, channel] * contribution_weight,
                )

            if source_thermal is not None:
                thermal_values = source_thermal[repeated_source_indexes]
                thermal_valid = np.isfinite(thermal_values)
                if np.any(thermal_valid):
                    thermal_rows = target_rows[thermal_valid]
                    thermal_columns = target_columns[thermal_valid]
                    thermal_weights = contribution_weight[thermal_valid]
                    np.add.at(
                        self._thermal_sum,
                        (thermal_rows, thermal_columns),
                        thermal_values[thermal_valid] * thermal_weights,
                    )
                    np.add.at(
                        self._thermal_weight,
                        (thermal_rows, thermal_columns),
                        thermal_weights,
                    )

            if source_semantic is not None:
                semantic_values = source_semantic[repeated_source_indexes]
                for class_id in semantic_class_ids:
                    class_mask = semantic_values == class_id
                    evidence = self._semantic_evidence.setdefault(
                        class_id, np.zeros(self.config.shape, dtype=np.float64)
                    )
                    np.add.at(
                        evidence,
                        (target_rows[class_mask], target_columns[class_mask]),
                        contribution_weight[class_mask],
                    )

            touched = local_frame_weight > 0.0
            greater = touched & (local_frame_weight > self._dominant_source_weight)
            tied = (
                touched
                & np.isclose(local_frame_weight, self._dominant_source_weight, atol=1e-12)
                & (self._dominant_source_id == "")
            )
            replace = greater | tied
            self._dominant_source_weight[replace] = local_frame_weight[replace]
            self._dominant_source_id[replace] = source.frame_id
            self._frame_provenance[source.frame_id] = _freeze(
                {
                    "frame_id": source.frame_id,
                    "timestamp_tai_ns": source.timestamp_tai_ns,
                    "pose_id": pose.pose_id,
                    "calibration_id": intrinsics.calibration_id,
                    "time_offset_ns": time_offset,
                    "integrated_samples": int(len(np.unique(repeated_source_indexes))),
                    "mean_projected_uncertainty_m": mean_uncertainty,
                    "max_projected_uncertainty_m": maximum_uncertainty,
                    "source": source.provenance,
                }
            )
            result = FrameIntegrationResult(
                frame_id=source.frame_id,
                accepted=True,
                reason="accepted",
                integrated_samples=int(len(np.unique(repeated_source_indexes))),
                **result_counts,
            )
            self._accepted_results.append(result)
            return result

    def snapshot(self) -> MosaicSnapshot:
        """Return an owned, read-only snapshot; future integration cannot mutate it."""

        with self._lock:
            valid = self._weight_sum > 0.0
            rgb = np.full((*self.config.shape, 3), np.nan, dtype=np.float32)
            rgb[valid] = (self._rgb_sum[valid] / self._weight_sum[valid, None]).astype(
                np.float32
            )
            thermal = np.full(self.config.shape, np.nan, dtype=np.float32)
            thermal_valid = self._thermal_weight > 0.0
            thermal[thermal_valid] = (
                self._thermal_sum[thermal_valid] / self._thermal_weight[thermal_valid]
            ).astype(np.float32)
            height = np.full(self.config.shape, np.nan, dtype=np.float32)
            height[valid] = (self._height_sum[valid] / self._weight_sum[valid]).astype(
                np.float32
            )
            uncertainty = np.full(self.config.shape, np.nan, dtype=np.float32)
            uncertainty[valid] = (
                self._uncertainty_sum[valid] / self._weight_sum[valid]
            ).astype(np.float32)
            support = (-np.expm1(-self._weight_sum)).astype(np.float32)

            semantic_class = np.full(self.config.shape, -1, dtype=np.int32)
            semantic_confidence = np.zeros(self.config.shape, dtype=np.float32)
            class_evidence: dict[int, np.ndarray] = {}
            semantic_weight_sum = np.zeros(self.config.shape, dtype=np.float32)
            if self._semantic_evidence:
                class_ids = sorted(self._semantic_evidence)
                total_evidence = np.zeros(self.config.shape, dtype=np.float64)
                winning_evidence = np.zeros(self.config.shape, dtype=np.float64)
                winning_class = np.full(self.config.shape, -1, dtype=np.int32)
                for class_id in class_ids:
                    evidence = self._semantic_evidence[class_id]
                    total_evidence += evidence
                    better = evidence > winning_evidence
                    winning_evidence[better] = evidence[better]
                    winning_class[better] = class_id
                    class_evidence[class_id] = evidence.astype(np.float32)
                semantic_valid = total_evidence > 0.0
                semantic_class[semantic_valid] = winning_class[semantic_valid]
                semantic_confidence[semantic_valid] = (
                    winning_evidence[semantic_valid] / total_evidence[semantic_valid]
                ).astype(np.float32)
                semantic_weight_sum = total_evidence.astype(np.float32)

            # Python mappings preserve insertion order.  Keeping that order makes
            # source indexes stable as additional frames are appended.
            source_ids = tuple(self._frame_provenance)
            source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
            dominant_index = np.full(self.config.shape, -1, dtype=np.int32)
            for source_id, index in source_lookup.items():
                dominant_index[self._dominant_source_id == source_id] = index

            metadata = {
                "coordinate_reference": {
                    "type": "local_tangent_plane",
                    "axes": "ENU",
                    "horizontal_datum": self.frame.datum,
                    "vertical_datum": "WGS84_ellipsoid",
                    "frame_id": self.frame.frame_id,
                    "origin": {
                        "latitude_deg": self.frame.origin.latitude_deg,
                        "longitude_deg": self.frame.origin.longitude_deg,
                        "ellipsoid_height_m": self.frame.origin.ellipsoid_height_m,
                    },
                },
                "grid": {
                    "north_up": True,
                    "row_zero_north_m": self.config.north_max_m,
                    "east_min_m": self.config.east_min_m,
                    "east_max_m": self.config.east_max_m,
                    "north_min_m": self.config.north_min_m,
                    "north_max_m": self.config.north_max_m,
                    "ground_elevation_enu_m": self.config.ground_elevation_enu_m,
                    "resolution_m": self.config.resolution_m,
                    "shape": self.config.shape,
                },
                "surface_model": {
                    "kind": (
                        "north_up_height_field"
                        if self.config.surface_elevation_enu_m is not None
                        else "horizontal_ground_plane"
                    ),
                    "surface_model_id": self.config.surface_model_id,
                    "intersection_tolerance_m": (
                        self.config.surface_intersection_tolerance_m
                        if self.config.surface_elevation_enu_m is not None
                        else None
                    ),
                    "elevation_sigma_m": (
                        float(self.config.surface_elevation_sigma_m)
                        if np.ndim(self.config.surface_elevation_sigma_m) == 0
                        else {
                            "kind": "per_cell",
                            "minimum": float(
                                np.min(self.config.surface_elevation_sigma_m)
                            ),
                            "maximum": float(
                                np.max(self.config.surface_elevation_sigma_m)
                            ),
                        }
                    ),
                },
                "safety_policy": {
                    "min_downward_cosine": self.config.min_downward_cosine,
                    "max_slant_range_m": self.config.max_slant_range_m,
                    "max_projected_uncertainty_m": self.config.max_projected_uncertainty_m,
                    "frame_uncertainty_percentile": self.config.frame_uncertainty_percentile,
                    "max_pose_time_offset_ns": self.config.max_pose_time_offset_ns,
                    "max_cells": self.config.max_cells,
                    "max_semantic_classes": self.config.max_semantic_classes,
                    "max_peak_memory_bytes": self.config.max_peak_memory_bytes,
                    "estimated_peak_memory_bytes": (
                        self.config.estimated_peak_memory_bytes(
                            len(self._semantic_evidence)
                        )
                    ),
                    "max_sampled_pixels_per_frame": (
                        self.config.max_sampled_pixels_per_frame
                    ),
                    "surface_ray_max_work_per_call": (
                        self.config.surface_ray_max_work
                    ),
                },
                "accepted_frame_count": len(self._accepted_results),
                "rejected_frame_count": len(self._rejected_results),
                "rejections": tuple(
                    {"frame_id": item.frame_id, "reason": item.reason}
                    for item in self._rejected_results
                ),
                "weighting": "bilinear * source_support * incidence * uncertainty_weight",
                "support_transform": "1-exp(-weight_sum)",
            }
            return MosaicSnapshot(
                rgb=rgb,
                thermal_normalized=thermal,
                height_enu_m=height,
                semantic_class=semantic_class,
                semantic_confidence=semantic_confidence,
                coverage_count=self._coverage_count,
                support=support,
                weight_sum=self._weight_sum.astype(np.float32),
                projected_uncertainty_m=uncertainty,
                valid=valid,
                dominant_source_index=dominant_index,
                class_evidence=class_evidence,
                source_ids=source_ids,
                provenance={
                    source_id: self._frame_provenance[source_id] for source_id in source_ids
                },
                metadata=metadata,
                thermal_weight_sum=self._thermal_weight.astype(np.float32),
                semantic_weight_sum=semantic_weight_sum,
            )


__all__ = [
    "CameraIntrinsics",
    "FrameIntegrationResult",
    "GeodeticCoordinate",
    "GroundProjection",
    "LocalENUFrame",
    "MappingFrame",
    "MosaicSnapshot",
    "OrthoMosaicConfig",
    "OrthoMosaicGrid",
    "Quaternion",
    "enu_to_geodetic",
    "geodetic_to_enu",
    "intersect_ground_plane",
    "intersect_height_field",
    "project_pixels_to_height_field",
    "project_pixels_to_ground",
    "projected_ground_uncertainty",
    "sample_surface_elevation",
]
