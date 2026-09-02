"""Mission-level integration for the OpenPRISM tactical terrain atlas.

``mapping.py`` contains the geometry and mosaic accumulator.  This module is
the strict boundary between those algorithms and capture poses produced by a
Pixhawk.  It resolves the otherwise easy-to-miss differences in time scales,
vertical datums, and camera axes before allowing an image into the map.

The live atlas is intentionally a local, north-up 2.5-D product.  It is a
fast operational map, not a substitute for post-flight bundle adjustment,
multi-view stereo, surveyed control, or a calibrated elevation model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
from math import hypot, radians
import os
from pathlib import Path
import shutil
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Mapping
import uuid

import numpy as np
from PIL import Image, ImageDraw

from .mapping import (
    CameraIntrinsics,
    FrameIntegrationResult,
    GeodeticCoordinate,
    LocalENUFrame,
    MappingFrame,
    MosaicSnapshot,
    OrthoMosaicConfig,
    OrthoMosaicGrid,
    Quaternion,
    VehicleCameraPose,
    enu_to_geodetic,
    geodetic_to_enu,
    project_pixels_to_ground,
    project_pixels_to_height_field,
    projected_ground_uncertainty,
    sample_surface_elevation,
)
from .pixhawk import CameraPoseRecord, export_odm_geo_txt


_FIX_RANK = {
    "no_gps": 0,
    "no_fix": 1,
    "2d_fix": 2,
    "3d_fix": 3,
    "dgps": 4,
    "rtk_float": 5,
    "rtk_fixed": 6,
    "static": 7,
    "ppp": 8,
}


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    """Coerce integer-like scalar values without silently truncating policy."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer") from error
    try:
        exact = result == value
    except Exception:
        exact = False
    if not exact:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _jsonable(value: Any) -> Any:
    """Convert frozen/NumPy metadata to a deterministic JSON-compatible tree."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            _jsonable(item)
            for item in sorted(value, key=lambda item: repr(item))
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("metadata cannot contain NaN or infinite values")
    return value


def _mission_locked(method):
    """Serialize mission state mutations and coherent bundle snapshots."""

    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._mission_lock:
            return method(self, *args, **kwargs)

    return locked


def _cleanup_failed_staging(method):
    """Remove an unpublished generation if bundle serialization raises."""

    @wraps(method)
    def guarded(self, *args, **kwargs):
        self._active_bundle_staging = None
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            staging = self._active_bundle_staging
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            self._active_bundle_staging = None

    return guarded


def _bundle_publication_locked(method):
    """Serialize publishers across mapper instances and processes.

    The lock is deliberately fail-fast. A leftover lock is evidence of an
    interrupted publisher and must be inspected instead of silently stolen.
    """

    @wraps(method)
    def guarded(self, directory, *args, **kwargs):
        destination = Path(directory).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        lock_path = destination / ".openprism-publish.lock"
        lock_token = uuid.uuid4().hex
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"another atlas publisher is active or needs recovery: {lock_path}"
            ) from error
        try:
            payload = json.dumps(
                {
                    "pid": os.getpid(),
                    "token": lock_token,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            ).encode("utf-8")
            os.write(descriptor, payload)
            os.fsync(descriptor)
            return method(self, destination, *args, **kwargs)
        finally:
            os.close(descriptor)
            try:
                current_lock = json.loads(lock_path.read_text(encoding="utf-8"))
                if current_lock.get("token") == lock_token:
                    lock_path.unlink()
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                pass

    return guarded


def _stamp_preview_warning(path: Path, warning: str) -> None:
    """Burn a provenance warning into a staged preview before publication."""

    with Image.open(path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    text = str(warning).strip()
    if not text:
        return
    box_height = min(max(18, image.height // 8), image.height)
    draw.rectangle((0, 0, image.width, box_height), fill=(96, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    image.save(path, format="PNG")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AtlasMissionConfig:
    """Geospatial, timing, and acceptance policy for one tactical atlas.

    ``geoid_separation_m`` is ``ellipsoid height - MSL height`` at the mission
    site.  It is mandatory because MAVLink camera altitudes are MSL while the
    dependency-light WGS84 transform in :mod:`openprism.mapping` consumes
    ellipsoidal height.

    ``tai_minus_utc_s`` must be supplied when UTC capture timestamps are used.
    It is deliberately not hard-coded: leap-second policy is external state.
    A boot clock may instead be anchored with ``boot_epoch_tai_ns``.
    """

    mission_id: str
    origin_latitude_deg: float
    origin_longitude_deg: float
    ground_elevation_msl_m: float
    geoid_separation_m: float
    east_min_m: float
    east_max_m: float
    north_min_m: float
    north_max_m: float
    resolution_m: float
    horizontal_crs_id: str
    input_vertical_datum_id: str
    geoid_model_id: str
    geoid_separation_uncertainty_m: float
    surface_elevation_enu_m: np.ndarray | None = None
    surface_elevation_sigma_m: np.ndarray | float = 0.0
    surface_model_id: str | None = None
    tai_minus_utc_s: int | None = None
    boot_epoch_tai_ns: int | None = None
    boot_epoch_uncertainty_ns: int | None = None
    boot_clock_domain: str | None = None
    max_time_uncertainty_ns: int = 20_000_000
    default_time_uncertainty_ns: int | None = None
    maximum_pose_time_offset_ns: int = 100_000_000
    max_projected_uncertainty_m: float = 5.0
    min_downward_cosine: float = 0.1
    max_slant_range_m: float = 2_000.0
    minimum_fix_type: int = 3
    require_rtk_fixed: bool = False
    minimum_agl_m: float = 2.0
    maximum_agl_m: float = 1_000.0
    default_horizontal_accuracy_m: float | None = None
    default_vertical_accuracy_m: float | None = None
    default_attitude_accuracy_deg: float | None = None
    timing_velocity_bound_mps: float | None = None
    timing_angular_rate_bound_deg_s: float | None = None
    camera_extrinsic_calibration_id: str | None = None
    camera_lever_arm_optical_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_thermal_time_offset_ns: int = 50_000_000
    max_thermal_time_uncertainty_ns: int = 20_000_000
    max_thermal_registration_rms_px: float = 2.0
    real_sensor_data: bool | None = None
    real_navigation_data: bool | None = None
    operator_freshness_ttl_s: float | None = None
    object_retention_ns: int = 30_000_000_000
    max_object_observations: int = 100_000

    def __post_init__(self) -> None:
        if not self.mission_id.strip():
            raise ValueError("mission_id is required")
        latitude = _finite("origin_latitude_deg", self.origin_latitude_deg)
        longitude = _finite("origin_longitude_deg", self.origin_longitude_deg)
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("origin_latitude_deg must be within [-90, 90]")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("origin_longitude_deg must be within [-180, 180]")
        for name in (
            "ground_elevation_msl_m",
            "geoid_separation_m",
            "east_min_m",
            "east_max_m",
            "north_min_m",
            "north_max_m",
            "resolution_m",
            "max_projected_uncertainty_m",
            "min_downward_cosine",
            "max_slant_range_m",
            "minimum_agl_m",
            "maximum_agl_m",
            "max_thermal_registration_rms_px",
            "geoid_separation_uncertainty_m",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.geoid_separation_uncertainty_m < 0.0:
            raise ValueError("geoid_separation_uncertainty_m must be non-negative")
        for name in (
            "horizontal_crs_id",
            "input_vertical_datum_id",
            "geoid_model_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        if self.horizontal_crs_id.upper().replace(" ", "") != "EPSG:4326":
            raise ValueError(
                "horizontal_crs_id must be EPSG:4326 for geodetic latitude/longitude"
            )
        object.__setattr__(self, "horizontal_crs_id", "EPSG:4326")
        if self.minimum_agl_m < 0.0 or self.maximum_agl_m <= self.minimum_agl_m:
            raise ValueError("AGL limits must define a positive interval")
        for name in (
            "max_time_uncertainty_ns",
            "maximum_pose_time_offset_ns",
            "minimum_fix_type",
            "max_thermal_time_offset_ns",
            "max_thermal_time_uncertainty_ns",
            "object_retention_ns",
            "max_object_observations",
        ):
            minimum = 1 if name == "max_object_observations" else 0
            value = _exact_int(name, getattr(self, name), minimum=minimum)
            object.__setattr__(self, name, value)
        if self.default_time_uncertainty_ns is not None:
            default_time_uncertainty_ns = _exact_int(
                "default_time_uncertainty_ns",
                self.default_time_uncertainty_ns,
                minimum=0,
            )
            object.__setattr__(
                self,
                "default_time_uncertainty_ns",
                default_time_uncertainty_ns,
            )
        if self.max_thermal_registration_rms_px <= 0.0:
            raise ValueError("max_thermal_registration_rms_px must be positive")
        for name in (
            "default_horizontal_accuracy_m",
            "default_vertical_accuracy_m",
            "default_attitude_accuracy_deg",
            "timing_velocity_bound_mps",
            "timing_angular_rate_bound_deg_s",
        ):
            value = getattr(self, name)
            if value is not None:
                value = _finite(name, value)
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative")
                object.__setattr__(self, name, value)
        lever_arm = tuple(float(value) for value in self.camera_lever_arm_optical_m)
        if len(lever_arm) != 3 or not np.all(np.isfinite(lever_arm)):
            raise ValueError("camera_lever_arm_optical_m must contain three finite values")
        object.__setattr__(self, "camera_lever_arm_optical_m", lever_arm)
        if self.camera_extrinsic_calibration_id is not None:
            calibration_id = self.camera_extrinsic_calibration_id.strip()
            if not calibration_id:
                raise ValueError("camera_extrinsic_calibration_id cannot be blank")
            object.__setattr__(self, "camera_extrinsic_calibration_id", calibration_id)
        if self.tai_minus_utc_s is not None:
            tai_offset = _exact_int("tai_minus_utc_s", self.tai_minus_utc_s)
            if not 0 <= tai_offset <= 100:
                raise ValueError("tai_minus_utc_s must be within [0, 100]")
            object.__setattr__(self, "tai_minus_utc_s", tai_offset)
        if self.boot_epoch_tai_ns is not None:
            boot_epoch = _exact_int(
                "boot_epoch_tai_ns", self.boot_epoch_tai_ns, minimum=0
            )
            object.__setattr__(self, "boot_epoch_tai_ns", boot_epoch)
            if self.boot_epoch_uncertainty_ns is None:
                raise ValueError(
                    "boot_epoch_uncertainty_ns is required with boot_epoch_tai_ns"
                )
            boot_uncertainty = _exact_int(
                "boot_epoch_uncertainty_ns",
                self.boot_epoch_uncertainty_ns,
                minimum=0,
            )
            object.__setattr__(
                self, "boot_epoch_uncertainty_ns", boot_uncertainty
            )
            if not self.boot_clock_domain or not self.boot_clock_domain.strip():
                raise ValueError(
                    "boot_clock_domain is required with boot_epoch_tai_ns"
                )
            object.__setattr__(
                self, "boot_clock_domain", self.boot_clock_domain.strip()
            )
        elif self.boot_clock_domain is not None or self.boot_epoch_uncertainty_ns is not None:
            raise ValueError(
                "boot clock domain/uncertainty requires boot_epoch_tai_ns"
            )
        for name in ("real_sensor_data", "real_navigation_data"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be true, false, or null")
        if not isinstance(self.require_rtk_fixed, (bool, np.bool_)):
            raise ValueError("require_rtk_fixed must be boolean")
        object.__setattr__(self, "require_rtk_fixed", bool(self.require_rtk_fixed))
        if self.operator_freshness_ttl_s is not None:
            freshness = _finite(
                "operator_freshness_ttl_s", self.operator_freshness_ttl_s
            )
            if freshness <= 0.0:
                raise ValueError("operator_freshness_ttl_s must be positive")
            object.__setattr__(self, "operator_freshness_ttl_s", freshness)
        if self.surface_elevation_enu_m is not None:
            surface = np.array(
                self.surface_elevation_enu_m, dtype=np.float64, copy=True
            )
            surface.setflags(write=False)
            object.__setattr__(self, "surface_elevation_enu_m", surface)
        if np.ndim(self.surface_elevation_sigma_m) > 0:
            surface_sigma = np.array(
                self.surface_elevation_sigma_m, dtype=np.float64, copy=True
            )
            surface_sigma.setflags(write=False)
            object.__setattr__(self, "surface_elevation_sigma_m", surface_sigma)

        # Reuse the core policy validation now rather than failing on first use.
        self.mosaic_config()

    @property
    def ground_ellipsoid_height_m(self) -> float:
        return self.ground_elevation_msl_m + self.geoid_separation_m

    def local_frame(self) -> LocalENUFrame:
        return LocalENUFrame(
            GeodeticCoordinate(
                self.origin_latitude_deg,
                self.origin_longitude_deg,
                self.ground_ellipsoid_height_m,
            ),
            frame_id=f"{self.mission_id}:enu",
        )

    def mosaic_config(self) -> OrthoMosaicConfig:
        return OrthoMosaicConfig(
            east_min_m=self.east_min_m,
            east_max_m=self.east_max_m,
            north_min_m=self.north_min_m,
            north_max_m=self.north_max_m,
            resolution_m=self.resolution_m,
            ground_elevation_enu_m=0.0,
            surface_elevation_enu_m=self.surface_elevation_enu_m,
            surface_elevation_sigma_m=self.surface_elevation_sigma_m,
            surface_model_id=self.surface_model_id,
            min_downward_cosine=self.min_downward_cosine,
            max_slant_range_m=self.max_slant_range_m,
            max_projected_uncertainty_m=self.max_projected_uncertainty_m,
            max_pose_time_offset_ns=self.maximum_pose_time_offset_ns,
        )


@dataclass(frozen=True, slots=True)
class AtlasFrameResult:
    """Mission-level decision, including gates before geometric integration."""

    image_name: str
    accepted: bool
    reason: str
    integration: FrameIntegrationResult | None = None


@dataclass(frozen=True, slots=True)
class RegisteredThermalFrameContract:
    """Evidence that a thermal frame is safe in the RGB map pixel frame.

    The array passed to :meth:`AtlasMissionMapper.integrate_capture` must have
    already been rectified and registered into the declared RGB calibration.
    Normalization must be fixed for the mission (or radiometrically calibrated),
    never recomputed independently for each frame.
    """

    sensor_id: str
    reference_camera_calibration_id: str
    registration_calibration_id: str
    normalization_id: str
    capture_time_offset_ns: int
    capture_time_uncertainty_ns: int
    registration_rms_px: float
    source_unit: str = "relative_thermal_intensity"
    normalization_scope: str = "mission_fixed"
    registration_frame: str = "rgb_rectified_pixels"
    radiometric_calibration_id: str | None = None
    native_evidence_id: str | None = None
    nuc_state: str = "unknown"
    nuc_epoch_id: str = "unknown"

    def __post_init__(self) -> None:
        for name in (
            "sensor_id",
            "reference_camera_calibration_id",
            "registration_calibration_id",
            "normalization_id",
            "nuc_state",
            "nuc_epoch_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        for name in ("capture_time_offset_ns", "capture_time_uncertainty_ns"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or int(raw) != raw:
                raise ValueError(f"{name} must be an integer")
            value = int(raw)
            if name.endswith("uncertainty_ns") and value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        rms = _finite("registration_rms_px", self.registration_rms_px)
        if rms < 0.0:
            raise ValueError("registration_rms_px must be non-negative")
        object.__setattr__(self, "registration_rms_px", rms)
        if self.registration_frame != "rgb_rectified_pixels":
            raise ValueError("thermal values must be registered to RGB rectified pixels")
        if self.normalization_scope not in {
            "mission_fixed",
            "radiometric_calibrated",
        }:
            raise ValueError(
                "normalization_scope must be mission_fixed or radiometric_calibrated"
            )
        allowed_units = {
            "relative_thermal_intensity",
            "raw_sensor_count",
            "kelvin",
            "celsius",
        }
        if self.source_unit not in allowed_units:
            raise ValueError(f"unsupported thermal source_unit: {self.source_unit}")
        if self.source_unit in {"kelvin", "celsius"}:
            if not self.radiometric_calibration_id or not str(
                self.radiometric_calibration_id
            ).strip():
                raise ValueError(
                    "temperature units require radiometric_calibration_id"
                )
            object.__setattr__(
                self,
                "radiometric_calibration_id",
                str(self.radiometric_calibration_id).strip(),
            )
        if self.source_unit != "relative_thermal_intensity":
            native_id = (
                "" if self.native_evidence_id is None else str(self.native_evidence_id).strip()
            )
            if not native_id:
                raise ValueError(
                    "raw or temperature source units require native_evidence_id"
                )
            object.__setattr__(self, "native_evidence_id", native_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "reference_camera_calibration_id": (
                self.reference_camera_calibration_id
            ),
            "registration_calibration_id": self.registration_calibration_id,
            "registration_frame": self.registration_frame,
            "registration_rms_px": self.registration_rms_px,
            "capture_time_offset_ns": self.capture_time_offset_ns,
            "capture_time_uncertainty_ns": self.capture_time_uncertainty_ns,
            "source_unit": self.source_unit,
            "mapped_value_kind": "normalized_tactical_layer",
            "normalization_scope": self.normalization_scope,
            "normalization_id": self.normalization_id,
            "radiometric_calibration_id": self.radiometric_calibration_id,
            "native_evidence_id": self.native_evidence_id,
            "nuc_state": self.nuc_state,
            "nuc_epoch_id": self.nuc_epoch_id,
        }

    @property
    def mission_fingerprint(self) -> tuple[Any, ...]:
        """Fields that must remain invariant before values may be averaged."""

        return (
            self.sensor_id,
            self.reference_camera_calibration_id,
            self.registration_calibration_id,
            self.normalization_id,
            self.source_unit,
            self.normalization_scope,
            self.radiometric_calibration_id,
            self.native_evidence_id,
            self.nuc_state,
            self.nuc_epoch_id,
        )


@dataclass(frozen=True, slots=True)
class SemanticFrameContract:
    """Mission-fixed identity and class policy for a rectified semantic layer."""

    model_id: str
    model_artifact_id: str
    taxonomy_id: str
    reference_camera_calibration_id: str
    class_labels: Mapping[int, str]
    dynamic_class_ids: frozenset[int] = frozenset()
    unknown_class_id: int = -1
    coordinate_space: str = "rgb_rectified_pixels"

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_artifact_id",
            "taxonomy_id",
            "reference_camera_calibration_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        if self.coordinate_space != "rgb_rectified_pixels":
            raise ValueError("semantic values must use RGB rectified pixel coordinates")
        unknown = _exact_int("unknown_class_id", self.unknown_class_id)
        labels: dict[int, str] = {}
        for raw_class_id, raw_label in dict(self.class_labels).items():
            class_id = _exact_int("semantic class id", raw_class_id, minimum=0)
            if class_id > np.iinfo(np.int32).max:
                raise ValueError("semantic class ids must fit signed int32")
            label = str(raw_label).strip()
            if not label:
                raise ValueError("semantic class labels cannot be blank")
            labels[class_id] = label
        if not labels:
            raise ValueError("class_labels must declare at least one class")
        dynamic = frozenset(
            _exact_int("dynamic class id", item, minimum=0)
            for item in self.dynamic_class_ids
        )
        if unknown < np.iinfo(np.int32).min or unknown > np.iinfo(np.int32).max:
            raise ValueError("unknown_class_id must fit signed int32")
        if not dynamic.issubset(labels):
            raise ValueError("dynamic_class_ids must be declared in class_labels")
        if unknown in labels:
            raise ValueError("unknown_class_id must not collide with a declared class")
        object.__setattr__(self, "unknown_class_id", unknown)
        object.__setattr__(self, "class_labels", MappingProxyType(labels))
        object.__setattr__(self, "dynamic_class_ids", dynamic)

    @property
    def mission_fingerprint(self) -> tuple[Any, ...]:
        return (
            self.model_id,
            self.model_artifact_id,
            self.taxonomy_id,
            self.reference_camera_calibration_id,
            tuple(sorted(self.class_labels.items())),
            tuple(sorted(self.dynamic_class_ids)),
            self.unknown_class_id,
            self.coordinate_space,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_artifact_id": self.model_artifact_id,
            "taxonomy_id": self.taxonomy_id,
            "reference_camera_calibration_id": (
                self.reference_camera_calibration_id
            ),
            "coordinate_space": self.coordinate_space,
            "class_labels": {
                str(class_id): label
                for class_id, label in sorted(self.class_labels.items())
            },
            "dynamic_class_ids": sorted(self.dynamic_class_ids),
            "unknown_class_id": self.unknown_class_id,
        }


@dataclass(frozen=True, slots=True)
class ImageObjectObservation:
    """One transient object anchor in a source image.

    The anchor should touch the terrain: usually the bottom-centre of a person
    or vehicle box.  Projecting the box centre would place tall objects too far
    away in an oblique view. ``from_xywh`` interprets boxes as half-open pixel
    sample extents and anchors on the centre of their last included pixel row.
    """

    object_id: str
    label: str
    anchor_u_px: float
    anchor_v_px: float
    reference_camera_calibration_id: str
    coordinate_space: str = "rgb_rectified_pixels"
    confidence: float | None = None
    source: str = "model"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id.strip() or not self.label.strip():
            raise ValueError("object_id and label are required")
        calibration_id = str(self.reference_camera_calibration_id).strip()
        if not calibration_id:
            raise ValueError("reference_camera_calibration_id is required")
        object.__setattr__(
            self, "reference_camera_calibration_id", calibration_id
        )
        if self.coordinate_space != "rgb_rectified_pixels":
            raise ValueError("object anchors must use RGB rectified pixel coordinates")
        object.__setattr__(self, "anchor_u_px", _finite("anchor_u_px", self.anchor_u_px))
        object.__setattr__(self, "anchor_v_px", _finite("anchor_v_px", self.anchor_v_px))
        if self.confidence is not None:
            confidence = _finite("confidence", self.confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be within [0, 1]")
            object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )

    @classmethod
    def from_xywh(
        cls,
        object_id: str,
        label: str,
        x_px: float,
        y_px: float,
        width_px: float,
        height_px: float,
        **kwargs: Any,
    ) -> "ImageObjectObservation":
        width = _finite("width_px", width_px)
        height = _finite("height_px", height_px)
        if width <= 0.0 or height <= 0.0:
            raise ValueError("object box width and height must be positive")
        x = _finite("x_px", x_px)
        y = _finite("y_px", y_px)
        return cls(
            object_id=object_id,
            label=label,
            anchor_u_px=x + max(width - 1.0, 0.0) / 2.0,
            anchor_v_px=y + max(height - 1.0, 0.0),
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class GeoObjectObservation:
    """A time-stamped object projected onto the terrain, never baked into it."""

    object_id: str
    label: str
    source_frame_id: str
    timestamp_tai_ns: int
    east_m: float
    north_m: float
    elevation_enu_m: float
    longitude_deg: float
    latitude_deg: float
    ground_altitude_msl_m: float
    horizontal_uncertainty_m: float
    pixel_uncertainty_px: float
    reference_camera_calibration_id: str
    coordinate_space: str
    pose_id: str
    surface_model_id: str | None
    confidence: float | None
    source: str
    provenance: Mapping[str, Any]

    def as_geojson_feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "id": self.object_id,
            "properties": {
                "object_id": self.object_id,
                "label": self.label,
                "source_frame_id": self.source_frame_id,
                # Nanosecond TAI epochs exceed JavaScript's exact integer range.
                "timestamp_tai_ns": str(self.timestamp_tai_ns),
                "timestamp_encoding": "decimal_string_tai_nanoseconds",
                "east_m": self.east_m,
                "north_m": self.north_m,
                "elevation_enu_m": self.elevation_enu_m,
                "ground_altitude_msl_m": self.ground_altitude_msl_m,
                "horizontal_uncertainty_m": self.horizontal_uncertainty_m,
                "pixel_uncertainty_px": self.pixel_uncertainty_px,
                "reference_camera_calibration_id": (
                    self.reference_camera_calibration_id
                ),
                "coordinate_space": self.coordinate_space,
                "pose_id": self.pose_id,
                "surface_model_id": self.surface_model_id,
                "confidence": self.confidence,
                "source": self.source,
                "provenance": _jsonable(self.provenance),
            },
            "geometry": {
                "type": "Point",
                "coordinates": [self.longitude_deg, self.latitude_deg],
            },
        }


@dataclass(frozen=True, slots=True)
class GeoObjectProjectionResult:
    source_frame_id: str
    object_id: str
    accepted: bool
    reason: str
    observation: GeoObjectObservation | None = None


@dataclass(frozen=True, slots=True)
class AtlasBundlePaths:
    directory: Path
    arrays: Path
    metadata: Path
    rgb_preview: Path
    thermal_preview: Path
    support_preview: Path
    bounds_geojson: Path
    objects_geojson: Path
    odm_geo: Path
    manifest: Path


class AtlasMissionMapper:
    """Fuse calibrated RGB-T captures and Pixhawk poses into one local atlas."""

    def __init__(
        self,
        config: AtlasMissionConfig,
        intrinsics: CameraIntrinsics,
    ) -> None:
        if not intrinsics.calibration_id:
            raise ValueError("a traceable camera calibration_id is required")
        self.config = config
        self.intrinsics = intrinsics
        self._mission_lock = threading.RLock()
        self.local_frame = config.local_frame()
        # Deliberately private: direct OrthoMosaicGrid integration would bypass
        # this mission boundary's time, datum, fix, extrinsic, and layer gates.
        self._grid = OrthoMosaicGrid(self.local_frame, config.mosaic_config())
        self._results: list[AtlasFrameResult] = []
        self._accepted_records: list[CameraPoseRecord] = []
        self._accepted_context: dict[
            str, tuple[CameraPoseRecord, int, VehicleCameraPose]
        ] = {}
        self._object_results: list[GeoObjectProjectionResult] = []
        self._latest_object_timestamp_tai_ns: int | None = None
        self._thermal_mission_fingerprint: tuple[Any, ...] | None = None
        self._semantic_mission_fingerprint: tuple[Any, ...] | None = None
        self._active_bundle_staging: Path | None = None

    @property
    def results(self) -> tuple[AtlasFrameResult, ...]:
        return tuple(self._results)

    @property
    def accepted_records(self) -> tuple[CameraPoseRecord, ...]:
        return tuple(self._accepted_records)

    @property
    def object_results(self) -> tuple[GeoObjectProjectionResult, ...]:
        return tuple(self._object_results)

    def _reject(self, record: CameraPoseRecord, reason: str) -> AtlasFrameResult:
        result = AtlasFrameResult(record.image_name, False, reason)
        self._results.append(result)
        return result

    def _store_object_result(
        self, result: GeoObjectProjectionResult
    ) -> GeoObjectProjectionResult:
        """Bound the transient track layer by mission time and record count."""

        self._object_results.append(result)
        if result.observation is not None and self.config.object_retention_ns > 0:
            self._latest_object_timestamp_tai_ns = max(
                result.observation.timestamp_tai_ns,
                self._latest_object_timestamp_tai_ns
                if self._latest_object_timestamp_tai_ns is not None
                else result.observation.timestamp_tai_ns,
            )
            cutoff = (
                self._latest_object_timestamp_tai_ns
                - self.config.object_retention_ns
            )
            self._object_results[:] = [
                item
                for item in self._object_results
                if item.observation is None
                or item.observation.timestamp_tai_ns >= cutoff
            ]
        excess = len(self._object_results) - self.config.max_object_observations
        if excess > 0:
            del self._object_results[:excess]
        return result

    def _capture_tai_ns(self, record: CameraPoseRecord) -> int | None:
        if record.capture_utc_ns is not None and self.config.tai_minus_utc_s is not None:
            return int(record.capture_utc_ns) + self.config.tai_minus_utc_s * 1_000_000_000
        if (
            record.capture_monotonic_ns is not None
            and self.config.boot_epoch_tai_ns is not None
        ):
            return self.config.boot_epoch_tai_ns + int(record.capture_monotonic_ns)
        return None

    def _accuracy(
        self,
        measured: float | None,
        fallback: float | None,
    ) -> float | None:
        return measured if measured is not None else fallback

    @_mission_locked
    def integrate_capture(
        self,
        record: CameraPoseRecord,
        rgb: np.ndarray,
        *,
        thermal_normalized: np.ndarray | None = None,
        thermal_contract: RegisteredThermalFrameContract | None = None,
        semantic_class: np.ndarray | None = None,
        semantic_contract: SemanticFrameContract | None = None,
        static_validity: np.ndarray | None = None,
        dynamic_mask: np.ndarray | None = None,
        support: np.ndarray | float = 1.0,
        pixel_uncertainty_px: np.ndarray | float = 0.0,
        sample_stride: int = 1,
        provenance: Mapping[str, Any] = MappingProxyType({}),
    ) -> AtlasFrameResult:
        """Integrate one capture after datum, time, quality, and dynamics gates.

        ``dynamic_mask`` is true for transient pixels (people, vehicles, rotor
        blades, etc.).  Those pixels are deliberately excluded from the static
        terrain mosaic and should be stored in a separate tracking layer.
        """

        if (
            record.capture_monotonic_ns is not None
            and self.config.boot_epoch_tai_ns is not None
            and record.clock_domain != self.config.boot_clock_domain
        ):
            return self._reject(record, "boot_clock_domain_mismatch")
        utc_candidate = (
            int(record.capture_utc_ns)
            + self.config.tai_minus_utc_s * 1_000_000_000
            if record.capture_utc_ns is not None
            and self.config.tai_minus_utc_s is not None
            else None
        )
        boot_candidate = (
            self.config.boot_epoch_tai_ns + int(record.capture_monotonic_ns)
            if record.capture_monotonic_ns is not None
            and self.config.boot_epoch_tai_ns is not None
            else None
        )
        anchor_disagreement_ns = (
            abs(utc_candidate - boot_candidate)
            if utc_candidate is not None and boot_candidate is not None
            else None
        )
        if (
            utc_candidate is not None
            and boot_candidate is not None
            and anchor_disagreement_ns is not None
            and anchor_disagreement_ns > self.config.max_time_uncertainty_ns
        ):
            return self._reject(record, "utc_boot_time_anchors_disagree")
        timestamp_tai_ns = self._capture_tai_ns(record)
        if timestamp_tai_ns is None:
            return self._reject(record, "capture_time_has_no_explicit_tai_anchor")
        base_time_uncertainty_ns = (
            record.time_uncertainty_ns
            if record.time_uncertainty_ns is not None
            else self.config.default_time_uncertainty_ns
        )
        if base_time_uncertainty_ns is None:
            return self._reject(record, "capture_time_uncertainty_unknown")
        time_uncertainty_ns = int(base_time_uncertainty_ns)
        if anchor_disagreement_ns is not None:
            time_uncertainty_ns = max(time_uncertainty_ns, anchor_disagreement_ns)
        elif boot_candidate is not None and utc_candidate is None:
            # The boot-to-TAI anchor is part of the exposure-time error budget.
            time_uncertainty_ns += int(self.config.boot_epoch_uncertainty_ns or 0)
        if time_uncertainty_ns > self.config.max_time_uncertainty_ns:
            return self._reject(record, "capture_time_uncertainty_exceeds_limit")

        thermal_alignment_time_bound_s = 0.0
        thermal_registration_rms_px = 0.0
        if thermal_normalized is None and thermal_contract is not None:
            return self._reject(record, "thermal_contract_without_thermal_frame")
        if thermal_normalized is not None:
            if thermal_contract is None:
                return self._reject(record, "thermal_registration_contract_unknown")
            if (
                thermal_contract.reference_camera_calibration_id
                != self.intrinsics.calibration_id
            ):
                return self._reject(
                    record, "thermal_reference_calibration_mismatch"
                )
            if (
                abs(thermal_contract.capture_time_offset_ns)
                + thermal_contract.capture_time_uncertainty_ns
                > self.config.max_thermal_time_offset_ns
            ):
                return self._reject(record, "thermal_time_offset_exceeds_limit")
            if (
                thermal_contract.capture_time_uncertainty_ns
                > self.config.max_thermal_time_uncertainty_ns
            ):
                return self._reject(
                    record, "thermal_time_uncertainty_exceeds_limit"
                )
            if (
                thermal_contract.registration_rms_px
                > self.config.max_thermal_registration_rms_px
            ):
                return self._reject(
                    record, "thermal_registration_error_exceeds_limit"
                )
            if (
                self._thermal_mission_fingerprint is not None
                and thermal_contract.mission_fingerprint
                != self._thermal_mission_fingerprint
            ):
                return self._reject(record, "thermal_mission_contract_changed")
            thermal_alignment_time_bound_s = (
                abs(thermal_contract.capture_time_offset_ns)
                + thermal_contract.capture_time_uncertainty_ns
            ) * 1e-9
            thermal_registration_rms_px = thermal_contract.registration_rms_px

        if semantic_class is None and semantic_contract is not None:
            return self._reject(record, "semantic_contract_without_semantic_frame")
        if semantic_class is not None:
            if semantic_contract is None:
                return self._reject(record, "semantic_contract_unknown")
            if (
                semantic_contract.reference_camera_calibration_id
                != self.intrinsics.calibration_id
            ):
                return self._reject(
                    record, "semantic_reference_calibration_mismatch"
                )
            if (
                self._semantic_mission_fingerprint is not None
                and semantic_contract.mission_fingerprint
                != self._semantic_mission_fingerprint
            ):
                return self._reject(record, "semantic_mission_contract_changed")

        if isinstance(record.fix_type, int):
            fix_rank = record.fix_type
        elif isinstance(record.fix_type, str):
            normalized_fix = record.fix_type.strip().lower().replace(" ", "_")
            fix_rank = _FIX_RANK.get(normalized_fix)
        else:
            fix_rank = None
        if fix_rank is None:
            return self._reject(record, "navigation_fix_unknown")
        if fix_rank < self.config.minimum_fix_type:
            return self._reject(record, "navigation_fix_below_minimum")
        # Gate on the canonical fix type, not a free-text status label.  The
        # record validates their consistency, and this remains defense in depth
        # for records loaded through another serialization boundary.
        if self.config.require_rtk_fixed and fix_rank != 6:
            return self._reject(record, "rtk_fixed_required")

        if record.position_reference == "unspecified":
            return self._reject(record, "capture_position_reference_unknown")
        position_is_vehicle_origin = (
            record.position_reference == "vehicle_navigation_origin"
        )
        attitude_uses_vehicle_mount = "ATTITUDE_QUATERNION" in record.attitude_source
        if (
            position_is_vehicle_origin or attitude_uses_vehicle_mount
        ) and not self.config.camera_extrinsic_calibration_id:
            return self._reject(record, "camera_extrinsic_calibration_unknown")

        horizontal_accuracy = self._accuracy(
            record.horizontal_accuracy_m,
            self.config.default_horizontal_accuracy_m,
        )
        vertical_accuracy = self._accuracy(
            record.vertical_accuracy_m,
            self.config.default_vertical_accuracy_m,
        )
        attitude_accuracy = self._accuracy(
            record.attitude_accuracy_deg,
            self.config.default_attitude_accuracy_deg,
        )
        if horizontal_accuracy is None:
            return self._reject(record, "horizontal_accuracy_unknown")
        if vertical_accuracy is None:
            return self._reject(record, "vertical_accuracy_unknown")
        if attitude_accuracy is None:
            return self._reject(record, "attitude_accuracy_unknown")
        vertical_accuracy_with_datum_m = hypot(
            float(vertical_accuracy),
            self.config.geoid_separation_uncertainty_m,
        )

        timing_sigma_s = time_uncertainty_ns * 1e-9
        if timing_sigma_s > 0.0 and self.config.timing_velocity_bound_mps is None:
            return self._reject(record, "timing_velocity_bound_unknown")
        if (
            timing_sigma_s > 0.0
            and self.config.timing_angular_rate_bound_deg_s is None
        ):
            return self._reject(record, "timing_angular_rate_bound_unknown")
        if (
            thermal_alignment_time_bound_s > 0.0
            and self.config.timing_velocity_bound_mps is None
        ):
            return self._reject(record, "thermal_velocity_bound_unknown")
        if (
            thermal_alignment_time_bound_s > 0.0
            and self.config.timing_angular_rate_bound_deg_s is None
        ):
            return self._reject(record, "thermal_angular_rate_bound_unknown")
        translation_timing_sigma_m = (
            0.0
            if self.config.timing_velocity_bound_mps is None
            else self.config.timing_velocity_bound_mps * timing_sigma_s
        )
        attitude_timing_sigma_rad = radians(
            0.0
            if self.config.timing_angular_rate_bound_deg_s is None
            else self.config.timing_angular_rate_bound_deg_s * timing_sigma_s
        )
        thermal_translation_sigma_m = (
            0.0
            if self.config.timing_velocity_bound_mps is None
            else self.config.timing_velocity_bound_mps
            * thermal_alignment_time_bound_s
        )
        thermal_attitude_sigma_rad = radians(
            0.0
            if self.config.timing_angular_rate_bound_deg_s is None
            else self.config.timing_angular_rate_bound_deg_s
            * thermal_alignment_time_bound_s
        )
        conservative_translation_sigma_m = hypot(
            translation_timing_sigma_m, thermal_translation_sigma_m
        )
        conservative_attitude_timing_sigma_rad = hypot(
            attitude_timing_sigma_rad, thermal_attitude_sigma_rad
        )

        camera_position = GeodeticCoordinate(
            record.latitude_deg,
            record.longitude_deg,
            record.altitude_msl_m + self.config.geoid_separation_m,
        )
        navigation_origin_enu = geodetic_to_enu(camera_position, self.local_frame)
        camera_orientation = Quaternion(
            *record.quaternion_camera_optical_to_enu_wxyz
        )
        lever_arm_enu = (
            camera_orientation.rotate(
                np.asarray(self.config.camera_lever_arm_optical_m)
            )
            if position_is_vehicle_origin
            else np.zeros(3, dtype=np.float64)
        )
        camera_center_enu = navigation_origin_enu + lever_arm_enu
        if self._grid.config.surface_elevation_enu_m is None:
            local_ground_elevation_enu_m = 0.0
        else:
            sampled_ground, sampled_valid = sample_surface_elevation(
                np.asarray([camera_center_enu[0]]),
                np.asarray([camera_center_enu[1]]),
                self._grid.config,
            )
            if not bool(sampled_valid[0]):
                return self._reject(record, "camera_ground_elevation_unknown")
            local_ground_elevation_enu_m = float(sampled_ground[0])
        agl_m = float(camera_center_enu[2] - local_ground_elevation_enu_m)
        if not self.config.minimum_agl_m <= agl_m <= self.config.maximum_agl_m:
            return self._reject(record, "camera_agl_outside_limits")

        rgb_array = np.asarray(rgb)
        if rgb_array.shape[:2] != (self.intrinsics.height, self.intrinsics.width):
            return self._reject(record, "rgb_geometry_does_not_match_calibration")
        thermal_array = (
            None
            if thermal_normalized is None
            else np.asarray(thermal_normalized)
        )
        if thermal_array is not None and thermal_array.shape != rgb_array.shape[:2]:
            return self._reject(record, "thermal_registered_geometry_mismatch")
        semantic_array: np.ndarray | None = None
        if semantic_class is not None:
            candidate_semantic = np.asarray(semantic_class)
            if candidate_semantic.shape != rgb_array.shape[:2]:
                return self._reject(record, "semantic_geometry_mismatch")
            if (
                np.issubdtype(candidate_semantic.dtype, np.bool_)
                or not np.issubdtype(candidate_semantic.dtype, np.integer)
            ):
                return self._reject(record, "semantic_values_must_be_integers")
            int32_limits = np.iinfo(np.int32)
            semantic_min = int(np.min(candidate_semantic))
            semantic_max = int(np.max(candidate_semantic))
            if (
                semantic_min < int32_limits.min
                or semantic_max > int32_limits.max
            ):
                return self._reject(record, "semantic_values_outside_int32")
            semantic_array = candidate_semantic.astype(np.int32, copy=False)
            allowed = set(semantic_contract.class_labels)
            allowed.add(semantic_contract.unknown_class_id)
            if any(int(value) not in allowed for value in np.unique(semantic_array)):
                return self._reject(record, "semantic_class_not_in_taxonomy")
        validity = np.ones(rgb_array.shape[:2], dtype=bool)
        if static_validity is not None:
            supplied = np.asarray(static_validity, dtype=bool)
            if supplied.shape != validity.shape:
                return self._reject(record, "static_validity_geometry_mismatch")
            validity &= supplied
        if dynamic_mask is not None:
            moving = np.asarray(dynamic_mask, dtype=bool)
            if moving.shape != validity.shape:
                return self._reject(record, "dynamic_mask_geometry_mismatch")
            validity &= ~moving
        semantic_dynamic_count = 0
        if semantic_array is not None and semantic_contract.dynamic_class_ids:
            semantic_dynamic = np.isin(
                semantic_array,
                tuple(semantic_contract.dynamic_class_ids),
            )
            semantic_dynamic_count = int(np.count_nonzero(semantic_dynamic))
            validity &= ~semantic_dynamic

        pose = VehicleCameraPose(
            position=camera_position,
            enu_from_body=camera_orientation,
            # The record already carries camera-optical -> ENU.  Treat that
            # optical frame as the named intermediate and use no hidden basis
            # conversion inside the mapping contract.
            body_from_camera=Quaternion.identity(),
            timestamp_tai_ns=timestamp_tai_ns,
            pose_id=(
                f"pixhawk:{record.image_index}"
                if record.image_index is not None
                else f"pixhawk:{record.image_name}:{timestamp_tai_ns}"
            ),
            position_sigma_enu_m=(
                hypot(float(horizontal_accuracy), conservative_translation_sigma_m),
                hypot(float(horizontal_accuracy), conservative_translation_sigma_m),
                hypot(vertical_accuracy_with_datum_m, conservative_translation_sigma_m),
            ),
            attitude_sigma_rad=hypot(
                radians(float(attitude_accuracy)),
                conservative_attitude_timing_sigma_rad,
            ),
            camera_position_body_m=(
                self.config.camera_lever_arm_optical_m
                if position_is_vehicle_origin
                else (0.0, 0.0, 0.0)
            ),
        )
        source_provenance = {
            "mission_id": self.config.mission_id,
            "capture_pose": record.as_dict(),
            "vertical_conversion": {
                "input": self.config.input_vertical_datum_id,
                "output": "WGS84_ellipsoid_height",
                "horizontal_crs_id": self.config.horizontal_crs_id,
                "geoid_model_id": self.config.geoid_model_id,
                "geoid_separation_m": self.config.geoid_separation_m,
                "geoid_separation_uncertainty_m": (
                    self.config.geoid_separation_uncertainty_m
                ),
                "navigation_vertical_accuracy_m": float(vertical_accuracy),
                "vertical_accuracy_with_datum_m": vertical_accuracy_with_datum_m,
                "camera_agl_m": agl_m,
                "ground_elevation_enu_m_at_camera": (
                    local_ground_elevation_enu_m
                ),
                "navigation_origin_enu_m": navigation_origin_enu.tolist(),
                "camera_center_enu_m": camera_center_enu.tolist(),
            },
            "time_conversion": {
                "output": "TAI",
                "tai_minus_utc_s": self.config.tai_minus_utc_s,
                "boot_epoch_tai_ns": self.config.boot_epoch_tai_ns,
                "capture_time_uncertainty_ns": time_uncertainty_ns,
                "base_capture_time_uncertainty_ns": base_time_uncertainty_ns,
                "utc_boot_anchor_disagreement_ns": anchor_disagreement_ns,
                "boot_epoch_uncertainty_ns": self.config.boot_epoch_uncertainty_ns,
                "uncertainty_source": (
                    "capture_record"
                    if record.time_uncertainty_ns is not None
                    else "configured_conservative_default"
                ),
                "translation_timing_sigma_m": translation_timing_sigma_m,
                "attitude_timing_sigma_rad": attitude_timing_sigma_rad,
                "thermal_alignment_time_bound_s": (
                    thermal_alignment_time_bound_s
                ),
                "thermal_translation_sigma_m": thermal_translation_sigma_m,
                "thermal_attitude_sigma_rad": thermal_attitude_sigma_rad,
            },
            "camera_extrinsics": {
                "calibration_id": self.config.camera_extrinsic_calibration_id,
                "lever_arm_optical_m": self.config.camera_lever_arm_optical_m,
                "lever_arm_applied": position_is_vehicle_origin,
            },
            "thermal_registration": (
                None if thermal_contract is None else thermal_contract.as_dict()
            ),
            "semantic_contract": (
                None if semantic_contract is None else semantic_contract.as_dict()
            ),
            "dynamic_pixels_excluded": int(np.count_nonzero(~validity)),
            "semantic_dynamic_pixels_excluded": semantic_dynamic_count,
            "caller": dict(provenance),
        }
        mapping_frame = MappingFrame(
            frame_id=record.image_name,
            timestamp_tai_ns=timestamp_tai_ns,
            rgb=rgb_array,
            intrinsics=self.intrinsics,
            pose=pose,
            thermal_normalized=thermal_array,
            semantic_class=semantic_array,
            validity=validity,
            support=support,
            pixel_uncertainty_px=np.hypot(
                np.asarray(pixel_uncertainty_px, dtype=np.float64),
                thermal_registration_rms_px,
            ),
            provenance=source_provenance,
        )
        integration = self._grid.integrate(mapping_frame, sample_stride=sample_stride)
        result = AtlasFrameResult(
            record.image_name,
            integration.accepted,
            integration.reason,
            integration,
        )
        self._results.append(result)
        if integration.accepted:
            if thermal_contract is not None:
                self._thermal_mission_fingerprint = (
                    thermal_contract.mission_fingerprint
                )
            if semantic_contract is not None:
                self._semantic_mission_fingerprint = (
                    semantic_contract.mission_fingerprint
                )
            self._accepted_records.append(record)
            self._accepted_context[record.image_name] = (
                record,
                timestamp_tai_ns,
                pose,
            )
        return result

    @_mission_locked
    def project_object(
        self,
        source_frame_id: str,
        observation: ImageObjectObservation,
        *,
        pixel_uncertainty_px: float = 2.0,
    ) -> GeoObjectProjectionResult:
        """Project a detection anchor into the transient GeoJSON object layer."""

        if (
            observation.reference_camera_calibration_id
            != self.intrinsics.calibration_id
        ):
            result = GeoObjectProjectionResult(
                source_frame_id,
                observation.object_id,
                False,
                "object_reference_calibration_mismatch",
            )
            return self._store_object_result(result)
        context = self._accepted_context.get(source_frame_id)
        if context is None:
            result = GeoObjectProjectionResult(
                source_frame_id,
                observation.object_id,
                False,
                "source_frame_was_not_accepted_into_atlas",
            )
            return self._store_object_result(result)
        record, timestamp_tai_ns, pose = context
        pixel_sigma = _finite("pixel_uncertainty_px", pixel_uncertainty_px)
        if pixel_sigma < 0.0:
            raise ValueError("pixel_uncertainty_px must be non-negative")
        object_pixel = np.array(
            [[observation.anchor_u_px, observation.anchor_v_px]]
        )
        if self._grid.config.surface_elevation_enu_m is None:
            projection = project_pixels_to_ground(
                object_pixel,
                self.intrinsics,
                pose,
                self.local_frame,
                ground_elevation_enu_m=0.0,
                min_downward_cosine=self.config.min_downward_cosine,
                max_slant_range_m=self.config.max_slant_range_m,
            )
        else:
            projection = project_pixels_to_height_field(
                object_pixel,
                self.intrinsics,
                pose,
                self.local_frame,
                self._grid.config,
            )
        if not bool(projection.valid[0]):
            result = GeoObjectProjectionResult(
                source_frame_id,
                observation.object_id,
                False,
                "object_anchor_has_no_safe_ground_intersection",
            )
            return self._store_object_result(result)
        uncertainty = float(
            projected_ground_uncertainty(
                projection,
                self.intrinsics,
                pose,
                frame_timestamp_tai_ns=timestamp_tai_ns,
                pixel_sigma_px=pixel_sigma,
                ground_elevation_enu_m=projection.points_enu_m[:, 2],
                config=(
                    self._grid.config
                    if self._grid.config.surface_elevation_enu_m is not None
                    else None
                ),
            )[0]
        )
        if uncertainty > self.config.max_projected_uncertainty_m:
            result = GeoObjectProjectionResult(
                source_frame_id,
                observation.object_id,
                False,
                "object_projected_uncertainty_exceeds_limit",
            )
            return self._store_object_result(result)

        east_m, north_m, elevation_enu_m = (
            float(value) for value in projection.points_enu_m[0]
        )
        coordinate = enu_to_geodetic(
            projection.points_enu_m[0], self.local_frame
        )
        located = GeoObjectObservation(
            object_id=observation.object_id,
            label=observation.label,
            source_frame_id=source_frame_id,
            timestamp_tai_ns=timestamp_tai_ns,
            east_m=east_m,
            north_m=north_m,
            elevation_enu_m=elevation_enu_m,
            longitude_deg=coordinate.longitude_deg,
            latitude_deg=coordinate.latitude_deg,
            ground_altitude_msl_m=(
                coordinate.ellipsoid_height_m - self.config.geoid_separation_m
            ),
            horizontal_uncertainty_m=uncertainty,
            pixel_uncertainty_px=pixel_sigma,
            reference_camera_calibration_id=(
                observation.reference_camera_calibration_id
            ),
            coordinate_space=observation.coordinate_space,
            pose_id=pose.pose_id,
            surface_model_id=self._grid.config.surface_model_id,
            confidence=observation.confidence,
            source=observation.source,
            provenance=MappingProxyType(
                {
                    "anchor_uv_px": [
                        observation.anchor_u_px,
                        observation.anchor_v_px,
                    ],
                    "capture_source": record.source_message,
                    "pose_id": pose.pose_id,
                    "surface_model_id": self._grid.config.surface_model_id,
                    "reference_camera_calibration_id": (
                        observation.reference_camera_calibration_id
                    ),
                    "coordinate_space": observation.coordinate_space,
                    "pixel_uncertainty_px": pixel_sigma,
                    "observation": dict(observation.provenance),
                }
            ),
        )
        result = GeoObjectProjectionResult(
            source_frame_id,
            observation.object_id,
            True,
            "accepted",
            located,
        )
        return self._store_object_result(result)

    @_mission_locked
    def snapshot(self) -> MosaicSnapshot:
        return self._grid.snapshot()

    def _bounds_geojson(self) -> dict[str, Any]:
        config = self.config
        corners_enu = (
            (config.east_min_m, config.north_min_m, 0.0),
            (config.east_max_m, config.north_min_m, 0.0),
            (config.east_max_m, config.north_max_m, 0.0),
            (config.east_min_m, config.north_max_m, 0.0),
            (config.east_min_m, config.north_min_m, 0.0),
        )
        ring = []
        for corner in corners_enu:
            point = enu_to_geodetic(np.asarray(corner), self.local_frame)
            ring.append([point.longitude_deg, point.latitude_deg])
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "mission_id": config.mission_id,
                        "kind": "configured_grid_bounds_not_scanned_coverage",
                        "vertical_datum": "WGS84 ellipsoid",
                        "ground_elevation_msl_m": config.ground_elevation_msl_m,
                    },
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                }
            ],
        }

    @_mission_locked
    @_bundle_publication_locked
    @_cleanup_failed_staging
    def save_bundle(
        self,
        directory: str | Path,
        *,
        overwrite: bool = False,
        metadata_additions: Mapping[str, Any] | None = None,
        preview_warning: str | None = None,
    ) -> AtlasBundlePaths:
        """Publish an immutable generation and atomically swap ``CURRENT``."""

        destination = Path(directory).resolve()
        current_pointer = destination / "CURRENT"
        legacy_names = (
            "atlas_layers.npz",
            "atlas_metadata.json",
            "atlas_rgb.png",
            "atlas_thermal.png",
            "atlas_support.png",
            "atlas_bounds.geojson",
            "atlas_objects.geojson",
            "odm_geo.txt",
        )
        existing = [destination / name for name in legacy_names if (destination / name).exists()]
        if (current_pointer.exists() or existing) and not overwrite:
            detail = ["CURRENT"] if current_pointer.exists() else []
            detail.extend(path.name for path in existing)
            raise FileExistsError(
                "atlas bundle already contains: " + ", ".join(detail)
            )

        generation_id = uuid.uuid4().hex
        published_at_utc = datetime.now(timezone.utc).isoformat()
        generation_parent = destination / ".generations"
        generation_directory = generation_parent / generation_id
        final_paths = AtlasBundlePaths(
            directory=generation_directory,
            arrays=generation_directory / "atlas_layers.npz",
            metadata=generation_directory / "atlas_metadata.json",
            rgb_preview=generation_directory / "atlas_rgb.png",
            thermal_preview=generation_directory / "atlas_thermal.png",
            support_preview=generation_directory / "atlas_support.png",
            bounds_geojson=generation_directory / "atlas_bounds.geojson",
            objects_geojson=generation_directory / "atlas_objects.geojson",
            odm_geo=generation_directory / "odm_geo.txt",
            manifest=generation_directory / "bundle_manifest.json",
        )
        generation_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(
            prefix=f".staging-{generation_id}-",
            dir=generation_parent,
        ))
        self._active_bundle_staging = staging
        paths = AtlasBundlePaths(
            directory=staging,
            arrays=staging / final_paths.arrays.name,
            metadata=staging / final_paths.metadata.name,
            rgb_preview=staging / final_paths.rgb_preview.name,
            thermal_preview=staging / final_paths.thermal_preview.name,
            support_preview=staging / final_paths.support_preview.name,
            bounds_geojson=staging / final_paths.bounds_geojson.name,
            objects_geojson=staging / final_paths.objects_geojson.name,
            odm_geo=staging / final_paths.odm_geo.name,
            manifest=staging / final_paths.manifest.name,
        )

        snapshot = self.snapshot()
        mission_source_ids = tuple(self._accepted_context)
        if snapshot.source_ids != mission_source_ids:
            raise RuntimeError(
                "mosaic provenance does not match mission-gated capture records"
            )
        if len(self._accepted_records) != len(mission_source_ids):
            raise RuntimeError(
                "accepted capture records and mission contexts are inconsistent"
            )
        array_layers = dict(snapshot.array_layers)
        if self._grid.config.surface_elevation_enu_m is not None:
            array_layers["input_surface_elevation_enu_m"] = np.asarray(
                self._grid.config.surface_elevation_enu_m
            )
            if np.ndim(self._grid.config.surface_elevation_sigma_m) > 0:
                array_layers["input_surface_elevation_sigma_m"] = np.asarray(
                    self._grid.config.surface_elevation_sigma_m
                )
        np.savez_compressed(paths.arrays, **array_layers)

        valid = np.asarray(snapshot.valid)
        rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)
        rgb[valid] = np.clip(
            np.rint(np.asarray(snapshot.rgb)[valid] * 255.0), 0, 255
        ).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(paths.rgb_preview, format="PNG")

        thermal = np.asarray(snapshot.thermal_normalized)
        thermal_valid = np.isfinite(thermal)
        thermal_preview = np.zeros((*valid.shape, 3), dtype=np.uint8)
        if np.any(thermal_valid):
            value = np.clip(thermal[thermal_valid], 0.0, 1.0)
            thermal_preview[thermal_valid, 0] = np.rint(255.0 * value).astype(np.uint8)
            thermal_preview[thermal_valid, 1] = np.rint(
                255.0 * np.sqrt(value)
            ).astype(np.uint8)
            thermal_preview[thermal_valid, 2] = np.rint(
                96.0 * (1.0 - value)
            ).astype(np.uint8)
        Image.fromarray(thermal_preview, mode="RGB").save(
            paths.thermal_preview, format="PNG"
        )

        thermal_has_evidence = bool(
            np.any(np.asarray(snapshot.thermal_weight_sum) > 0.0)
        )
        if thermal_has_evidence:
            support = np.minimum(
                np.asarray(snapshot.rgb_support),
                np.asarray(snapshot.thermal_support),
            )
            support_preview_kind = "joint_rgb_thermal_support_minimum"
        else:
            support = np.asarray(snapshot.rgb_support)
            support_preview_kind = "rgb_support_no_thermal_evidence"
        support = np.clip(support, 0.0, 1.0)
        support_preview = np.zeros((*valid.shape, 3), dtype=np.uint8)
        support_preview[..., 0] = np.rint(255.0 * (1.0 - support)).astype(np.uint8)
        support_preview[..., 1] = np.rint(255.0 * support).astype(np.uint8)
        support_preview[..., 2] = np.rint(180.0 * support).astype(np.uint8)
        support_preview[~valid] = 0
        Image.fromarray(support_preview, mode="RGB").save(
            paths.support_preview, format="PNG"
        )
        if preview_warning is not None:
            for preview_path in (
                paths.rgb_preview,
                paths.thermal_preview,
                paths.support_preview,
            ):
                _stamp_preview_warning(preview_path, preview_warning)

        preflight_rejections = [
            {"image_name": item.image_name, "reason": item.reason}
            for item in self._results
            if not item.accepted
        ]
        object_rejections = [
            {
                "source_frame_id": item.source_frame_id,
                "object_id": item.object_id,
                "reason": item.reason,
            }
            for item in self._object_results
            if not item.accepted
        ]
        metadata = {
            "schema_version": "openprism.atlas-bundle/0.2",
            "profile": "openprism_reference_runtime_not_portable_standard",
            "generation_id": generation_id,
            "publication": {
                "published_at_utc": published_at_utc,
                "temporal_role": "immutable_mission_snapshot",
                "operator_freshness_ttl_s": self.config.operator_freshness_ttl_s,
            },
            "mission_id": self.config.mission_id,
            "product": (
                "live_tactical_2.5d_height_field_mosaic"
                if self._grid.config.surface_elevation_enu_m is not None
                else "live_tactical_flat_ground_mosaic"
            ),
            "survey_grade": False,
            "terrain_reconstruction": False,
            "geometry_role": (
                "orthodrape_onto_supplied_height_field"
                if self._grid.config.surface_elevation_enu_m is not None
                else "orthodrape_onto_configured_horizontal_plane"
            ),
            "data_provenance": {
                "real_sensor_data": self.config.real_sensor_data,
                "real_navigation_data": self.config.real_navigation_data,
                "classification": (
                    "captured_evidence"
                    if self.config.real_sensor_data is True
                    and self.config.real_navigation_data is True
                    else "synthetic"
                    if self.config.real_sensor_data is False
                    or self.config.real_navigation_data is False
                    else "unverified"
                ),
            },
            "mission_contract": {
                "origin": {
                    "latitude_deg": self.config.origin_latitude_deg,
                    "longitude_deg": self.config.origin_longitude_deg,
                    "ground_elevation_msl_m": (
                        self.config.ground_elevation_msl_m
                    ),
                    "horizontal_crs_id": self.config.horizontal_crs_id,
                    "input_vertical_datum_id": (
                        self.config.input_vertical_datum_id
                    ),
                    "geoid_model_id": self.config.geoid_model_id,
                    "geoid_separation_m": self.config.geoid_separation_m,
                    "geoid_separation_uncertainty_m": (
                        self.config.geoid_separation_uncertainty_m
                    ),
                    "geoid_separation_source": (
                        "operator_supplied_from_declared_geoid_model"
                    ),
                },
                "bounds_kind": "configured_grid_bounds_not_scanned_coverage",
                "bounds_enu_m": {
                    "east": [self.config.east_min_m, self.config.east_max_m],
                    "north": [self.config.north_min_m, self.config.north_max_m],
                },
                "resolution_m": self.config.resolution_m,
                "time": {
                    "tai_minus_utc_s": self.config.tai_minus_utc_s,
                    "boot_epoch_tai_ns": self.config.boot_epoch_tai_ns,
                    "boot_epoch_uncertainty_ns": (
                        self.config.boot_epoch_uncertainty_ns
                    ),
                    "boot_clock_domain": self.config.boot_clock_domain,
                    "max_time_uncertainty_ns": (
                        self.config.max_time_uncertainty_ns
                    ),
                },
                "navigation_gates": {
                    "minimum_fix_type": self.config.minimum_fix_type,
                    "require_rtk_fixed": self.config.require_rtk_fixed,
                    "minimum_agl_m": self.config.minimum_agl_m,
                    "maximum_agl_m": self.config.maximum_agl_m,
                    "default_horizontal_accuracy_m": (
                        self.config.default_horizontal_accuracy_m
                    ),
                    "default_vertical_accuracy_m": (
                        self.config.default_vertical_accuracy_m
                    ),
                    "default_attitude_accuracy_deg": (
                        self.config.default_attitude_accuracy_deg
                    ),
                },
                "thermal_gates": {
                    "max_time_offset_ns": self.config.max_thermal_time_offset_ns,
                    "max_time_uncertainty_ns": (
                        self.config.max_thermal_time_uncertainty_ns
                    ),
                    "max_registration_rms_px": (
                        self.config.max_thermal_registration_rms_px
                    ),
                },
                "object_layer": {
                    "retention_ns": self.config.object_retention_ns,
                    "max_observations": self.config.max_object_observations,
                    "embedded_in_static_mosaic": False,
                },
            },
            "active_layer_contracts": {
                "thermal_mission_fingerprint": _jsonable(
                    self._thermal_mission_fingerprint
                ),
                "semantic_mission_fingerprint": _jsonable(
                    self._semantic_mission_fingerprint
                ),
            },
            "camera_calibration_id": self.intrinsics.calibration_id,
            "camera_calibration": {
                "calibration_id": self.intrinsics.calibration_id,
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
                "fx_px": self.intrinsics.fx_px,
                "fy_px": self.intrinsics.fy_px,
                "cx_px": self.intrinsics.cx_px,
                "cy_px": self.intrinsics.cy_px,
                "calibration_rms_px": self.intrinsics.calibration_rms_px,
                "distortion_model": self.intrinsics.distortion_model,
            },
            "accepted_capture_count": len(self._accepted_records),
            "accepted_capture_records": [
                record.as_dict() for record in self._accepted_records
            ],
            "rejected_capture_count": len(preflight_rejections),
            "mission_rejections": preflight_rejections,
            "mosaic": _jsonable(snapshot.metadata),
            "layers": sorted(array_layers),
            "source_ids": list(snapshot.source_ids),
            "source_provenance": _jsonable(snapshot.provenance),
            "geolocated_object_count": sum(
                item.accepted for item in self._object_results
            ),
            "rejected_object_count": sum(
                not item.accepted for item in self._object_results
            ),
            "object_rejections": object_rejections,
            "source_image_policy": {
                "mode": "external_references_only",
                "images_embedded_in_bundle": False,
                "warning": (
                    "Retain immutable source images separately; this tactical "
                    "bundle is a derived cache, not a replay archive."
                ),
            },
            "odm_geo_export": {
                "kind": "position_metadata_only_not_runnable_odm_project",
                "images_staged": False,
                "orientation_serialized": False,
                "accuracy_serialized": False,
                "full_pose_and_accuracy_location": (
                    "atlas_metadata.json.accepted_capture_records"
                ),
            },
            "write_contract": "transactionally_staged_metadata_committed_last",
            "support_preview_kind": support_preview_kind,
        }
        if metadata_additions:
            collisions = set(metadata) & set(metadata_additions)
            if collisions:
                raise ValueError(
                    "metadata_additions cannot replace reserved keys: "
                    + ", ".join(sorted(str(key) for key in collisions))
                )
            metadata.update(dict(metadata_additions))
        metadata = _jsonable(metadata)
        paths.metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths.bounds_geojson.write_text(
            json.dumps(
                self._bounds_geojson(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        paths.objects_geojson.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        item.observation.as_geojson_feature()
                        for item in self._object_results
                        if item.observation is not None
                    ],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        paths.odm_geo.write_text(
            export_odm_geo_txt(self._accepted_records),
            encoding="utf-8",
        )
        artifact_paths = (
            paths.arrays,
            paths.metadata,
            paths.rgb_preview,
            paths.thermal_preview,
            paths.support_preview,
            paths.bounds_geojson,
            paths.objects_geojson,
            paths.odm_geo,
        )
        manifest = {
            "schema_version": "openprism.atlas-manifest/0.1",
            "generation_id": generation_id,
            "files": {
                path.name: {
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in artifact_paths
            },
        }
        paths.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest_sha256 = _sha256_file(paths.manifest)

        # The immutable directory becomes visible first; a single atomic
        # pointer swap then publishes the complete generation. Old generations
        # remain recoverable and concurrent readers never see mixed files.
        os.replace(staging, generation_directory)
        self._active_bundle_staging = None
        pointer_payload = {
            "schema_version": "openprism.atlas-current/0.1",
            "generation_id": generation_id,
            "manifest_sha256": manifest_sha256,
        }
        pointer_temporary = destination / f".CURRENT-{generation_id}.tmp"
        with pointer_temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(pointer_payload, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pointer_temporary, current_pointer)
        return final_paths


__all__ = [
    "AtlasBundlePaths",
    "AtlasFrameResult",
    "AtlasMissionConfig",
    "AtlasMissionMapper",
    "GeoObjectObservation",
    "GeoObjectProjectionResult",
    "ImageObjectObservation",
    "RegisteredThermalFrameContract",
    "SemanticFrameContract",
]
