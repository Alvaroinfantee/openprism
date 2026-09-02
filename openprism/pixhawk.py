"""Pixhawk/MAVLink capture-pose bridge for OpenPRISM mapping.

The module intentionally has no import-time dependency on :mod:`pymavlink`.
It accepts normalized MAVLink-style dictionaries so recorded telemetry, unit
tests, and a live MAVLink connection all pass through the same strict timing
and coordinate-frame boundary.

Coordinate conventions
----------------------
``ATTITUDE_QUATERNION`` and ``CAMERA_IMAGE_CAPTURED.q`` are interpreted as
Hamilton quaternions in ``(w, x, y, z)`` order that rotate coordinates from
MAVLink ``LOCAL_NED`` (north, east, down) into camera/vehicle ``BODY_FRD``
(front, right, down).  This is the MAVSDK convention for MAVLink attitude.
OpenPRISM exposes a pose quaternion that rotates camera ``FLU`` coordinates
(front, left, up) into local ``ENU`` coordinates (east, north, up).  The
conversion is a full basis change, not a component shuffle.

Time conventions
----------------
Boot/monotonic time and UTC are retained as separate fields.  Interpolation is
never performed between them unless the input itself supplies the requested
time in a common domain.  No offset is inferred from message arrival order.
This matters because GPS, attitude, and imagery normally run at different
sample rates and may be transported with unrelated latency.

Primary references:
* https://mavlink.io/en/messages/common.html
* https://mavsdk.mavlink.io/main/en/cpp/api_reference/structmavsdk_1_1_quaternion.html
* https://docs.opendronemap.org/geo/
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlparse


MAVLINK_ATTITUDE_CONVENTION = (
    "Hamilton wxyz coordinate rotation MAV_FRAME_LOCAL_NED -> "
    "MAV_FRAME_BODY_FRD/camera_FRD"
)
OPENPRISM_ATTITUDE_CONVENTION = (
    "Hamilton wxyz coordinate rotation camera_FLU -> local_ENU"
)
OPENPRISM_OPTICAL_ATTITUDE_CONVENTION = (
    "Hamilton wxyz coordinate rotation camera_OpenCV_optical -> local_ENU"
)
GEODETIC_POSITION_FRAME = (
    "horizontal=EPSG:4326 (WGS84 latitude/longitude); "
    "vertical=MSL height (datum/geoid declared by mission)"
)
DECLARED_CAMERA_IMAGE_ATTITUDE_PROFILE = (
    "declared_hamilton_wxyz_local_ned_to_camera_frd"
)


class PixhawkBridgeError(ValueError):
    """Base class for rejected Pixhawk bridge input."""


class MessageFormatError(PixhawkBridgeError):
    """A required MAVLink field is absent, invalid, or contradictory."""


class TimingError(PixhawkBridgeError):
    """A capture cannot be related to telemetry in time."""


class TimingDomainError(TimingError):
    """Samples use unknown or incompatible time bases/clock domains."""


class InterpolationError(TimingError):
    """Telemetry does not safely bracket a requested exposure time."""


class OptionalDependencyError(RuntimeError):
    """An explicitly requested optional integration is unavailable."""


def _exact_nonnegative_int(
    value: Any,
    field: str,
    *,
    error_type: type[ValueError] = ValueError,
) -> int:
    """Normalize an integer-valued identifier without lossy ``int()`` coercion."""

    if isinstance(value, bool):
        raise error_type(f"{field} must be a non-negative integer")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise error_type(f"{field} must be a non-negative integer") from error
    if (
        not decimal.is_finite()
        or decimal != decimal.to_integral_value()
        or decimal < 0
    ):
        raise error_type(f"{field} must be a non-negative integer")
    return int(decimal)


@dataclass(frozen=True, slots=True)
class ImageReference:
    """An image that should be matched to a MAVLink capture event.

    A timestamp is optional when a capture message supplies the event time.
    When provided, ``clock_domain`` and exactly one of the two time fields are
    required; this prevents a numeric timestamp from silently acquiring the
    wrong epoch.
    """

    filename: str
    image_index: int | None = None
    capture_monotonic_ns: int | None = None
    capture_utc_ns: int | None = None
    clock_domain: str | None = None

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError("image filename is required")
        if self.image_index is not None:
            object.__setattr__(
                self,
                "image_index",
                _exact_nonnegative_int(self.image_index, "image_index"),
            )
        for name in ("capture_monotonic_ns", "capture_utc_ns"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _exact_nonnegative_int(value, name),
                )
        if (
            self.capture_monotonic_ns is not None
            and self.capture_utc_ns is not None
        ):
            raise ValueError(
                "an image reference may specify monotonic or UTC time, not both"
            )
        if (
            self.capture_monotonic_ns is not None
            or self.capture_utc_ns is not None
        ) and not self.clock_domain:
            raise ValueError("a timestamped image reference requires clock_domain")


@dataclass(frozen=True, slots=True)
class CameraPoseRecord:
    """JSON-ready geodetic camera pose associated with one image.

    ``capture_*`` values identify the calibrated exposure instant; ``event_*``
    values retain the unmodified source timestamp.  The UTC and monotonic
    values are separate on purpose and may both be populated by
    ``CAMERA_IMAGE_CAPTURED``.
    """

    image_name: str
    image_index: int | None
    latitude_deg: float
    longitude_deg: float
    altitude_msl_m: float
    relative_altitude_m: float | None
    quaternion_camera_flu_to_enu_wxyz: tuple[float, float, float, float]
    quaternion_camera_optical_to_enu_wxyz: tuple[float, float, float, float]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    capture_monotonic_ns: int | None
    capture_utc_ns: int | None
    event_monotonic_ns: int | None
    event_utc_ns: int | None
    clock_domain: str
    time_basis: str
    time_uncertainty_ns: int | None
    horizontal_accuracy_m: float | None
    vertical_accuracy_m: float | None
    attitude_accuracy_deg: float | None
    fix_type: int | str | None
    fix_quality: str
    rtk_status: str
    source_message: str
    position_source: str
    attitude_source: str
    interpolation_span_ns: int | None
    position_frame: str = GEODETIC_POSITION_FRAME
    relative_altitude_reference: str = "unspecified"
    position_reference: str = "unspecified"
    input_attitude_profile: str = "externally_constructed_record"
    input_attitude_convention: str = MAVLINK_ATTITUDE_CONVENTION
    attitude_convention: str = OPENPRISM_ATTITUDE_CONVENTION
    camera_axis_convention: str = "FLU: x-forward, y-left, z-up"
    image_match_basis: str = "unspecified"
    system_id: int | None = None
    component_id: int | None = None
    camera_id: int | None = None

    def __post_init__(self) -> None:
        def finite_float(name: str, value: Any) -> float:
            try:
                result = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be finite") from error
            if not math.isfinite(result):
                raise ValueError(f"{name} must be finite")
            return result

        def optional_nonnegative_int(name: str, value: Any) -> int | None:
            return (
                None
                if value is None
                else _exact_nonnegative_int(value, name)
            )

        image_name = str(self.image_name).strip()
        if not image_name:
            raise ValueError("image_name is required")
        object.__setattr__(self, "image_name", image_name)
        image_index = optional_nonnegative_int("image_index", self.image_index)
        object.__setattr__(self, "image_index", image_index)

        for name in (
            "latitude_deg",
            "longitude_deg",
            "altitude_msl_m",
            "yaw_deg",
            "pitch_deg",
            "roll_deg",
        ):
            object.__setattr__(self, name, finite_float(name, getattr(self, name)))
        if self.relative_altitude_m is not None:
            object.__setattr__(
                self,
                "relative_altitude_m",
                finite_float("relative_altitude_m", self.relative_altitude_m),
            )
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("latitude must be in [-90, 90]")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("longitude must be in [-180, 180]")

        for name in (
            "quaternion_camera_flu_to_enu_wxyz",
            "quaternion_camera_optical_to_enu_wxyz",
        ):
            try:
                quaternion = tuple(
                    finite_float(name, value) for value in getattr(self, name)
                )
            except TypeError as error:
                raise ValueError(f"{name} must contain four finite values") from error
            if len(quaternion) != 4:
                raise ValueError(f"{name} must contain four finite values")
            norm = math.sqrt(sum(value * value for value in quaternion))
            if abs(norm - 1.0) > 1e-6:
                raise ValueError("camera quaternion must be normalized")
            object.__setattr__(self, name, quaternion)
        r_enu_from_flu = _quaternion_to_matrix(
            self.quaternion_camera_flu_to_enu_wxyz
        )
        r_flu_from_optical = (
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        )
        expected_optical = _matmul(r_enu_from_flu, r_flu_from_optical)
        actual_optical = _quaternion_to_matrix(
            self.quaternion_camera_optical_to_enu_wxyz
        )
        if max(
            abs(expected_optical[row][column] - actual_optical[row][column])
            for row in range(3)
            for column in range(3)
        ) > 1e-6:
            raise ValueError("FLU and optical camera quaternions contradict")
        for name in (
            "horizontal_accuracy_m",
            "vertical_accuracy_m",
            "attitude_accuracy_deg",
        ):
            value = getattr(self, name)
            if value is not None:
                value = finite_float(name, value)
                if value < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative")
                object.__setattr__(self, name, value)

        for name in (
            "capture_monotonic_ns",
            "capture_utc_ns",
            "event_monotonic_ns",
            "event_utc_ns",
            "time_uncertainty_ns",
            "interpolation_span_ns",
            "system_id",
            "component_id",
            "camera_id",
        ):
            object.__setattr__(
                self,
                name,
                optional_nonnegative_int(name, getattr(self, name)),
            )
        if self.time_basis not in {
            "mavlink_system_boot",
            "utc",
            "mavlink_system_boot+utc",
        }:
            raise ValueError("unsupported time_basis")
        if self.capture_monotonic_ns is None and self.capture_utc_ns is None:
            raise ValueError("at least one capture time is required")
        expected_time_presence = {
            "mavlink_system_boot": (True, False),
            "utc": (False, True),
            "mavlink_system_boot+utc": (True, True),
        }[self.time_basis]
        actual_time_presence = (
            self.capture_monotonic_ns is not None,
            self.capture_utc_ns is not None,
        )
        if actual_time_presence != expected_time_presence:
            raise ValueError("time_basis contradicts populated capture timestamps")
        clock_domain = str(self.clock_domain).strip()
        if not clock_domain:
            raise ValueError("clock_domain is required")
        object.__setattr__(self, "clock_domain", clock_domain)

        fix_type = self.fix_type
        if isinstance(fix_type, str):
            fix_type = fix_type.strip().lower().replace(" ", "_")
            if not fix_type:
                raise ValueError("fix_type cannot be blank")
        elif fix_type is not None:
            fix_type = optional_nonnegative_int("fix_type", fix_type)
            if fix_type is not None and fix_type not in _FIX_LABELS:
                raise ValueError("numeric fix_type must be a MAVLink GPS_FIX_TYPE value")
        object.__setattr__(self, "fix_type", fix_type)
        for name in (
            "fix_quality",
            "rtk_status",
            "source_message",
            "position_source",
            "attitude_source",
            "image_match_basis",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        # These labels are redundant derivatives of GPS_FIX_TYPE.  Canonicalize
        # rather than trusting deserialized free text that could bypass an RTK
        # gate downstream.
        canonical_quality, canonical_rtk = _fix_quality(fix_type)
        object.__setattr__(self, "fix_quality", canonical_quality)
        object.__setattr__(self, "rtk_status", canonical_rtk)

        if self.position_frame != GEODETIC_POSITION_FRAME:
            raise ValueError("CameraPoseRecord position_frame is fixed by its contract")
        if self.relative_altitude_reference not in {
            "ground",
            "home",
            "unspecified",
            "not_available",
        }:
            raise ValueError("unsupported relative_altitude_reference")
        if self.position_reference not in {
            "camera_optical_center",
            "vehicle_navigation_origin",
            "unspecified",
        }:
            raise ValueError("unsupported position_reference")
        input_profile = str(self.input_attitude_profile).strip()
        if not input_profile:
            raise ValueError("input_attitude_profile is required")
        object.__setattr__(self, "input_attitude_profile", input_profile)
        if self.input_attitude_convention != MAVLINK_ATTITUDE_CONVENTION:
            raise ValueError("input_attitude_convention contradicts the bridge contract")
        if self.attitude_convention != OPENPRISM_ATTITUDE_CONVENTION:
            raise ValueError("attitude_convention contradicts the bridge contract")
        if self.camera_axis_convention != "FLU: x-forward, y-left, z-up":
            raise ValueError("camera_axis_convention contradicts the bridge contract")

    def as_dict(self) -> dict[str, Any]:
        """Return a record containing only JSON-native values."""

        return {
            "image_name": self.image_name,
            "image_index": self.image_index,
            "geodetic": {
                "latitude_deg": self.latitude_deg,
                "longitude_deg": self.longitude_deg,
                "altitude_msl_m": self.altitude_msl_m,
                "relative_altitude_m": self.relative_altitude_m,
                "relative_altitude_reference": self.relative_altitude_reference,
                "frame": self.position_frame,
                "position_reference": self.position_reference,
            },
            "camera_pose": {
                # The generic quaternion follows the OpenCV optical convention
                # consumed by openprism.mapping. The named FLU representation
                # is retained for robotics/vehicle-frame consumers.
                "quaternion_wxyz": list(
                    self.quaternion_camera_optical_to_enu_wxyz
                ),
                "quaternion_camera_optical_to_enu_wxyz": list(
                    self.quaternion_camera_optical_to_enu_wxyz
                ),
                "quaternion_camera_flu_to_enu_wxyz": list(
                    self.quaternion_camera_flu_to_enu_wxyz
                ),
                "yaw_deg": self.yaw_deg,
                "pitch_deg": self.pitch_deg,
                "roll_deg": self.roll_deg,
                "convention": OPENPRISM_OPTICAL_ATTITUDE_CONVENTION,
                "optical_axes": "OpenCV: x-right, y-down, z-forward",
                "flu_convention": self.attitude_convention,
                "flu_axes": self.camera_axis_convention,
            },
            "capture_time": {
                "monotonic_ns": self.capture_monotonic_ns,
                "utc_ns": self.capture_utc_ns,
                "event_monotonic_ns": self.event_monotonic_ns,
                "event_utc_ns": self.event_utc_ns,
                "basis": self.time_basis,
                "clock_domain": self.clock_domain,
                "uncertainty_ns": self.time_uncertainty_ns,
            },
            "uncertainty": {
                "horizontal_m": self.horizontal_accuracy_m,
                "vertical_m": self.vertical_accuracy_m,
                "attitude_deg": self.attitude_accuracy_deg,
            },
            "quality": {
                "fix_type": self.fix_type,
                "fix_quality": self.fix_quality,
                "rtk_status": self.rtk_status,
            },
            "provenance": {
                "source_message": self.source_message,
                "position_source": self.position_source,
                "attitude_source": self.attitude_source,
                "input_attitude_convention": self.input_attitude_convention,
                "input_attitude_profile": self.input_attitude_profile,
                "interpolation_span_ns": self.interpolation_span_ns,
                "image_match_basis": self.image_match_basis,
                "system_id": self.system_id,
                "component_id": self.component_id,
                "camera_id": self.camera_id,
            },
        }


@dataclass(frozen=True, slots=True)
class PixhawkBridgeConfig:
    """Timing, mounting, and matching policy for :class:`PixhawkBridge`."""

    # Delay from a CAMERA_TRIGGER command/event to optical exposure.  This is
    # intentionally *not* applied to CAMERA_IMAGE_CAPTURED, whose timestamp and
    # pose already describe the captured image, or to a timestamped image
    # reference supplied as an exposure record.
    shutter_lag_ms: float = 0.0
    shutter_lag_uncertainty_ms: float = 0.0
    # Optional signed calibration correction for sources that already report
    # exposure time.  Leave at zero unless measured against a common clock.
    captured_event_time_correction_ms: float = 0.0
    captured_event_time_uncertainty_ms: float = 0.0
    max_interpolation_gap_ms: float = 250.0
    # Hard local-navigation-frame dynamics bounds used to bound deviation from
    # the constant-velocity/geodesic interpolation models. A genuine temporal
    # interpolation is rejected unless the corresponding bound is declared.
    # Zero is an explicit constant-velocity/angular-velocity assumption.
    position_acceleration_bound_mps2: float | None = None
    angular_acceleration_bound_deg_s2: float | None = None
    # Coordinate rotation BODY_FRD -> camera_FRD. Identity means a forward,
    # rigidly aligned camera; a mapping deployment should supply calibration.
    camera_from_body_quaternion_wxyz: tuple[float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    require_image_match: bool = True
    system_id: int | None = None
    camera_id: int | None = None
    capture_component_id: int | None = None
    navigation_component_id: int | None = None
    gps_message_type: str | None = None
    # Standard CAMERA_TRIGGER has no camera-id field. Set this only when the
    # autopilot trigger output is physically routed to one known payload.
    unidentified_trigger_camera_id: int | None = None
    # CAMERA_IMAGE_CAPTURED does not standardize q direction/reference point.
    # A producer profile must therefore declare both before direct pose use.
    camera_image_attitude_profile: str | None = None
    camera_image_position_reference: str = "unspecified"

    def __post_init__(self) -> None:
        if not math.isfinite(self.shutter_lag_ms):
            raise ValueError("shutter_lag_ms must be finite")
        if (
            not math.isfinite(self.shutter_lag_uncertainty_ms)
            or self.shutter_lag_uncertainty_ms < 0.0
        ):
            raise ValueError("shutter_lag_uncertainty_ms must be non-negative")
        if not math.isfinite(self.captured_event_time_correction_ms):
            raise ValueError("captured_event_time_correction_ms must be finite")
        if (
            not math.isfinite(self.captured_event_time_uncertainty_ms)
            or self.captured_event_time_uncertainty_ms < 0.0
        ):
            raise ValueError(
                "captured_event_time_uncertainty_ms must be non-negative"
            )
        if (
            not math.isfinite(self.max_interpolation_gap_ms)
            or self.max_interpolation_gap_ms <= 0.0
        ):
            raise ValueError("max_interpolation_gap_ms must be positive")
        for name in (
            "position_acceleration_bound_mps2",
            "angular_acceleration_bound_deg_s2",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite and non-negative")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{name} must be finite and non-negative"
                ) from error
            if not math.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, normalized)
        _normalize_quaternion(self.camera_from_body_quaternion_wxyz)
        if (
            self.camera_image_attitude_profile is not None
            and self.camera_image_attitude_profile
            != DECLARED_CAMERA_IMAGE_ATTITUDE_PROFILE
        ):
            raise ValueError("unsupported camera_image_attitude_profile")
        if self.camera_image_position_reference not in {
            "camera_optical_center",
            "vehicle_navigation_origin",
            "unspecified",
        }:
            raise ValueError("unsupported camera_image_position_reference")
        for name in (
            "system_id",
            "camera_id",
            "capture_component_id",
            "navigation_component_id",
            "unidentified_trigger_camera_id",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _exact_nonnegative_int(value, name),
                )
        if self.gps_message_type is not None:
            gps_message_type = str(self.gps_message_type).strip().upper()
            if gps_message_type not in {"GPS_RAW_INT", "GPS2_RAW"}:
                raise ValueError(
                    "gps_message_type must be GPS_RAW_INT or GPS2_RAW"
                )
            object.__setattr__(self, "gps_message_type", gps_message_type)


@dataclass(frozen=True, slots=True)
class _TimeMark:
    monotonic_ns: int | None
    utc_ns: int | None
    boot_domain: str | None
    uncertainty_ns: int | None

    @property
    def time_basis(self) -> str:
        if self.monotonic_ns is not None and self.utc_ns is not None:
            return "mavlink_system_boot+utc"
        if self.monotonic_ns is not None:
            return "mavlink_system_boot"
        if self.utc_ns is not None:
            return "utc"
        raise TimingError("message has no usable timestamp")


@dataclass(frozen=True, slots=True)
class _TimedSample:
    time: _TimeMark
    message: Mapping[str, Any]


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MessageFormatError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise MessageFormatError(f"{field} must be finite")
    return result


def _optional_nonnegative(value: Any, field: str) -> float | None:
    if value is None:
        return None
    result = _finite_float(value, field)
    if result < 0.0:
        raise MessageFormatError(f"{field} must be non-negative")
    return result


def _message_type(message: Mapping[str, Any]) -> str:
    for key in ("mavpackettype", "message_type", "_type", "type", "name"):
        value = message.get(key)
        if value:
            return str(value).upper()
    raise MessageFormatError("MAVLink dictionary is missing a message type")


def _source_identifier(
    message: Mapping[str, Any],
    keys: Sequence[str],
    field: str,
) -> int | None:
    for key in keys:
        value = message.get(key)
        if value is None:
            continue
        return _exact_nonnegative_int(
            value,
            field,
            error_type=MessageFormatError,
        )
    return None


def _source_system_id(message: Mapping[str, Any]) -> int | None:
    return _source_identifier(
        message,
        ("_srcSystem", "source_system", "system_id", "sysid"),
        "source system id",
    )


def _source_component_id(message: Mapping[str, Any]) -> int | None:
    return _source_identifier(
        message,
        ("_srcComponent", "source_component", "component_id", "compid"),
        "source component id",
    )


def _source_system(message: Mapping[str, Any]) -> str:
    value = _source_system_id(message)
    return "unknown" if value is None else str(value)


def _boot_domain(message: Mapping[str, Any]) -> str:
    explicit = message.get("clock_domain") or message.get("boot_clock_domain")
    if explicit:
        return str(explicit)
    system = _source_system(message)
    component = _source_component_id(message)
    if component is None:
        return f"mavlink:system:{system}:boot"
    return f"mavlink:system:{system}:component:{component}:boot"


def _int_ns(value: Any, multiplier: int, field: str) -> int:
    # Do not route epoch-scale integers through IEEE-754; a float conversion
    # can move a capture by hundreds of nanoseconds before multiplication.
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise MessageFormatError(f"{field} must be numeric") from error
    if not number.is_finite():
        raise MessageFormatError(f"{field} must be finite")
    if number < 0:
        raise MessageFormatError(f"{field} must be non-negative")
    result = int(
        (number * Decimal(multiplier)).to_integral_value(rounding=ROUND_HALF_EVEN)
    )
    if result < 0:
        raise MessageFormatError(f"{field} overflows timestamp range")
    return result


def _time_mark(message: Mapping[str, Any]) -> _TimeMark:
    monotonic_ns: int | None = None
    utc_ns: int | None = None
    domain: str | None = None

    if message.get("capture_monotonic_ns") is not None:
        monotonic_ns = _int_ns(
            message["capture_monotonic_ns"], 1, "capture_monotonic_ns"
        )
        domain = _boot_domain(message)
    elif message.get("monotonic_ns") is not None:
        monotonic_ns = _int_ns(message["monotonic_ns"], 1, "monotonic_ns")
        domain = _boot_domain(message)
    elif message.get("time_boot_ms") is not None:
        monotonic_ns = _int_ns(message["time_boot_ms"], 1_000_000, "time_boot_ms")
        domain = _boot_domain(message)

    if message.get("capture_utc_ns") is not None:
        utc_ns = _int_ns(message["capture_utc_ns"], 1, "capture_utc_ns")
    elif message.get("utc_ns") is not None:
        utc_ns = _int_ns(message["utc_ns"], 1, "utc_ns")
    elif message.get("time_utc") not in (None, 0, "0"):
        utc_ns = _int_ns(message["time_utc"], 1_000, "time_utc")
    elif message.get("time_utc_us") not in (None, 0, "0"):
        utc_ns = _int_ns(message["time_utc_us"], 1_000, "time_utc_us")

    # MAVLink time_usec is deliberately ambiguous. Honor an explicit basis;
    # otherwise follow MAVLink's documented magnitude convention. Zero means
    # unknown here (unlike the explicitly boot-relative time_boot_ms field).
    if monotonic_ns is None and utc_ns is None and message.get("time_usec") not in (
        None,
        0,
        "0",
    ):
        value_us = _int_ns(message["time_usec"], 1, "time_usec")
        declared_basis = str(message.get("time_basis", "")).lower()
        if declared_basis in {"boot", "monotonic", "mavlink_system_boot"}:
            monotonic_ns = value_us * 1_000
            domain = _boot_domain(message)
        elif declared_basis in {"utc", "unix", "epoch", "unix_epoch"}:
            utc_ns = value_us * 1_000
        elif not declared_basis:
            if value_us >= 1_000_000_000_000:
                utc_ns = value_us * 1_000
            else:
                monotonic_ns = value_us * 1_000
                domain = _boot_domain(message)
        else:
            raise TimingDomainError(f"unsupported time_basis: {declared_basis}")

    if monotonic_ns is None and utc_ns is None:
        raise TimingError(f"{_message_type(message)} has no usable timestamp")

    uncertainty = message.get("time_uncertainty_ns")
    if uncertainty is not None:
        uncertainty_ns = _int_ns(uncertainty, 1, "time_uncertainty_ns")
    else:
        uncertainty_ns = None
    return _TimeMark(monotonic_ns, utc_ns, domain, uncertainty_ns)


def _normalize_quaternion(values: Sequence[Any]) -> tuple[float, float, float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 4:
        raise MessageFormatError("quaternion must contain four wxyz components")
    q = tuple(_finite_float(value, "quaternion") for value in values)
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-12:
        raise MessageFormatError("zero quaternion is invalid/unknown")
    return tuple(value / norm for value in q)  # type: ignore[return-value]


def quaternion_multiply(
    left: Sequence[Any], right: Sequence[Any]
) -> tuple[float, float, float, float]:
    """Hamilton product of normalized ``wxyz`` quaternions."""

    w1, x1, y1, z1 = _normalize_quaternion(left)
    w2, x2, y2, z2 = _normalize_quaternion(right)
    return _normalize_quaternion(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def quaternion_conjugate(values: Sequence[Any]) -> tuple[float, float, float, float]:
    """Return the inverse of a normalized quaternion."""

    w, x, y, z = _normalize_quaternion(values)
    return (w, -x, -y, -z)


def _quaternion_to_matrix(values: Sequence[Any]) -> tuple[tuple[float, ...], ...]:
    w, x, y, z = _normalize_quaternion(values)
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _matrix_to_quaternion(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    m = matrix
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        q = (
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        q = (
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        q = (
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        )
    normalized = _normalize_quaternion(q)
    # q and -q encode the same rotation. Canonicalize for stable serialization.
    if normalized[0] < 0.0:
        return tuple(-value for value in normalized)  # type: ignore[return-value]
    return normalized


def _matmul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(left[row][k] * right[k][column] for k in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def mavlink_ned_frd_to_openprism_enu_flu(
    quaternion_ned_to_frd_wxyz: Sequence[Any],
) -> tuple[float, float, float, float]:
    """Convert MAVLink NED->FRD attitude to camera FLU->ENU pose.

    The result is the camera orientation expected by OpenPRISM's mapping
    contract.  This function assumes the input already represents the camera;
    compose the vehicle attitude with mount calibration first when using
    ``ATTITUDE_QUATERNION``.
    """

    # Invert the coordinate rotation NED->FRD to obtain FRD->NED.
    r_frd_from_ned = _quaternion_to_matrix(quaternion_ned_to_frd_wxyz)
    r_ned_from_frd = tuple(zip(*r_frd_from_ned))
    # NED -> ENU: (n, e, d) -> (e, n, -d).
    r_enu_from_ned = (
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    # FLU -> FRD: (f, l, u) -> (f, -l, -u).
    r_frd_from_flu = (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    converted = _matmul(
        _matmul(r_enu_from_ned, r_ned_from_frd), r_frd_from_flu
    )
    return _matrix_to_quaternion(converted)


def mavlink_ned_frd_to_openprism_enu_optical(
    quaternion_ned_to_frd_wxyz: Sequence[Any],
) -> tuple[float, float, float, float]:
    """Convert MAVLink NED->camera-FRD to OpenCV-optical->ENU pose.

    OpenPRISM's terrain mapper casts rays in OpenCV optical coordinates
    ``(+right, +down, +forward)``. This result can therefore be used directly
    as the camera-to-ENU orientation after vertical-datum conversion.
    """

    r_enu_from_flu = _quaternion_to_matrix(
        mavlink_ned_frd_to_openprism_enu_flu(quaternion_ned_to_frd_wxyz)
    )
    # OpenCV optical -> FLU: right=-left, down=-up, forward=forward.
    r_flu_from_optical = (
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    )
    return _matrix_to_quaternion(_matmul(r_enu_from_flu, r_flu_from_optical))


def _aeronautical_euler_deg(
    quaternion_ned_to_frd_wxyz: Sequence[Any],
) -> tuple[float, float, float]:
    """Return yaw, pitch, roll of FRD in NED using intrinsic ZYX angles."""

    r_frd_from_ned = _quaternion_to_matrix(quaternion_ned_to_frd_wxyz)
    r = tuple(zip(*r_frd_from_ned))  # FRD -> NED
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(r[2][1], r[2][2])
        yaw = math.atan2(r[1][0], r[0][0])
    else:
        # Gimbal-lock convention: keep roll at zero and retain yaw.
        roll = 0.0
        yaw = math.atan2(-r[0][1], r[1][1])
    return tuple(math.degrees(value) for value in (yaw, pitch, roll))  # type: ignore[return-value]


def _slerp(
    first: Sequence[Any], second: Sequence[Any], fraction: float
) -> tuple[float, float, float, float]:
    q0 = _normalize_quaternion(first)
    q1 = _normalize_quaternion(second)
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = tuple(-value for value in q1)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(
            tuple(a + fraction * (b - a) for a, b in zip(q0, q1))
        )
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    left = math.sin((1.0 - fraction) * theta) / sin_theta
    right = math.sin(fraction * theta) / sin_theta
    return _normalize_quaternion(
        tuple(left * a + right * b for a, b in zip(q0, q1))
    )


def _coordinate(message: Mapping[str, Any], field: str) -> float | None:
    value = message.get(field)
    if value is None or value in (2_147_483_647, -2_147_483_648):
        return None
    coordinate = _finite_float(value, field) / 10_000_000.0
    limit = 90.0 if field == "lat" else 180.0
    if not -limit <= coordinate <= limit:
        raise MessageFormatError(f"{field} is outside WGS84 bounds")
    return coordinate


def _altitude(message: Mapping[str, Any], field: str) -> float | None:
    value = message.get(field)
    if value is None or value in (2_147_483_647, -2_147_483_648):
        return None
    return _finite_float(value, field) / 1_000.0


def _attitude_quaternion(message: Mapping[str, Any]) -> tuple[float, ...] | None:
    kind = _message_type(message)
    if kind == "ATTITUDE_QUATERNION":
        values = tuple(message.get(key) for key in ("q1", "q2", "q3", "q4"))
        if any(value is None for value in values):
            return None
    else:
        values = message.get("q") or message.get("quaternion_wxyz")
        if values is None:
            return None
    try:
        return _normalize_quaternion(values)
    except MessageFormatError:
        return None


def _accuracy(message: Mapping[str, Any], *, attitude: bool = False) -> float | None:
    keys = (
        ("attitude_accuracy_deg", "orientation_accuracy_deg")
        if attitude
        else ("horizontal_accuracy_m", "position_accuracy_m")
    )
    for key in keys:
        if message.get(key) is not None:
            return _optional_nonnegative(message[key], key)
    if attitude:
        component_keys = (
            "roll_accuracy_deg",
            "pitch_accuracy_deg",
            "yaw_accuracy_deg",
        )
        component_values = [message.get(key) for key in component_keys]
        if all(value is not None for value in component_values):
            return max(
                _optional_nonnegative(value, "attitude component accuracy") or 0.0
                for value in component_values
            )
        if any(value is not None for value in component_values):
            return None
    return None


def _vertical_accuracy(message: Mapping[str, Any]) -> float | None:
    for key in ("vertical_accuracy_m", "altitude_accuracy_m"):
        if message.get(key) is not None:
            return _optional_nonnegative(message[key], key)
    return None


def _gps_raw_accuracy(message: Mapping[str, Any], field: str) -> float | None:
    # MAVLink GPS_RAW_INT extension h_acc/v_acc is expressed in millimetres.
    if message.get(field) in (None, 4_294_967_295):
        return None
    value = _optional_nonnegative(message[field], field)
    return None if value is None else value / 1_000.0


_FIX_LABELS = {
    0: "no_gps",
    1: "no_fix",
    2: "2d_fix",
    3: "3d_fix",
    4: "dgps",
    5: "rtk_float",
    6: "rtk_fixed",
    7: "static",
    8: "ppp",
}


def _fix_quality(fix_type: int | str | None) -> tuple[str, str]:
    if fix_type is None:
        return ("unknown", "unknown")
    if isinstance(fix_type, str):
        label = fix_type.strip().lower().replace(" ", "_")
    else:
        label = _FIX_LABELS.get(int(fix_type), f"fix_type_{int(fix_type)}")
    if "rtk_fixed" in label or label == "fixed":
        return (label, "fixed")
    if "rtk_float" in label or label == "float":
        return (label, "float")
    return (label, "not_rtk" if label not in {"unknown", "no_gps", "no_fix"} else label)


def _quality_fields(message: Mapping[str, Any]) -> tuple[int | str | None, str, str]:
    fix_type: int | str | None = message.get("fix_type")
    explicit_quality = message.get("fix_quality") or message.get("gps_fix_quality")
    explicit_rtk = message.get("rtk_status")
    quality, rtk = _fix_quality(fix_type)
    if explicit_quality is not None:
        quality = str(explicit_quality)
    if explicit_rtk is not None:
        rtk = str(explicit_rtk)
    return fix_type, quality, rtk


def _time_for_basis(mark: _TimeMark, basis: str, domain: str) -> int | None:
    if basis == "boot":
        if mark.monotonic_ns is None or mark.boot_domain != domain:
            return None
        return mark.monotonic_ns
    if basis == "utc":
        return mark.utc_ns
    raise AssertionError(basis)


def _choose_interpolation_basis(
    target: _TimeMark, samples: Sequence[_TimedSample]
) -> tuple[str, str, int]:
    if target.monotonic_ns is not None:
        if not target.boot_domain:
            raise TimingDomainError("boot timestamp has no clock domain")
        matching = [
            sample
            for sample in samples
            if sample.time.monotonic_ns is not None
            and sample.time.boot_domain == target.boot_domain
        ]
        if matching:
            return "boot", target.boot_domain, target.monotonic_ns
    if target.utc_ns is not None and any(
        sample.time.utc_ns is not None for sample in samples
    ):
        return "utc", "UTC", target.utc_ns

    available_boot_domains = sorted(
        {
            sample.time.boot_domain
            for sample in samples
            if sample.time.monotonic_ns is not None and sample.time.boot_domain
        }
    )
    if target.monotonic_ns is not None and available_boot_domains:
        raise TimingDomainError(
            "capture and telemetry boot clocks differ: "
            f"{target.boot_domain!r} vs {available_boot_domains!r}"
        )
    target_basis = "UTC" if target.utc_ns is not None else "boot"
    sample_bases = sorted(
        {
            "UTC" if sample.time.utc_ns is not None else "boot"
            for sample in samples
        }
    )
    raise TimingDomainError(
        f"cannot interpolate {target_basis} capture time against {sample_bases} telemetry"
    )


def _bracket(
    target: _TimeMark,
    samples: Sequence[_TimedSample],
    max_gap_ns: int,
) -> tuple[_TimedSample, _TimedSample, float, int, str, str]:
    if not samples:
        raise InterpolationError("required telemetry stream is absent")
    basis, domain, target_ns = _choose_interpolation_basis(target, samples)
    comparable = [
        (value, sample)
        for sample in samples
        if (value := _time_for_basis(sample.time, basis, domain)) is not None
    ]
    comparable.sort(key=lambda item: item[0])
    for value, sample in comparable:
        if value == target_ns:
            return sample, sample, 0.0, 0, basis, domain
    lower = next(
        (item for item in reversed(comparable) if item[0] < target_ns), None
    )
    upper = next((item for item in comparable if item[0] > target_ns), None)
    if lower is None or upper is None:
        raise InterpolationError(
            "telemetry does not bracket capture time; extrapolation is forbidden"
        )
    span = upper[0] - lower[0]
    if span > max_gap_ns:
        raise InterpolationError(
            f"telemetry bracket {span / 1e6:.3f} ms exceeds "
            f"{max_gap_ns / 1e6:.3f} ms limit"
        )
    fraction = (target_ns - lower[0]) / span
    return lower[1], upper[1], fraction, span, basis, domain


def _interpolate_longitude(first: float, second: float, fraction: float) -> float:
    delta = ((second - first + 180.0) % 360.0) - 180.0
    value = first + fraction * delta
    return ((value + 180.0) % 360.0) - 180.0


def _max_optional(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _bounded_pair_max(
    first: float | None,
    second: float | None,
    *,
    same_sample: bool,
) -> float | None:
    """Return a conservative bracket metric; one missing endpoint is unknown."""

    if same_sample:
        return first
    if first is None or second is None:
        return None
    return max(first, second)


def _fix_rank(value: int | str | None) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_")
        reverse = {label: rank for rank, label in _FIX_LABELS.items()}
        if normalized in {"fixed", "rtk-fixed"}:
            return 6
        if normalized in {"float", "rtk-float"}:
            return 5
        return reverse.get(normalized)
    return None


def _worst_fix_type(
    first: int | str | None,
    second: int | str | None,
) -> int | str | None:
    if first is None or second is None:
        return None
    first_rank = _fix_rank(first)
    second_rank = _fix_rank(second)
    if first_rank is None or second_rank is None:
        return first if first == second else None
    return min(first_rank, second_rank)


def _interpolation_motion_model_uncertainty(
    span_ns: int,
    fraction: float,
    acceleration_bound: float | None,
    policy_field: str,
) -> float:
    """Bound linear/geodesic interpolation remainder under bounded acceleration.

    For a twice-differentiable trajectory with second-derivative norm bounded
    by ``a``, the deviation from the endpoint chord at fraction ``f`` is at
    most ``0.5 * a * T**2 * f * (1-f)``.  The attitude bound is interpreted as
    geodesic angular acceleration on SO(3), expressed in degrees per second².
    """

    if span_ns == 0:
        return 0.0
    if acceleration_bound is None:
        raise InterpolationError(
            f"{policy_field} must be declared for non-exact interpolation"
        )
    duration_s = span_ns / 1_000_000_000.0
    return (
        0.5
        * acceleration_bound
        * duration_s
        * duration_s
        * fraction
        * (1.0 - fraction)
    )


def _interpolate_position(
    target: _TimeMark,
    samples: Sequence[_TimedSample],
    max_gap_ns: int,
    acceleration_bound_mps2: float | None,
) -> tuple[
    float,
    float,
    float,
    float | None,
    int,
    float,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    first, second, fraction, span, _, _ = _bracket(target, samples, max_gap_ns)
    values: list[tuple[float, float, float, float | None]] = []
    for sample in (first, second):
        lat = _coordinate(sample.message, "lat")
        lon = _coordinate(sample.message, "lon")
        alt = _altitude(sample.message, "alt")
        relative = _altitude(sample.message, "relative_alt")
        if lat is None or lon is None or alt is None:
            raise MessageFormatError("GLOBAL_POSITION_INT has an invalid position")
        values.append((lat, lon, alt, relative))
    lat = values[0][0] + fraction * (values[1][0] - values[0][0])
    lon = _interpolate_longitude(values[0][1], values[1][1], fraction)
    alt = values[0][2] + fraction * (values[1][2] - values[0][2])
    relative = None
    if values[0][3] is not None and values[1][3] is not None:
        relative = values[0][3] + fraction * (values[1][3] - values[0][3])
    model_uncertainty_m = _interpolation_motion_model_uncertainty(
        span,
        fraction,
        acceleration_bound_mps2,
        "position_acceleration_bound_mps2",
    )
    return (
        lat,
        lon,
        alt,
        relative,
        span,
        model_uncertainty_m,
        first.message,
        second.message,
    )


def _interpolate_attitude(
    target: _TimeMark,
    samples: Sequence[_TimedSample],
    max_gap_ns: int,
    acceleration_bound_deg_s2: float | None,
) -> tuple[
    tuple[float, float, float, float],
    int,
    float,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    first, second, fraction, span, _, _ = _bracket(target, samples, max_gap_ns)
    q0 = _attitude_quaternion(first.message)
    q1 = _attitude_quaternion(second.message)
    if q0 is None or q1 is None:
        raise MessageFormatError("ATTITUDE_QUATERNION has an invalid quaternion")
    model_uncertainty_deg = _interpolation_motion_model_uncertainty(
        span,
        fraction,
        acceleration_bound_deg_s2,
        "angular_acceleration_bound_deg_s2",
    )
    return (
        _slerp(q0, q1, fraction),
        span,
        model_uncertainty_deg,
        first.message,
        second.message,
    )


def _normalise_image_references(
    images: Sequence[str | Path | Mapping[str, Any] | ImageReference]
    | Mapping[int, str | Path]
    | None,
) -> list[ImageReference]:
    if images is None:
        return []
    if isinstance(images, Mapping):
        return [
            ImageReference(str(filename), index)
            for index, filename in images.items()
        ]
    result: list[ImageReference] = []
    for sequence_index, image in enumerate(images):
        if isinstance(image, ImageReference):
            result.append(image)
        elif isinstance(image, (str, Path)):
            result.append(ImageReference(str(image), sequence_index))
        elif isinstance(image, Mapping):
            filename = image.get("filename") or image.get("image_name") or image.get("path")
            if filename is None:
                raise MessageFormatError("image reference dictionary needs filename")
            index = image.get("image_index", image.get("index", sequence_index))
            result.append(
                ImageReference(
                    filename=str(filename),
                    image_index=index,
                    capture_monotonic_ns=(
                        image["capture_monotonic_ns"]
                        if image.get("capture_monotonic_ns") is not None
                        else None
                    ),
                    capture_utc_ns=(
                        image["capture_utc_ns"]
                        if image.get("capture_utc_ns") is not None
                        else None
                    ),
                    clock_domain=(
                        str(image["clock_domain"])
                        if image.get("clock_domain") is not None
                        else None
                    ),
                )
            )
        else:
            raise TypeError(f"unsupported image reference: {type(image)!r}")
    return result


def _event_image_index(event: Mapping[str, Any]) -> int | None:
    """Return an exact MAVLink image index/sequence identifier, if present."""

    if event.get("image_index") is not None:
        raw_index = event["image_index"]
        field = "image_index"
    elif event.get("seq") is not None:
        raw_index = event["seq"]
        field = "seq"
    else:
        return None
    return _exact_nonnegative_int(
        raw_index,
        field,
        error_type=MessageFormatError,
    )


def _url_filename(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().rstrip("\x00")
    if not text:
        return None
    parsed = urlparse(text)
    windows_drive = len(parsed.scheme) == 1 and len(text) >= 3 and text[1] == ":"
    path = parsed.path if parsed.scheme and not windows_drive else text
    path = path.replace("\\", "/")
    name = PurePosixPath(unquote(path)).name
    return name or None


def _match_image(
    event: Mapping[str, Any],
    event_order: int,
    references: Sequence[ImageReference],
    used: set[int],
    require_match: bool,
) -> tuple[ImageReference, str]:
    event_image_index = _event_image_index(event)
    image_index = event_image_index if event_image_index is not None else event_order
    event_name = _url_filename(event.get("file_url"))

    index_matches = {
        index
        for index, reference in enumerate(references)
        if event_image_index is not None and reference.image_index == image_index
    }
    name_matches = {
        index
        for index, reference in enumerate(references)
        if event_name
        and Path(reference.filename).name.casefold() == event_name.casefold()
    }
    if require_match and references:
        if event_image_index is not None and event_name:
            matches = index_matches & name_matches
            basis = "image_index+file_url"
            if index_matches and name_matches and not matches:
                raise MessageFormatError(
                    f"capture index {image_index} conflicts with file_url {event_name!r}"
                )
        elif event_image_index is not None:
            matches = index_matches
            basis = "image_index"
        elif event_name:
            matches = name_matches
            basis = "file_url"
        else:
            matches = set()
            basis = "none"
        available = sorted(matches - used)
        if len(available) != 1:
            raise MessageFormatError(
                f"capture index {image_index} has no unique image match"
            )
        selected = available[0]
        used.add(selected)
        return references[selected], basis

    candidates: list[tuple[int, ImageReference, str]] = []
    candidates.extend(
        (index, references[index], "image_index") for index in sorted(index_matches)
    )
    candidates.extend(
        (index, references[index], "file_url") for index in sorted(name_matches)
    )
    if event_order < len(references):
        candidates.append((event_order, references[event_order], "list_order_fallback"))
    for index, reference, basis in candidates:
        if index not in used:
            used.add(index)
            return reference, basis
    if require_match and references:
        raise MessageFormatError(
            f"capture index {image_index} has no unique image match"
        )
    generated = event_name or f"image_{image_index:06d}.jpg"
    return ImageReference(generated, image_index), (
        "event_file_url" if event_name else "generated_event_index"
    )


def _event_time(event: Mapping[str, Any], image: ImageReference) -> _TimeMark:
    try:
        return _time_mark(event)
    except TimingError:
        if image.capture_monotonic_ns is not None:
            return _TimeMark(
                image.capture_monotonic_ns,
                None,
                image.clock_domain,
                None,
            )
        if image.capture_utc_ns is not None:
            return _TimeMark(None, image.capture_utc_ns, None, None)
        raise


def _validate_image_time(
    event_time: _TimeMark, image: ImageReference, tolerance_ns: int
) -> None:
    if image.capture_monotonic_ns is not None:
        if event_time.monotonic_ns is None:
            raise TimingDomainError(
                "image has monotonic time but capture event has only UTC"
            )
        if image.clock_domain != event_time.boot_domain:
            raise TimingDomainError(
                f"image/event clock domains differ: {image.clock_domain!r} vs "
                f"{event_time.boot_domain!r}"
            )
        difference = abs(image.capture_monotonic_ns - event_time.monotonic_ns)
        if difference > tolerance_ns:
            raise TimingError("image and capture-event timestamps exceed matching tolerance")
    elif image.capture_utc_ns is not None:
        if event_time.utc_ns is None:
            raise TimingDomainError("image has UTC time but capture event has only boot time")
        if abs(image.capture_utc_ns - event_time.utc_ns) > tolerance_ns:
            raise TimingError("image and capture-event timestamps exceed matching tolerance")


def _shift_time(mark: _TimeMark, lag_ns: int, uncertainty_ns: int) -> _TimeMark:
    monotonic = None if mark.monotonic_ns is None else mark.monotonic_ns + lag_ns
    utc = None if mark.utc_ns is None else mark.utc_ns + lag_ns
    if monotonic is not None and monotonic < 0 or utc is not None and utc < 0:
        raise TimingError("shutter lag shifts capture before the clock origin")
    combined_uncertainty = (
        None
        if mark.uncertainty_ns is None
        else mark.uncertainty_ns + uncertainty_ns
    )
    return _TimeMark(
        monotonic,
        utc,
        mark.boot_domain,
        combined_uncertainty,
    )


def _quality_from_pair(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[int | str | None, str, str]:
    fix0, quality0, rtk0 = _quality_fields(first)
    fix1, quality1, rtk1 = _quality_fields(second)
    if first == second:
        return fix0, quality0, rtk0
    selected_fix = _worst_fix_type(fix0, fix1)
    if selected_fix is not None:
        quality, rtk = _fix_quality(selected_fix)
        return selected_fix, quality, rtk
    return None, "unknown", "unknown"


def _bracket_quality(
    target: _TimeMark,
    samples: Sequence[_TimedSample],
    max_gap_ns: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], int] | None:
    """Return a bounded quality bracket, never a one-sided nearest sample."""

    if not samples:
        return None
    try:
        first, second, _, span, _, _ = _bracket(target, samples, max_gap_ns)
    except (InterpolationError, TimingDomainError):
        # GPS quality is optional evidence. A one-sided or over-wide stream must
        # not upgrade the capture. An incomparable clock domain likewise cannot
        # contribute evidence, but need not invalidate an otherwise authoritative
        # direct capture record.
        return None
    if _pair_clock_uncertainty_ns(first.message, second.message) is None:
        # Optional enrichment is all-or-nothing. Consuming quality whose temporal
        # association is unknown would poison an otherwise authoritative record
        # (or let a configured downstream default appear to validate this sample).
        return None
    return first.message, second.message, span


def _pair_clock_uncertainty_ns(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> int | None:
    """Return the worst endpoint clock uncertainty, or unknown if either is."""

    first_uncertainty = _time_mark(first).uncertainty_ns
    second_uncertainty = _time_mark(second).uncertainty_ns
    if first_uncertainty is None or second_uncertainty is None:
        return None
    return max(first_uncertainty, second_uncertainty)


def _combined_interpolation_time_uncertainty_ns(
    capture_uncertainty_ns: int | None,
    endpoint_uncertainties_ns: Sequence[int | None],
) -> int | None:
    """Bound capture-to-telemetry alignment error across all used streams."""

    if capture_uncertainty_ns is None:
        return None
    if not endpoint_uncertainties_ns:
        return capture_uncertainty_ns
    if any(value is None for value in endpoint_uncertainties_ns):
        return None
    return capture_uncertainty_ns + max(
        int(value) for value in endpoint_uncertainties_ns if value is not None
    )


_INTERPOLATED_TELEMETRY_STREAMS = {
    "GLOBAL_POSITION_INT",
    "ATTITUDE_QUATERNION",
    "GPS_RAW_INT",
    "GPS2_RAW",
}


def _telemetry_samples_equivalent(
    kind: str,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    """Compare duplicate samples, accounting for quaternion sign equivalence."""

    try:
        if first == second:
            return True
    except (TypeError, ValueError):
        return False
    if kind != "ATTITUDE_QUATERNION":
        return False

    first_copy = dict(first)
    second_copy = dict(second)
    first_q = _attitude_quaternion(first_copy)
    second_q = _attitude_quaternion(second_copy)
    if first_q is None or second_q is None:
        return False
    dot = abs(sum(left * right for left, right in zip(first_q, second_q)))
    if not math.isclose(dot, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return False
    for copy in (first_copy, second_copy):
        for field in ("q1", "q2", "q3", "q4", "q", "quaternion_wxyz"):
            copy.pop(field, None)
    try:
        return first_copy == second_copy
    except (TypeError, ValueError):
        return False


def _validate_equal_timestamp_samples(
    typed: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    """Reject contradictory samples at one stream/source/domain timestamp."""

    seen: dict[
        tuple[str, str, str, int | None, int | None, int], Mapping[str, Any]
    ] = {}
    for kind, message in typed:
        if kind not in _INTERPOLATED_TELEMETRY_STREAMS:
            continue
        mark = _time_mark(message)
        system = _source_system_id(message)
        component = _source_component_id(message)
        time_keys: list[tuple[str, str, int]] = []
        if mark.monotonic_ns is not None:
            time_keys.append(("boot", mark.boot_domain or "", mark.monotonic_ns))
        if mark.utc_ns is not None:
            time_keys.append(("utc", "UTC", mark.utc_ns))
        for basis, domain, timestamp_ns in time_keys:
            key = (kind, basis, domain, system, component, timestamp_ns)
            previous = seen.get(key)
            if previous is not None and not _telemetry_samples_equivalent(
                kind, previous, message
            ):
                raise MessageFormatError(
                    f"conflicting {kind} samples at equal {basis} timestamp "
                    f"{timestamp_ns} in domain {domain!r} for system "
                    f"{system!r}, component {component!r}"
                )
            seen[key] = message


def _validate_implicit_boot_domains(
    typed: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    """Reject reboot/component ambiguity in undeclared boot-clock domains."""

    last_by_source: dict[tuple[int | None, int | None], int] = {}
    components_by_system: dict[int | None, set[int | None]] = {}
    for _, message in typed:
        if message.get("clock_domain") or message.get("boot_clock_domain"):
            continue
        try:
            mark = _time_mark(message)
        except TimingError:
            continue
        if mark.monotonic_ns is None:
            continue
        system = _source_system_id(message)
        component = _source_component_id(message)
        key = (system, component)
        previous = last_by_source.get(key)
        if previous is not None and mark.monotonic_ns < previous:
            raise TimingDomainError(
                "boot-clock reset/reordered epoch detected; assign a unique "
                "clock_domain to every reboot epoch"
            )
        last_by_source[key] = mark.monotonic_ns
        components_by_system.setdefault(system, set()).add(component)

    ambiguous = {
        system: components
        for system, components in components_by_system.items()
        if len(components) > 1
    }
    if ambiguous:
        raise TimingDomainError(
            "boot timestamps span multiple MAVLink components without a "
            "shared explicit clock_domain/TIMESYNC declaration"
        )


def match_capture_events(
    messages: Iterable[Mapping[str, Any]],
    images: Sequence[str | Path | Mapping[str, Any] | ImageReference]
    | Mapping[int, str | Path]
    | None = None,
    *,
    config: PixhawkBridgeConfig | None = None,
) -> tuple[CameraPoseRecord, ...]:
    """Match images to Pixhawk capture events and produce camera poses.

    Successful ``CAMERA_IMAGE_CAPTURED`` messages are preferred as they carry
    the camera pose at exposure.  If none exist, ``CAMERA_TRIGGER`` or a
    timestamped :class:`ImageReference` supplies capture times, while
    ``GLOBAL_POSITION_INT`` and ``ATTITUDE_QUATERNION`` are interpolated with
    no extrapolation and a bounded bracket.

    ``shutter_lag_ms`` is the calibrated trigger-to-exposure delay and is only
    applied to ``CAMERA_TRIGGER``.  ``CAMERA_IMAGE_CAPTURED`` and timestamped
    image references already represent exposure; they remain unshifted unless
    ``captured_event_time_correction_ms`` is explicitly calibrated.
    """

    policy = config or PixhawkBridgeConfig()
    normalized = [dict(message) for message in messages]
    typed = [(_message_type(message), message) for message in normalized]

    known_systems = {
        identifier
        for _, message in typed
        if (identifier := _source_system_id(message)) is not None
    }
    if policy.system_id is None and len(known_systems) > 1:
        raise MessageFormatError(
            "messages contain multiple MAVLink systems; select PixhawkBridgeConfig.system_id"
        )
    selected_system = (
        policy.system_id
        if policy.system_id is not None
        else (next(iter(known_systems)) if len(known_systems) == 1 else None)
    )
    if selected_system is not None:
        typed = [
            (kind, message)
            for kind, message in typed
            if _source_system_id(message) == selected_system
        ]
        if not typed:
            raise MessageFormatError(
                f"no MAVLink messages belong to selected system {selected_system}"
            )

    def event_camera_id(kind: str, message: Mapping[str, Any]) -> int | None:
        if message.get("camera_id") is not None:
            raw = message["camera_id"]
            field = "camera_id"
        elif message.get("camera_device_id") is not None:
            raw = message["camera_device_id"]
            field = "camera_device_id"
        else:
            raw = None
            field = "camera_id"
        if raw is not None:
            return _exact_nonnegative_int(
                raw,
                field,
                error_type=MessageFormatError,
            )
        if kind == "CAMERA_TRIGGER":
            return policy.unidentified_trigger_camera_id
        return None

    event_camera_ids = {
        identifier
        for kind, message in typed
        if kind in {"CAMERA_IMAGE_CAPTURED", "CAMERA_TRIGGER"}
        and (identifier := event_camera_id(kind, message)) is not None
    }
    if policy.camera_id is None and len(event_camera_ids) > 1:
        raise MessageFormatError(
            "capture events contain multiple cameras; select PixhawkBridgeConfig.camera_id"
        )
    selected_camera = (
        policy.camera_id
        if policy.camera_id is not None
        else (
            next(iter(event_camera_ids)) if len(event_camera_ids) == 1 else None
        )
    )
    if selected_camera is not None:
        typed = [
            (kind, message)
            for kind, message in typed
            if kind not in {"CAMERA_IMAGE_CAPTURED", "CAMERA_TRIGGER"}
            or event_camera_id(kind, message) == selected_camera
        ]

    capture_kinds = {"CAMERA_IMAGE_CAPTURED", "CAMERA_TRIGGER"}
    event_component_ids = {
        identifier
        for kind, message in typed
        if kind in capture_kinds
        and (identifier := _source_component_id(message)) is not None
    }
    if policy.capture_component_id is None and len(event_component_ids) > 1:
        raise MessageFormatError(
            "capture events contain multiple MAVLink components; select "
            "PixhawkBridgeConfig.capture_component_id"
        )
    selected_component = (
        policy.capture_component_id
        if policy.capture_component_id is not None
        else (
            next(iter(event_component_ids))
            if len(event_component_ids) == 1
            else None
        )
    )
    if selected_component is not None:
        typed = [
            (kind, message)
            for kind, message in typed
            if kind not in capture_kinds
            or _source_component_id(message) == selected_component
        ]
        if not any(kind in capture_kinds for kind, _ in typed):
            raise MessageFormatError(
                "no capture events belong to selected component "
                f"{selected_component}"
            )

    pose_stream_kinds = {"GLOBAL_POSITION_INT", "ATTITUDE_QUATERNION"}
    pose_stream_sources = {
        (_source_system_id(message), _source_component_id(message))
        for kind, message in typed
        if kind in pose_stream_kinds
    }
    if policy.navigation_component_id is None and len(pose_stream_sources) > 1:
        raise MessageFormatError(
            "position/attitude telemetry contains multiple MAVLink sources; "
            "select PixhawkBridgeConfig.navigation_component_id"
        )
    if policy.navigation_component_id is not None:
        had_pose_stream = any(kind in pose_stream_kinds for kind, _ in typed)
        typed = [
            (kind, message)
            for kind, message in typed
            if kind not in pose_stream_kinds
            or _source_component_id(message) == policy.navigation_component_id
        ]
        if had_pose_stream and not any(
            kind in pose_stream_kinds for kind, _ in typed
        ):
            raise MessageFormatError(
                "no position/attitude telemetry belongs to selected navigation "
                f"component {policy.navigation_component_id}"
            )

    gps_stream_kinds = {"GPS_RAW_INT", "GPS2_RAW"}
    if policy.gps_message_type is not None:
        typed = [
            (kind, message)
            for kind, message in typed
            if kind not in gps_stream_kinds or kind == policy.gps_message_type
        ]
    gps_stream_sources = {
        (kind, _source_system_id(message), _source_component_id(message))
        for kind, message in typed
        if kind in gps_stream_kinds
    }
    if len(gps_stream_sources) > 1:
        raise MessageFormatError(
            "GPS quality telemetry contains multiple message types or MAVLink "
            "sources; select PixhawkBridgeConfig.gps_message_type and prefilter "
            "to one source component"
        )

    _validate_equal_timestamp_samples(typed)
    _validate_implicit_boot_domains(typed)
    for kind, message in typed:
        if kind in capture_kinds:
            _event_image_index(message)

    captured: list[Mapping[str, Any]] = []
    for kind, message in typed:
        if kind != "CAMERA_IMAGE_CAPTURED":
            continue
        result = message.get("capture_result", 1)
        if result not in (0, 1, False, True):
            raise MessageFormatError("CAMERA_IMAGE_CAPTURED.capture_result is invalid")
        if bool(result):
            captured.append(message)
    captured_indexes = {
        image_index
        for message in captured
        if (image_index := _event_image_index(message)) is not None
    }
    captured_time_keys: set[tuple[str, str, int]] = set()
    for message in captured:
        mark = _time_mark(message)
        if mark.monotonic_ns is not None:
            captured_time_keys.add(
                ("boot", mark.boot_domain or "", mark.monotonic_ns)
            )
        if mark.utc_ns is not None:
            captured_time_keys.add(("utc", "UTC", mark.utc_ns))
    # Prefer CAMERA_IMAGE_CAPTURED only for the capture it supersedes.  A
    # different trigger in the same log remains a valid exposure candidate.
    # This prevents one acknowledged photo from erasing every trigger-only
    # photo in a partially acknowledged mission.
    events = []
    for kind, message in typed:
        if kind == "CAMERA_IMAGE_CAPTURED" and message in captured:
            events.append(message)
        elif kind == "CAMERA_TRIGGER":
            trigger_index = _event_image_index(message)
            mark = _time_mark(message)
            time_keys = set()
            if mark.monotonic_ns is not None:
                time_keys.add(("boot", mark.boot_domain or "", mark.monotonic_ns))
            if mark.utc_ns is not None:
                time_keys.add(("utc", "UTC", mark.utc_ns))
            same_index = (
                trigger_index is not None and trigger_index in captured_indexes
            )
            same_time = bool(time_keys & captured_time_keys)
            if not same_index and not same_time:
                events.append(message)

    references = _normalise_image_references(images)
    if not events:
        timestamped = [
            reference
            for reference in references
            if reference.capture_monotonic_ns is not None
            or reference.capture_utc_ns is not None
        ]
        if not timestamped:
            raise MessageFormatError(
                "no successful CAMERA_IMAGE_CAPTURED, CAMERA_TRIGGER, or "
                "timestamped image references"
            )
        events = [
            {
                "mavpackettype": "IMAGE_REFERENCE",
                "image_index": reference.image_index,
            }
            for reference in timestamped
        ]
        references = timestamped

    positions = [
        _TimedSample(_time_mark(message), message)
        for kind, message in typed
        if kind == "GLOBAL_POSITION_INT"
    ]
    attitudes = [
        _TimedSample(_time_mark(message), message)
        for kind, message in typed
        if kind == "ATTITUDE_QUATERNION"
    ]
    gps_quality = [
        _TimedSample(_time_mark(message), message)
        for kind, message in typed
        if kind in {"GPS_RAW_INT", "GPS2_RAW"}
    ]

    trigger_lag_ns = int(round(policy.shutter_lag_ms * 1_000_000.0))
    trigger_lag_uncertainty_ns = int(
        round(policy.shutter_lag_uncertainty_ms * 1_000_000.0)
    )
    captured_correction_ns = int(
        round(policy.captured_event_time_correction_ms * 1_000_000.0)
    )
    captured_correction_uncertainty_ns = int(
        round(policy.captured_event_time_uncertainty_ms * 1_000_000.0)
    )
    max_gap_ns = int(round(policy.max_interpolation_gap_ms * 1_000_000.0))
    mount = _normalize_quaternion(policy.camera_from_body_quaternion_wxyz)
    mount_identity = all(
        abs(actual - expected) < 1e-12
        for actual, expected in zip(mount, (1.0, 0.0, 0.0, 0.0))
    )

    used_images: set[int] = set()
    result: list[CameraPoseRecord] = []
    for order, event in enumerate(events):
        image, image_match_basis = _match_image(
            event,
            order,
            references,
            used_images,
            policy.require_image_match,
        )
        raw_time = _event_time(event, image)
        _validate_image_time(raw_time, image, max_gap_ns)
        kind = _message_type(event)
        if kind == "CAMERA_TRIGGER":
            time_correction_ns = trigger_lag_ns
            correction_uncertainty_ns = trigger_lag_uncertainty_ns
        else:
            time_correction_ns = captured_correction_ns
            correction_uncertainty_ns = captured_correction_uncertainty_ns
        target_time = _shift_time(
            raw_time, time_correction_ns, correction_uncertainty_ns
        )

        direct_lat = _coordinate(event, "lat") if "lat" in event else None
        direct_lon = _coordinate(event, "lon") if "lon" in event else None
        direct_alt = _altitude(event, "alt") if "alt" in event else None
        direct_relative = (
            _altitude(event, "relative_alt") if "relative_alt" in event else None
        )
        direct_q = _attitude_quaternion(event)
        # A correction to CAMERA_IMAGE_CAPTURED calibrates its clock label; it
        # does not invalidate the authoritative camera-center/gimbal pose that
        # accompanies that exposure. Trigger/image-reference corrections, by
        # contrast, require evaluating vehicle telemetry at the shifted time.
        must_interpolate = (
            time_correction_ns != 0 and kind != "CAMERA_IMAGE_CAPTURED"
        )

        interpolation_spans: list[int] = []
        interpolation_endpoint_uncertainties_ns: list[int | None] = []
        position_model_uncertainty_m = 0.0
        position_pair: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
        if (
            not must_interpolate
            and direct_lat is not None
            and direct_lon is not None
            and direct_alt is not None
        ):
            lat, lon, altitude, relative = (
                direct_lat,
                direct_lon,
                direct_alt,
                direct_relative,
            )
            position_source = kind
        else:
            (
                lat,
                lon,
                altitude,
                relative,
                span,
                position_model_uncertainty_m,
                first_position,
                second_position,
            ) = _interpolate_position(
                target_time,
                positions,
                max_gap_ns,
                policy.position_acceleration_bound_mps2,
            )
            interpolation_spans.append(span)
            position_pair = (first_position, second_position)
            interpolation_endpoint_uncertainties_ns.append(
                _pair_clock_uncertainty_ns(*position_pair)
            )
            position_source = "GLOBAL_POSITION_INT:bounded_linear_interpolation"

        relative_altitude_reference = (
            "not_available"
            if relative is None
            else "ground"
            if position_source == "CAMERA_IMAGE_CAPTURED"
            else "home"
            if "GLOBAL_POSITION_INT" in position_source
            else "unspecified"
        )
        if position_source == "CAMERA_IMAGE_CAPTURED":
            position_reference = str(
                event.get(
                    "camera_position_reference",
                    policy.camera_image_position_reference,
                )
            ).strip()
            if position_reference == "unspecified":
                raise MessageFormatError(
                    "CAMERA_IMAGE_CAPTURED position reference is unspecified; "
                    "declare camera_optical_center or vehicle_navigation_origin"
                )
        else:
            position_reference = "vehicle_navigation_origin"

        attitude_model_uncertainty_deg = 0.0
        attitude_pair: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
        if not must_interpolate and direct_q is not None:
            if kind == "CAMERA_IMAGE_CAPTURED":
                input_attitude_profile = str(
                    event.get(
                        "camera_attitude_profile",
                        policy.camera_image_attitude_profile or "",
                    )
                ).strip()
                if input_attitude_profile != DECLARED_CAMERA_IMAGE_ATTITUDE_PROFILE:
                    raise MessageFormatError(
                        "CAMERA_IMAGE_CAPTURED.q has no standardized frame/direction; "
                        "declare the validated camera attitude profile"
                    )
            else:
                input_attitude_profile = (
                    "declared_direct_non_camera_capture_orientation"
                )
            q_ned_to_camera_frd = _normalize_quaternion(direct_q)
            attitude_source = kind
        else:
            (
                q_ned_to_body_frd,
                span,
                attitude_model_uncertainty_deg,
                first_attitude,
                second_attitude,
            ) = _interpolate_attitude(
                target_time,
                attitudes,
                max_gap_ns,
                policy.angular_acceleration_bound_deg_s2,
            )
            interpolation_spans.append(span)
            attitude_pair = (first_attitude, second_attitude)
            interpolation_endpoint_uncertainties_ns.append(
                _pair_clock_uncertainty_ns(*attitude_pair)
            )
            q_ned_to_camera_frd = quaternion_multiply(
                mount, q_ned_to_body_frd
            )
            mount_note = "identity_mount_assumption" if mount_identity else "calibrated_mount"
            attitude_source = (
                "ATTITUDE_QUATERNION:bounded_slerp+" + mount_note
            )
            input_attitude_profile = "mavlink_attitude_quaternion_ned_to_body_frd"

        horizontal_accuracy = _accuracy(event)
        vertical_accuracy = _vertical_accuracy(event)
        attitude_accuracy = _accuracy(event, attitude=True)
        fix_type, fix_quality, rtk_status = _quality_fields(event)
        direct_capture_position_authoritative = (
            kind == "CAMERA_IMAGE_CAPTURED" and position_pair is None
        )
        direct_capture_fix_authoritative = (
            direct_capture_position_authoritative and fix_type is not None
        )
        direct_capture_horizontal_authoritative = (
            direct_capture_position_authoritative
            and horizontal_accuracy is not None
        )
        direct_capture_vertical_authoritative = (
            direct_capture_position_authoritative
            and vertical_accuracy is not None
        )

        if position_pair is not None:
            pair_horizontal_accuracy = _bounded_pair_max(
                _accuracy(position_pair[0]),
                _accuracy(position_pair[1]),
                same_sample=position_pair[0] == position_pair[1],
            )
            pair_vertical_accuracy = _bounded_pair_max(
                _vertical_accuracy(position_pair[0]),
                _vertical_accuracy(position_pair[1]),
                same_sample=position_pair[0] == position_pair[1],
            )
            if not direct_capture_horizontal_authoritative:
                horizontal_accuracy = (
                    None
                    if pair_horizontal_accuracy is None
                    else _max_optional(horizontal_accuracy, pair_horizontal_accuracy)
                )
            if not direct_capture_vertical_authoritative:
                vertical_accuracy = (
                    None
                    if pair_vertical_accuracy is None
                    else _max_optional(vertical_accuracy, pair_vertical_accuracy)
                )
            pair_fix, _, _ = _quality_from_pair(*position_pair)
            if not direct_capture_fix_authoritative:
                fix_type = (
                    _worst_fix_type(fix_type, pair_fix)
                    if fix_type is not None
                    else pair_fix
                )
            fix_quality, rtk_status = _fix_quality(fix_type)
        if attitude_pair is not None:
            pair_attitude_accuracy = _bounded_pair_max(
                _accuracy(attitude_pair[0], attitude=True),
                _accuracy(attitude_pair[1], attitude=True),
                same_sample=attitude_pair[0] == attitude_pair[1],
            )
            attitude_accuracy = (
                None
                if pair_attitude_accuracy is None
                else _max_optional(attitude_accuracy, pair_attitude_accuracy)
            )

        needs_external_quality = not (
            direct_capture_fix_authoritative
            and direct_capture_horizontal_authoritative
            and direct_capture_vertical_authoritative
        )
        quality_pair = (
            _bracket_quality(target_time, gps_quality, max_gap_ns)
            if needs_external_quality
            else None
        )
        if quality_pair is not None:
            first_quality, second_quality, quality_span = quality_pair
            same_quality_sample = first_quality == second_quality
            gps_horizontal_accuracy = _bounded_pair_max(
                _gps_raw_accuracy(first_quality, "h_acc"),
                _gps_raw_accuracy(second_quality, "h_acc"),
                same_sample=same_quality_sample,
            )
            gps_vertical_accuracy = _bounded_pair_max(
                _gps_raw_accuracy(first_quality, "v_acc"),
                _gps_raw_accuracy(second_quality, "v_acc"),
                same_sample=same_quality_sample,
            )
            quality_influenced_record = False
            if (
                not direct_capture_horizontal_authoritative
                and (position_pair is None or horizontal_accuracy is not None)
            ):
                horizontal_accuracy = _max_optional(
                    horizontal_accuracy, gps_horizontal_accuracy
                )
                quality_influenced_record |= gps_horizontal_accuracy is not None
            if (
                not direct_capture_vertical_authoritative
                and (position_pair is None or vertical_accuracy is not None)
            ):
                vertical_accuracy = _max_optional(
                    vertical_accuracy, gps_vertical_accuracy
                )
                quality_influenced_record |= gps_vertical_accuracy is not None
            gps_fix, _, _ = _quality_from_pair(first_quality, second_quality)
            if not direct_capture_fix_authoritative:
                fix_type = (
                    gps_fix
                    if fix_type is None
                    else _worst_fix_type(fix_type, gps_fix)
                )
                fix_quality, rtk_status = _fix_quality(fix_type)
                quality_influenced_record = True
            if quality_influenced_record:
                interpolation_spans.append(quality_span)
                interpolation_endpoint_uncertainties_ns.append(
                    _pair_clock_uncertainty_ns(first_quality, second_quality)
                )

        # Endpoint measurement accuracy and the deterministic interpolation
        # remainder bound are different error sources. Add the hard model bound
        # after all endpoint-quality evidence has been conservatively merged.
        # An unknown measurement accuracy remains unknown rather than being
        # replaced by the finite model contribution alone.
        if position_pair is not None:
            if horizontal_accuracy is not None:
                horizontal_accuracy += position_model_uncertainty_m
            if vertical_accuracy is not None:
                vertical_accuracy += position_model_uncertainty_m
        if attitude_pair is not None and attitude_accuracy is not None:
            attitude_accuracy += attitude_model_uncertainty_deg

        combined_time_uncertainty_ns = (
            _combined_interpolation_time_uncertainty_ns(
                target_time.uncertainty_ns,
                interpolation_endpoint_uncertainties_ns,
            )
        )

        converted_q = mavlink_ned_frd_to_openprism_enu_flu(
            q_ned_to_camera_frd
        )
        converted_optical_q = mavlink_ned_frd_to_openprism_enu_optical(
            q_ned_to_camera_frd
        )
        yaw, pitch, roll = _aeronautical_euler_deg(q_ned_to_camera_frd)
        event_image_index = _event_image_index(event)
        image_index = (
            event_image_index
            if event_image_index is not None
            else image.image_index
        )
        result.append(
            CameraPoseRecord(
                image_name=image.filename,
                image_index=image_index,
                latitude_deg=lat,
                longitude_deg=lon,
                altitude_msl_m=altitude,
                relative_altitude_m=relative,
                quaternion_camera_flu_to_enu_wxyz=converted_q,
                quaternion_camera_optical_to_enu_wxyz=converted_optical_q,
                yaw_deg=yaw,
                pitch_deg=pitch,
                roll_deg=roll,
                capture_monotonic_ns=target_time.monotonic_ns,
                capture_utc_ns=target_time.utc_ns,
                event_monotonic_ns=raw_time.monotonic_ns,
                event_utc_ns=raw_time.utc_ns,
                clock_domain=target_time.boot_domain or "UTC",
                time_basis=target_time.time_basis,
                time_uncertainty_ns=combined_time_uncertainty_ns,
                horizontal_accuracy_m=horizontal_accuracy,
                vertical_accuracy_m=vertical_accuracy,
                attitude_accuracy_deg=attitude_accuracy,
                fix_type=fix_type,
                fix_quality=fix_quality,
                rtk_status=rtk_status,
                source_message=kind,
                position_source=position_source,
                attitude_source=attitude_source,
                interpolation_span_ns=(
                    max(interpolation_spans) if interpolation_spans else None
                ),
                relative_altitude_reference=relative_altitude_reference,
                position_reference=position_reference,
                input_attitude_profile=input_attitude_profile,
                image_match_basis=image_match_basis,
                system_id=_source_system_id(event),
                component_id=_source_component_id(event),
                camera_id=event_camera_id(kind, event),
            )
        )
    return tuple(result)


class PixhawkBridge:
    """Small state-free facade suitable for the OpenPRISM mapping pipeline."""

    def __init__(self, config: PixhawkBridgeConfig | None = None) -> None:
        self.config = config or PixhawkBridgeConfig()

    def parse(
        self,
        messages: Iterable[Mapping[str, Any]],
        images: Sequence[str | Path | Mapping[str, Any] | ImageReference]
        | Mapping[int, str | Path]
        | None = None,
    ) -> tuple[CameraPoseRecord, ...]:
        return match_capture_events(messages, images, config=self.config)


def export_odm_geo_txt(
    records: Iterable[CameraPoseRecord],
    *,
    projection: str = "EPSG:4326",
    require_accuracy: bool = False,
) -> str:
    """Serialize camera poses in OpenDroneMap ``geo.txt`` format.

    The records are WGS84 geodetic, therefore this serializer accepts only an
    EPSG:4326 header.  Projected headers require an explicit geodetic transform
    upstream; silently labelling lon/lat as UTM would corrupt the map.

    ODM's angle convention is not MAVLink's aeronautical convention (ODM uses
    pitch 0 for nadir), and ``geo.txt`` has no null token before its accuracy
    columns.  This exporter therefore emits position-only rows.  It never
    writes ``0 0 0`` as an "unknown" orientation because ODM consumes those as
    real yaw/pitch/roll values and may propagate them into DLS processing.
    Full quaternions and accuracy remain in :meth:`CameraPoseRecord.as_dict`
    until a separately validated ODM orientation converter is supplied.

    ``require_accuracy`` validates that records carry accuracy; it does not
    serialize it into the positional format because doing so would require
    inventing the three preceding orientation values.
    """

    normalized_projection = projection.upper().replace(" ", "")
    if normalized_projection not in {"EPSG:4326", "EPSG4326"}:
        raise ValueError(
            "CameraPoseRecord stores WGS84 lon/lat; transform coordinates "
            "before using a projected ODM header"
        )
    lines = ["EPSG:4326"]
    for record in records:
        if any(character.isspace() for character in record.image_name):
            raise ValueError("ODM geo.txt image names may not contain whitespace")
        if require_accuracy and (
            record.horizontal_accuracy_m is None
            or record.vertical_accuracy_m is None
        ):
            raise ValueError(
                f"{record.image_name} lacks horizontal/vertical accuracy"
            )
        fields = [
            record.image_name,
            f"{record.longitude_deg:.10f}",
            f"{record.latitude_deg:.10f}",
            f"{record.altitude_msl_m:.3f}",
        ]
        lines.append(" ".join(fields))
    return "\n".join(lines) + "\n"


def write_odm_geo_txt(
    path: str | Path,
    records: Iterable[CameraPoseRecord],
    *,
    projection: str = "EPSG:4326",
    require_accuracy: bool = False,
) -> Path:
    """Write an OpenDroneMap geolocation file and return its resolved path."""

    destination = Path(path)
    destination.write_text(
        export_odm_geo_txt(
            records,
            projection=projection,
            require_accuracy=require_accuracy,
        ),
        encoding="utf-8",
    )
    return destination.resolve()


def iter_pymavlink_messages(
    source: str | Path | Any,
    *,
    baud: int = 115_200,
    message_types: Sequence[str] | None = None,
    timeout_s: float = 1.0,
    max_messages: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield normalized dictionaries from a pymavlink connection or path.

    ``source`` may be an existing object with ``recv_match`` (live or replay)
    or any path/connection string accepted by ``mavutil.mavlink_connection``.
    A receive timeout ends the generator, so this helper never creates an
    uninterruptible live loop.
    """

    try:
        from pymavlink import mavutil  # type: ignore[import-not-found]
    except ImportError as error:
        raise OptionalDependencyError(
            "pymavlink is optional; install it with `python -m pip install pymavlink` "
            "to read Pixhawk logs or live MAVLink connections"
        ) from error

    if timeout_s < 0.0:
        raise ValueError("timeout_s must be non-negative")
    if max_messages is not None and max_messages < 0:
        raise ValueError("max_messages must be non-negative")
    connection = (
        source
        if hasattr(source, "recv_match")
        else mavutil.mavlink_connection(str(source), baud=baud)
    )
    wanted = list(message_types) if message_types else None
    emitted = 0
    while max_messages is None or emitted < max_messages:
        message = connection.recv_match(
            type=wanted,
            blocking=True,
            timeout=timeout_s,
        )
        if message is None:
            break
        if getattr(message, "get_type", lambda: "")() == "BAD_DATA":
            continue
        payload = dict(message.to_dict())
        if "mavpackettype" not in payload:
            payload["mavpackettype"] = message.get_type()
        get_system = getattr(message, "get_srcSystem", None)
        get_component = getattr(message, "get_srcComponent", None)
        if callable(get_system):
            payload["_srcSystem"] = get_system()
        if callable(get_component):
            payload["_srcComponent"] = get_component()
        emitted += 1
        yield payload


# Explicit alias for callers that prefer the operation over the matching term.
parse_pixhawk_captures = match_capture_events


__all__ = [
    "CameraPoseRecord",
    "GEODETIC_POSITION_FRAME",
    "ImageReference",
    "InterpolationError",
    "MAVLINK_ATTITUDE_CONVENTION",
    "MessageFormatError",
    "OPENPRISM_ATTITUDE_CONVENTION",
    "OPENPRISM_OPTICAL_ATTITUDE_CONVENTION",
    "OptionalDependencyError",
    "PixhawkBridge",
    "PixhawkBridgeConfig",
    "PixhawkBridgeError",
    "TimingDomainError",
    "TimingError",
    "export_odm_geo_txt",
    "iter_pymavlink_messages",
    "match_capture_events",
    "mavlink_ned_frd_to_openprism_enu_flu",
    "mavlink_ned_frd_to_openprism_enu_optical",
    "parse_pixhawk_captures",
    "quaternion_conjugate",
    "quaternion_multiply",
    "write_odm_geo_txt",
]
