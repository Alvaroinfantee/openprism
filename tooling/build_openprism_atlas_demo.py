"""Build the deterministic OpenPRISM Atlas synthetic mission demonstration.

This program intentionally generates every pixel and pose from analytic
functions.  It does not read Caltech imagery, a Pixhawk log, or any other real
sensor data.  The resulting bundle is useful for exercising the complete
``CameraPoseRecord -> AtlasMissionMapper -> operator/ML layers`` path without
inventing real-world GPS provenance.

The demo models a nadir RGB/thermal camera flying a serpentine survey.  Each
frame contains an analytically rendered terrain field plus one moving hot
object.  The moving object is supplied to the mapper as ``dynamic_mask`` and
therefore remains visible in the saved source evidence while being excluded
from the static terrain atlas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


# Make direct execution (``python tooling/build_openprism_atlas_demo.py``)
# independent of the caller's current directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openprism.atlas import (
    AtlasBundlePaths,
    AtlasMissionConfig,
    AtlasMissionMapper,
    ImageObjectObservation,
    RegisteredThermalFrameContract,
    SemanticFrameContract,
)
from openprism.mapping import (
    CameraIntrinsics,
    GeodeticCoordinate,
    LocalENUFrame,
    Quaternion,
    enu_to_geodetic,
)
from openprism.pixhawk import CameraPoseRecord


SYNTHETIC_WARNING = "SYNTHETIC DEMONSTRATION - NOT REAL SENSOR OR GPS DATA"
GENERATOR_ID = "openprism.analytic-atlas-demo/1.0"
DYNAMIC_CLASS_ID = 90
DEFAULT_OUTPUT = REPOSITORY_ROOT / "output" / "openprism_atlas" / "latest"

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 72
FOCAL_LENGTH_PX = 90.0
FLIGHT_AGL_M = 36.0
GROUND_ELEVATION_MSL_M = 100.0
GEOID_SEPARATION_M = -32.0
ORIGIN_LATITUDE_DEG = 34.137700
ORIGIN_LONGITUDE_DEG = -118.125300
CAPTURE_START_UTC_NS = 1_735_689_600_000_000_000


@dataclass(frozen=True, slots=True)
class SyntheticFrame:
    """One generated frame and its exact synthetic ground coordinates."""

    record: CameraPoseRecord
    rgb: np.ndarray
    thermal_normalized: np.ndarray
    semantic_class: np.ndarray
    dynamic_mask: np.ndarray
    support: np.ndarray
    east_enu_m: np.ndarray
    north_enu_m: np.ndarray


@dataclass(frozen=True, slots=True)
class DemoBuildResult:
    """Paths and integration counts returned by :func:`build_demo`."""

    atlas: AtlasBundlePaths
    manifest: Path
    notice: Path
    source_frames: Path
    accepted_capture_count: int
    rejected_capture_count: int
    dynamic_pixel_count: int


def demo_intrinsics() -> CameraIntrinsics:
    """Return the explicit, synthetic rectified-pinhole calibration."""

    return CameraIntrinsics(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        fx_px=FOCAL_LENGTH_PX,
        fy_px=FOCAL_LENGTH_PX,
        cx_px=(IMAGE_WIDTH - 1) / 2.0,
        cy_px=(IMAGE_HEIGHT - 1) / 2.0,
        calibration_id="SYNTHETIC:analytic-pinhole-v1",
        calibration_rms_px=0.05,
    )


def demo_mission_config() -> AtlasMissionConfig:
    """Return the deterministic local-map contract used by the demo."""

    return AtlasMissionConfig(
        mission_id="SYNTHETIC-openprism-atlas-demo-v1",
        origin_latitude_deg=ORIGIN_LATITUDE_DEG,
        origin_longitude_deg=ORIGIN_LONGITUDE_DEG,
        ground_elevation_msl_m=GROUND_ELEVATION_MSL_M,
        geoid_separation_m=GEOID_SEPARATION_M,
        east_min_m=-60.0,
        east_max_m=60.0,
        north_min_m=-48.0,
        north_max_m=48.0,
        resolution_m=0.75,
        horizontal_crs_id="EPSG:4326",
        input_vertical_datum_id="SYNTHETIC:local-mean-sea-level-v1",
        geoid_model_id="SYNTHETIC:constant-separation-v1",
        geoid_separation_uncertainty_m=0.0,
        tai_minus_utc_s=37,
        max_time_uncertainty_ns=1_000_000,
        max_projected_uncertainty_m=1.0,
        min_downward_cosine=0.2,
        max_slant_range_m=100.0,
        minimum_fix_type=3,
        require_rtk_fixed=False,
        minimum_agl_m=5.0,
        maximum_agl_m=100.0,
        timing_velocity_bound_mps=12.0,
        timing_angular_rate_bound_deg_s=45.0,
        real_sensor_data=False,
        real_navigation_data=False,
    )


def flight_plan_enu() -> tuple[tuple[float, float], ...]:
    """Return a 16-capture serpentine plan with deliberate image overlap."""

    east_positions = (-36.0, -12.0, 12.0, 36.0)
    north_positions = (30.0, 10.0, -10.0, -30.0)
    plan: list[tuple[float, float]] = []
    for row, north in enumerate(north_positions):
        ordered_east = east_positions if row % 2 == 0 else tuple(reversed(east_positions))
        plan.extend((east, north) for east in ordered_east)
    return tuple(plan)


def analytic_terrain_layers(
    east_enu_m: np.ndarray,
    north_enu_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate deterministic RGB, normalized thermal, and land-cover fields.

    Semantic classes are: 0 vegetation, 1 road, 2 water, 3 structure, and
    4 exposed soil.  ``DYNAMIC_CLASS_ID`` is deliberately reserved for the
    moving-object overlay and never originates in this static field.
    """

    east = np.asarray(east_enu_m, dtype=np.float64)
    north = np.asarray(north_enu_m, dtype=np.float64)
    if east.shape != north.shape:
        raise ValueError("east and north terrain coordinates must have matching shapes")

    fine_texture = np.sin(0.31 * east) * np.cos(0.27 * north)
    broad_texture = np.sin(0.075 * (east + 1.7 * north))
    rgb = np.stack(
        (
            0.20 + 0.035 * fine_texture + 0.025 * broad_texture,
            0.46 + 0.075 * fine_texture + 0.035 * broad_texture,
            0.20 + 0.030 * fine_texture,
        ),
        axis=-1,
    )
    thermal = 0.45 + 0.055 * np.sin(0.08 * east - 0.06 * north)
    semantic = np.zeros(east.shape, dtype=np.uint8)

    soil = ((east + 31.0) / 15.0) ** 2 + ((north + 23.0) / 10.0) ** 2 < 1.0
    semantic[soil] = 4
    rgb[soil] = np.stack(
        (
            0.48 + 0.035 * fine_texture[soil],
            0.34 + 0.020 * fine_texture[soil],
            0.19 + 0.015 * fine_texture[soil],
        ),
        axis=-1,
    )
    thermal[soil] = 0.66 + 0.035 * broad_texture[soil]

    water = ((east + 23.0) / 12.0) ** 2 + ((north - 19.0) / 8.0) ** 2 < 1.0
    semantic[water] = 2
    ripple = 0.025 * np.sin(0.8 * east[water] + 0.4 * north[water])
    rgb[water] = np.stack(
        (
            0.08 + ripple,
            0.26 + ripple,
            0.48 + 1.5 * ripple,
        ),
        axis=-1,
    )
    thermal[water] = 0.24 + 0.018 * np.sin(0.19 * east[water])

    road_center_north = 0.16 * east + 1.5
    road = np.abs(north - road_center_north) <= 2.6
    semantic[road] = 1
    lane_texture = 0.018 * np.sin(1.1 * east[road])
    rgb[road] = np.stack(
        (
            0.39 + lane_texture,
            0.40 + lane_texture,
            0.41 + lane_texture,
        ),
        axis=-1,
    )
    thermal[road] = 0.62 + 0.025 * np.cos(0.09 * east[road])

    structure = (east >= 17.0) & (east <= 31.0) & (north >= -22.0) & (north <= -7.0)
    semantic[structure] = 3
    roof_pattern = 0.04 * (np.sin(0.9 * east[structure]) > 0.0)
    rgb[structure] = np.stack(
        (
            0.56 + roof_pattern,
            0.23 + 0.5 * roof_pattern,
            0.16 + 0.4 * roof_pattern,
        ),
        axis=-1,
    )
    thermal[structure] = 0.76 + 0.035 * np.cos(0.24 * north[structure])

    rgb_u8 = np.clip(np.rint(rgb * 255.0), 0.0, 255.0).astype(np.uint8)
    return rgb_u8, np.clip(thermal, 0.0, 1.0).astype(np.float32), semantic


def _camera_quaternions() -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    """Return camera-FLU and OpenCV-optical rotations for a north-up nadir view."""

    # Optical +x points east, optical +y points south, and the optical
    # boresight +z points down.  This makes image top equal map north.
    enu_from_optical = np.diag((1.0, -1.0, -1.0))
    # Optical axes expressed in camera FLU: right=-left, down=-up,
    # forward=forward.
    flu_from_optical = np.array(
        ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        dtype=np.float64,
    )
    enu_from_flu = enu_from_optical @ flu_from_optical.T
    flu = Quaternion.from_rotation_matrix(enu_from_flu)
    optical = Quaternion.from_rotation_matrix(enu_from_optical)
    return (
        (flu.w, flu.x, flu.y, flu.z),
        (optical.w, optical.x, optical.y, optical.z),
    )


def render_synthetic_frame(
    index: int,
    camera_east_m: float,
    camera_north_m: float,
    *,
    intrinsics: CameraIntrinsics | None = None,
    mission_config: AtlasMissionConfig | None = None,
) -> SyntheticFrame:
    """Render one source frame and the synthetic Pixhawk-style pose contract."""

    intrinsics = intrinsics or demo_intrinsics()
    mission_config = mission_config or demo_mission_config()
    if index < 0:
        raise ValueError("frame index must be non-negative")

    columns, rows = np.meshgrid(
        np.arange(intrinsics.width, dtype=np.float64),
        np.arange(intrinsics.height, dtype=np.float64),
    )
    east = camera_east_m + FLIGHT_AGL_M * (columns - intrinsics.cx_px) / intrinsics.fx_px
    north = camera_north_m - FLIGHT_AGL_M * (rows - intrinsics.cy_px) / intrinsics.fy_px
    rgb, thermal, semantic = analytic_terrain_layers(east, north)

    # A small, hot vehicle-like ellipse moves within every frame.  It is part
    # of the raw evidence but must not contaminate the static terrain product.
    angle = 0.41 * index
    object_east = camera_east_m + 5.0 * np.cos(angle)
    object_north = camera_north_m + 3.5 * np.sin(angle)
    delta_east = east - object_east
    delta_north = north - object_north
    along = np.cos(angle) * delta_east + np.sin(angle) * delta_north
    across = -np.sin(angle) * delta_east + np.cos(angle) * delta_north
    dynamic_mask = (along / 2.5) ** 2 + (across / 1.15) ** 2 <= 1.0
    rgb[dynamic_mask] = np.array((245, 58, 22), dtype=np.uint8)
    thermal[dynamic_mask] = np.float32(0.99)
    semantic[dynamic_mask] = np.uint8(DYNAMIC_CLASS_ID)

    radius_squared = (
        ((columns - intrinsics.cx_px) / (intrinsics.width / 2.0)) ** 2
        + ((rows - intrinsics.cy_px) / (intrinsics.height / 2.0)) ** 2
    )
    support = np.clip(0.98 - 0.18 * radius_squared, 0.65, 0.98).astype(np.float32)

    local_frame = LocalENUFrame(
        GeodeticCoordinate(
            mission_config.origin_latitude_deg,
            mission_config.origin_longitude_deg,
            mission_config.ground_ellipsoid_height_m,
        ),
        frame_id=f"{mission_config.mission_id}:synthetic-render",
    )
    camera_geodetic = enu_to_geodetic(
        np.array((camera_east_m, camera_north_m, FLIGHT_AGL_M), dtype=np.float64),
        local_frame,
    )
    camera_flu, camera_optical = _camera_quaternions()
    capture_utc_ns = CAPTURE_START_UTC_NS + index * 1_000_000_000
    image_name = f"source_frames/SYNTHETIC_frame_{index:03d}_rgb.png"
    record = CameraPoseRecord(
        image_name=image_name,
        image_index=index,
        latitude_deg=camera_geodetic.latitude_deg,
        longitude_deg=camera_geodetic.longitude_deg,
        altitude_msl_m=camera_geodetic.ellipsoid_height_m - GEOID_SEPARATION_M,
        relative_altitude_m=FLIGHT_AGL_M,
        quaternion_camera_flu_to_enu_wxyz=camera_flu,
        quaternion_camera_optical_to_enu_wxyz=camera_optical,
        yaw_deg=0.0,
        pitch_deg=-90.0,
        roll_deg=0.0,
        capture_monotonic_ns=None,
        capture_utc_ns=capture_utc_ns,
        event_monotonic_ns=None,
        event_utc_ns=capture_utc_ns,
        clock_domain="SYNTHETIC:deterministic-UTC-v1",
        time_basis="utc",
        time_uncertainty_ns=100_000,
        horizontal_accuracy_m=0.04,
        vertical_accuracy_m=0.06,
        attitude_accuracy_deg=0.02,
        fix_type=3,
        fix_quality="synthetic_exact",
        rtk_status="not_applicable_synthetic",
        source_message="SYNTHETIC_CAMERA_POSE_RECORD_NOT_TELEMETRY",
        position_source="analytic_synthetic_flight_plan",
        attitude_source="analytic_nadir_camera_transform",
        interpolation_span_ns=None,
        position_reference="camera_optical_center",
        input_attitude_profile="synthetic_analytic_camera_pose",
    )
    return SyntheticFrame(
        record=record,
        rgb=rgb,
        thermal_normalized=thermal,
        semantic_class=semantic,
        dynamic_mask=dynamic_mask,
        support=support,
        east_enu_m=east.astype(np.float32),
        north_enu_m=north.astype(np.float32),
    )


def _save_source_frame(destination: Path, frame: SyntheticFrame) -> dict[str, str]:
    stem = Path(frame.record.image_name).stem.removesuffix("_rgb")
    rgb_path = destination / frame.record.image_name
    thermal_path = destination / "source_frames" / f"{stem}_thermal_u16.png"
    semantic_path = destination / "source_frames" / f"{stem}_semantic.png"
    dynamic_path = destination / "source_frames" / f"{stem}_dynamic_mask.png"
    exact_path = destination / "source_frames" / f"{stem}_layers.npz"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(frame.rgb, mode="RGB").save(rgb_path, format="PNG")
    thermal_u16 = np.rint(frame.thermal_normalized * 65_535.0).astype(np.uint16)
    Image.fromarray(thermal_u16).save(thermal_path, format="PNG")
    Image.fromarray(frame.semantic_class, mode="L").save(semantic_path, format="PNG")
    Image.fromarray(frame.dynamic_mask.astype(np.uint8) * 255, mode="L").save(
        dynamic_path,
        format="PNG",
    )
    np.savez_compressed(
        exact_path,
        rgb=frame.rgb,
        thermal_normalized=frame.thermal_normalized,
        semantic_class=frame.semantic_class,
        dynamic_mask=frame.dynamic_mask,
        east_enu_m=frame.east_enu_m,
        north_enu_m=frame.north_enu_m,
        support=frame.support,
    )
    return {
        "rgb": frame.record.image_name,
        "thermal_u16": thermal_path.relative_to(destination).as_posix(),
        "semantic": semantic_path.relative_to(destination).as_posix(),
        "dynamic_mask": dynamic_path.relative_to(destination).as_posix(),
        "exact_layers": exact_path.relative_to(destination).as_posix(),
    }


def _stamp_synthetic_warning(path: Path) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    banner_height = 13
    draw.rectangle((0, 0, image.width, banner_height), fill=(8, 8, 8))
    draw.text((3, 2), "SYNTHETIC / NOT REAL", fill=(255, 218, 52))
    image.save(path, format="PNG")


def _preflight_destination(destination: Path, overwrite: bool) -> None:
    if not destination.exists():
        return
    if not destination.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {destination}")
    existing = tuple(destination.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {destination}; pass --overwrite to replace demo files"
        )


def build_demo(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    overwrite: bool = False,
) -> DemoBuildResult:
    """Generate, integrate, and save the deterministic synthetic mission."""

    destination = Path(output).expanduser().resolve()
    _preflight_destination(destination, overwrite)

    intrinsics = demo_intrinsics()
    mission_config = demo_mission_config()
    mapper = AtlasMissionMapper(mission_config, intrinsics)
    semantic_contract = SemanticFrameContract(
        model_id="SYNTHETIC:analytic-terrain-renderer-v1",
        model_artifact_id="SYNTHETIC:no-learned-weights",
        taxonomy_id="SYNTHETIC:terrain-taxonomy-v1",
        reference_camera_calibration_id=intrinsics.calibration_id,
        class_labels={
            0: "vegetation",
            1: "road",
            2: "water",
            3: "structure",
            4: "exposed_soil",
            DYNAMIC_CLASS_ID: "dynamic_vehicle",
        },
        dynamic_class_ids=frozenset({DYNAMIC_CLASS_ID}),
    )
    frames: list[SyntheticFrame] = []
    integration_records: list[dict[str, object]] = []
    for index, (east_m, north_m) in enumerate(flight_plan_enu()):
        frame = render_synthetic_frame(
            index,
            east_m,
            north_m,
            intrinsics=intrinsics,
            mission_config=mission_config,
        )
        result = mapper.integrate_capture(
            frame.record,
            frame.rgb,
            thermal_normalized=frame.thermal_normalized,
            thermal_contract=RegisteredThermalFrameContract(
                sensor_id="SYNTHETIC:analytic-thermal-v1",
                reference_camera_calibration_id=intrinsics.calibration_id,
                registration_calibration_id=(
                    "SYNTHETIC:co-rendered-rgb-thermal-registration-v1"
                ),
                normalization_id="SYNTHETIC:mission-fixed-thermal-scale-v1",
                capture_time_offset_ns=0,
                capture_time_uncertainty_ns=100_000,
                registration_rms_px=0.0,
                source_unit="relative_thermal_intensity",
                normalization_scope="mission_fixed",
                nuc_state="not_applicable_synthetic",
                nuc_epoch_id="SYNTHETIC:no-nuc-v1",
            ),
            semantic_class=frame.semantic_class,
            semantic_contract=semantic_contract,
            dynamic_mask=frame.dynamic_mask,
            support=frame.support,
            pixel_uncertainty_px=0.12,
            provenance={
                "data_origin": "analytic_synthetic_generator",
                "generator_id": GENERATOR_ID,
                "real_sensor_data": False,
                "real_gps_data": False,
                "external_dataset": None,
                "warning": SYNTHETIC_WARNING,
                "dynamic_policy": "true pixels excluded from static terrain atlas",
            },
        )
        if not result.accepted:
            raise RuntimeError(
                f"deterministic frame {frame.record.image_name} was rejected: {result.reason}"
            )
        dynamic_rows, dynamic_columns = np.nonzero(frame.dynamic_mask)
        object_result = mapper.project_object(
            frame.record.image_name,
            ImageObjectObservation(
                object_id="SYNTHETIC-moving-vehicle-001",
                label="synthetic_vehicle",
                anchor_u_px=float(np.mean(dynamic_columns)),
                anchor_v_px=float(np.mean(dynamic_rows)),
                reference_camera_calibration_id=intrinsics.calibration_id,
                confidence=1.0,
                source="analytic_synthetic_generator",
                provenance={
                    "synthetic": True,
                    "warning": SYNTHETIC_WARNING,
                },
            ),
            pixel_uncertainty_px=0.12,
        )
        if not object_result.accepted:
            raise RuntimeError(
                f"synthetic object for {frame.record.image_name} was rejected: "
                f"{object_result.reason}"
            )
        frames.append(frame)
        integration_records.append(
            {
                "image_name": frame.record.image_name,
                "accepted": result.accepted,
                "reason": result.reason,
                "integrated_samples": (
                    result.integration.integrated_samples if result.integration else 0
                ),
                "dynamic_pixel_count": int(np.count_nonzero(frame.dynamic_mask)),
                "transient_object_id": object_result.observation.object_id,
            }
        )

    demonstration_metadata = {
        "synthetic": True,
        "warning": SYNTHETIC_WARNING,
        "generator_id": GENERATOR_ID,
        "real_sensor_data": False,
        "real_gps_data": False,
        "caltech_or_other_external_imagery_used": False,
        "dynamic_class_id": DYNAMIC_CLASS_ID,
        "dynamic_pixels_excluded_from_static_atlas": True,
    }
    atlas_paths = mapper.save_bundle(
        destination,
        overwrite=overwrite,
        metadata_additions={"demonstration": demonstration_metadata},
        preview_warning=SYNTHETIC_WARNING,
    )
    source_files = [_save_source_frame(destination, frame) for frame in frames]

    manifest_path = destination / "synthetic_demo_manifest.json"
    total_dynamic_pixels = sum(
        int(np.count_nonzero(frame.dynamic_mask)) for frame in frames
    )
    manifest = {
        "schema_version": "openprism.synthetic-atlas-demo/1.0",
        "warning": SYNTHETIC_WARNING,
        "synthetic": True,
        "generator_id": GENERATOR_ID,
        "determinism": "closed-form analytic fields; no random-number generator",
        "data_provenance": {
            "real_sensor_data": False,
            "real_gps_data": False,
            "caltech_imagery_used": False,
            "external_datasets_used": [],
        },
        "mission": {
            "mission_id": mission_config.mission_id,
            "capture_count": len(frames),
            "accepted_capture_count": len(mapper.accepted_records),
            "rejected_capture_count": len(mapper.results) - len(mapper.accepted_records),
            "flight_plan_enu_m": [list(position) for position in flight_plan_enu()],
            "flight_agl_m": FLIGHT_AGL_M,
            "orientation": "north-up nadir OpenCV optical camera",
            "overlap": "deliberate along-track and cross-track overlap",
        },
        "camera": {
            "calibration_id": intrinsics.calibration_id,
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx_px": intrinsics.fx_px,
            "fy_px": intrinsics.fy_px,
            "cx_px": intrinsics.cx_px,
            "cy_px": intrinsics.cy_px,
        },
        "terrain": {
            "model": "analytic planar reflectance/thermal/semantic field",
            "static_semantic_classes": {
                "0": "vegetation",
                "1": "road",
                "2": "water",
                "3": "structure",
                "4": "exposed soil",
            },
        },
        "dynamic_object": {
            "semantic_class_id": DYNAMIC_CLASS_ID,
            "total_source_pixels": total_dynamic_pixels,
            "visible_in_source_evidence": True,
            "excluded_from_static_atlas": True,
        },
        "integrations": integration_records,
        "source_files": source_files,
        "bundle_files": {
            "arrays": atlas_paths.arrays.name,
            "metadata": atlas_paths.metadata.name,
            "rgb_preview": atlas_paths.rgb_preview.name,
            "thermal_preview": atlas_paths.thermal_preview.name,
            "support_preview": atlas_paths.support_preview.name,
            "bounds_geojson": atlas_paths.bounds_geojson.name,
            "objects_geojson": atlas_paths.objects_geojson.name,
            "odm_geo": atlas_paths.odm_geo.name,
            "bundle_manifest": atlas_paths.manifest.name,
            "generation_id": atlas_paths.directory.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    notice_path = destination / "SYNTHETIC_DEMO_NOTICE.txt"
    notice_path.write_text(
        SYNTHETIC_WARNING
        + "\n\n"
        + "All imagery, thermal values, semantic labels, timestamps, and coordinates "
        + "in this directory were generated analytically for software verification.\n"
        + "They are not Caltech data, are not a recorded Pixhawk mission, and must not "
        + "be represented as observed terrain or survey evidence.\n"
        + "Dynamic-object pixels remain in source_frames/ but were masked out of the "
        + "static atlas layers.\n",
        encoding="utf-8",
    )

    return DemoBuildResult(
        atlas=atlas_paths,
        manifest=manifest_path,
        notice=notice_path,
        source_frames=destination / "source_frames",
        accepted_capture_count=len(mapper.accepted_records),
        rejected_capture_count=len(mapper.results) - len(mapper.accepted_records),
        dynamic_pixel_count=total_dynamic_pixels,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clearly labelled, deterministic synthetic OpenPRISM Atlas mission."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"bundle directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the known demo files if the output directory is non-empty",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_demo(arguments.output, overwrite=arguments.overwrite)
    print(SYNTHETIC_WARNING)
    print(f"bundle: {result.atlas.directory}")
    print(
        "captures: "
        f"{result.accepted_capture_count} accepted, "
        f"{result.rejected_capture_count} rejected"
    )
    print(f"dynamic source pixels excluded: {result.dynamic_pixel_count}")
    print(f"manifest: {result.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "DYNAMIC_CLASS_ID",
    "DemoBuildResult",
    "SyntheticFrame",
    "SYNTHETIC_WARNING",
    "analytic_terrain_layers",
    "build_demo",
    "demo_intrinsics",
    "demo_mission_config",
    "flight_plan_enu",
    "render_synthetic_frame",
]
