from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openprism.atlas import (
    AtlasMissionConfig,
    AtlasMissionMapper,
    ImageObjectObservation,
    RegisteredThermalFrameContract,
    SemanticFrameContract,
)
from openprism.mapping import CameraIntrinsics, MappingFrame, Quaternion
from openprism.pixhawk import CameraPoseRecord


def _quaternion_tuple(rotation: np.ndarray) -> tuple[float, float, float, float]:
    value = Quaternion.from_rotation_matrix(rotation)
    return value.w, value.x, value.y, value.z


class AtlasMissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(
            width=4,
            height=4,
            fx_px=4.0,
            fy_px=4.0,
            cx_px=1.5,
            cy_px=1.5,
            calibration_id="camera-calibration:test:1",
            calibration_rms_px=0.1,
        )
        self.config = AtlasMissionConfig(
            mission_id="test-flight",
            origin_latitude_deg=34.0,
            origin_longitude_deg=-118.0,
            ground_elevation_msl_m=100.0,
            geoid_separation_m=-30.0,
            east_min_m=-20.0,
            east_max_m=20.0,
            north_min_m=-20.0,
            north_max_m=20.0,
            resolution_m=1.0,
            horizontal_crs_id="EPSG:4326",
            input_vertical_datum_id="TEST:MSL",
            geoid_model_id="TEST:constant-geoid",
            geoid_separation_uncertainty_m=0.05,
            tai_minus_utc_s=37,
            max_projected_uncertainty_m=3.0,
            timing_velocity_bound_mps=20.0,
            timing_angular_rate_bound_deg_s=90.0,
        )

    def thermal_contract(self, **changes: object) -> RegisteredThermalFrameContract:
        values: dict[str, object] = {
            "sensor_id": "thermal:test:1",
            "reference_camera_calibration_id": self.intrinsics.calibration_id,
            "registration_calibration_id": "rgb-thermal-registration:test:1",
            "normalization_id": "mission-fixed-scale:test:1",
            "capture_time_offset_ns": 0,
            "capture_time_uncertainty_ns": 1_000_000,
            "registration_rms_px": 0.25,
        }
        values.update(changes)
        return RegisteredThermalFrameContract(**values)  # type: ignore[arg-type]

    def semantic_contract(self, **changes: object) -> SemanticFrameContract:
        values: dict[str, object] = {
            "model_id": "semantic-model:test:1",
            "model_artifact_id": "sha256:test-weights",
            "taxonomy_id": "terrain-taxonomy:test:1",
            "reference_camera_calibration_id": self.intrinsics.calibration_id,
            "class_labels": {4: "soil", 90: "dynamic_vehicle"},
            "dynamic_class_ids": frozenset({90}),
        }
        values.update(changes)
        return SemanticFrameContract(**values)  # type: ignore[arg-type]

    def test_xywh_ground_anchor_uses_last_pixel_of_half_open_box(self) -> None:
        observation = ImageObjectObservation.from_xywh(
            "edge-track",
            "person",
            0.0,
            0.0,
            4.0,
            4.0,
            reference_camera_calibration_id=self.intrinsics.calibration_id,
        )
        self.assertEqual(observation.anchor_u_px, 1.5)
        self.assertEqual(observation.anchor_v_px, 3.0)
        self.assertLess(observation.anchor_v_px, 4.0)

    def record(self, name: str = "frame_0001.jpg", **changes: object) -> CameraPoseRecord:
        # Optical +x points east, +y south, and +z (boresight) down.
        optical_to_enu = _quaternion_tuple(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            )
        )
        flu_from_optical = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        optical_to_enu_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        )
        flu_to_enu = _quaternion_tuple(
            optical_to_enu_matrix @ flu_from_optical.T
        )
        values: dict[str, object] = {
            "image_name": name,
            "image_index": 1,
            "latitude_deg": 34.0,
            "longitude_deg": -118.0,
            "altitude_msl_m": 110.0,
            "relative_altitude_m": 10.0,
            "quaternion_camera_flu_to_enu_wxyz": flu_to_enu,
            "quaternion_camera_optical_to_enu_wxyz": optical_to_enu,
            "yaw_deg": 0.0,
            "pitch_deg": -90.0,
            "roll_deg": 0.0,
            "capture_monotonic_ns": 1_000_000_000,
            "capture_utc_ns": 1_700_000_000_000_000_000,
            "event_monotonic_ns": 1_000_000_000,
            "event_utc_ns": 1_700_000_000_000_000_000,
            "clock_domain": "mavlink:system:1:boot",
            "time_basis": "mavlink_system_boot+utc",
            "time_uncertainty_ns": 1_000_000,
            "horizontal_accuracy_m": 0.10,
            "vertical_accuracy_m": 0.15,
            "attitude_accuracy_deg": 0.10,
            "fix_type": 6,
            "fix_quality": "rtk_fixed",
            "rtk_status": "fixed",
            "source_message": "CAMERA_IMAGE_CAPTURED",
            "position_source": "CAMERA_IMAGE_CAPTURED",
            "attitude_source": "CAMERA_IMAGE_CAPTURED",
            "interpolation_span_ns": None,
            "position_reference": "camera_optical_center",
            "input_attitude_profile": "synthetic_test_camera_pose",
        }
        values.update(changes)
        return CameraPoseRecord(**values)  # type: ignore[arg-type]

    def test_pixhawk_capture_maps_to_north_up_static_layers(self) -> None:
        mapper = AtlasMissionMapper(self.config, self.intrinsics)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        thermal = np.full((4, 4), 0.75, dtype=np.float32)
        semantic = np.full((4, 4), 4, dtype=np.int32)

        result = mapper.integrate_capture(
            self.record(),
            rgb,
            thermal_normalized=thermal,
            thermal_contract=self.thermal_contract(),
            semantic_class=semantic,
            semantic_contract=self.semantic_contract(),
        )

        self.assertTrue(result.accepted, result.reason)
        snapshot = mapper.snapshot()
        self.assertGreater(np.count_nonzero(snapshot.valid), 0)
        self.assertTrue(np.allclose(snapshot.rgb[snapshot.valid, 0], 1.0))
        self.assertTrue(np.all(snapshot.semantic_class[snapshot.valid] == 4))
        self.assertEqual(snapshot.metadata["coordinate_reference"]["axes"], "ENU")
        source = snapshot.provenance["frame_0001.jpg"]["source"]
        self.assertEqual(source["vertical_conversion"]["input"], "TEST:MSL")
        self.assertEqual(source["time_conversion"]["output"], "TAI")

    def test_unknown_uncertainty_and_dynamic_only_frame_abstain(self) -> None:
        mapper = AtlasMissionMapper(self.config, self.intrinsics)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        unknown = mapper.integrate_capture(
            self.record(horizontal_accuracy_m=None), image
        )
        self.assertFalse(unknown.accepted)
        self.assertEqual(unknown.reason, "horizontal_accuracy_unknown")

        mapper = AtlasMissionMapper(self.config, self.intrinsics)
        dynamic = mapper.integrate_capture(
            self.record(name="dynamic.jpg"),
            image,
            dynamic_mask=np.ones((4, 4), dtype=bool),
        )
        self.assertFalse(dynamic.accepted)
        self.assertEqual(
            dynamic.reason,
            "no_supported_samples_after_uncertainty_gate",
        )
        self.assertEqual(np.count_nonzero(mapper.snapshot().valid), 0)

    def test_thermal_requires_registered_synchronized_mission_fixed_contract(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        thermal = np.full((4, 4), 0.5, dtype=np.float32)

        missing = AtlasMissionMapper(self.config, self.intrinsics).integrate_capture(
            self.record(), image, thermal_normalized=thermal
        )
        self.assertEqual(missing.reason, "thermal_registration_contract_unknown")

        mismatch = AtlasMissionMapper(self.config, self.intrinsics).integrate_capture(
            self.record(),
            image,
            thermal_normalized=thermal,
            thermal_contract=self.thermal_contract(
                reference_camera_calibration_id="different-rgb-calibration"
            ),
        )
        self.assertEqual(mismatch.reason, "thermal_reference_calibration_mismatch")

        late = AtlasMissionMapper(self.config, self.intrinsics).integrate_capture(
            self.record(),
            image,
            thermal_normalized=thermal,
            thermal_contract=self.thermal_contract(
                capture_time_offset_ns=self.config.max_thermal_time_offset_ns + 1
            ),
        )
        self.assertEqual(late.reason, "thermal_time_offset_exceeds_limit")

        with self.assertRaises(ValueError):
            self.thermal_contract(normalization_scope="per_frame_percentile")

    def test_semantics_require_one_mission_fixed_taxonomy_and_exclude_dynamics(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        semantic = np.full((4, 4), 4, dtype=np.int32)
        semantic[0, 0] = 90
        missing = AtlasMissionMapper(self.config, self.intrinsics).integrate_capture(
            self.record(), image, semantic_class=semantic
        )
        self.assertEqual(missing.reason, "semantic_contract_unknown")

        mapper = AtlasMissionMapper(self.config, self.intrinsics)
        accepted = mapper.integrate_capture(
            self.record(),
            image,
            semantic_class=semantic,
            semantic_contract=self.semantic_contract(),
        )
        self.assertTrue(accepted.accepted, accepted.reason)
        snapshot = mapper.snapshot()
        self.assertFalse(np.any(snapshot.semantic_class[snapshot.valid] == 90))
        changed = mapper.integrate_capture(
            self.record(name="frame_0002.jpg", image_index=2),
            image,
            semantic_class=semantic,
            semantic_contract=self.semantic_contract(model_artifact_id="different"),
        )
        self.assertEqual(changed.reason, "semantic_mission_contract_changed")

    def test_semantic_values_cannot_wrap_into_a_valid_taxonomy_class(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        contract = self.semantic_contract()
        for semantic in (
            np.full((4, 4), 2**32 + 4, dtype=np.uint64),
            np.full((4, 4), np.iinfo(np.int32).max + 1, dtype=np.int64),
            np.full((4, 4), np.iinfo(np.int32).min - 1, dtype=np.int64),
        ):
            mapper = AtlasMissionMapper(self.config, self.intrinsics)
            rejected = mapper.integrate_capture(
                self.record(),
                image,
                semantic_class=semantic,
                semantic_contract=contract,
            )
            self.assertFalse(rejected.accepted)
            self.assertEqual(rejected.reason, "semantic_values_outside_int32")
            self.assertFalse(np.any(mapper.snapshot().valid))

    def test_policy_integer_types_and_boot_anchor_uncertainty_are_not_truncated(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum_fix_type must be an integer"):
            replace(self.config, minimum_fix_type=3.9)
        with self.assertRaisesRegex(ValueError, "require_rtk_fixed must be boolean"):
            replace(self.config, require_rtk_fixed="false")

        utc_tai = 1_700_000_000_000_000_000 + 37_000_000_000
        boot_only = replace(
            self.config,
            tai_minus_utc_s=None,
            boot_epoch_tai_ns=utc_tai - 1_000_000_000,
            boot_epoch_uncertainty_ns=1_500_000,
            boot_clock_domain="mavlink:system:1:boot",
            max_time_uncertainty_ns=2_000_000,
        )
        record = self.record(
            capture_utc_ns=None,
            event_utc_ns=None,
            time_basis="mavlink_system_boot",
        )
        result = AtlasMissionMapper(boot_only, self.intrinsics).integrate_capture(
            record, np.zeros((4, 4, 3), dtype=np.uint8)
        )
        self.assertEqual(result.reason, "capture_time_uncertainty_exceeds_limit")

    def test_transient_object_is_geolocated_but_not_baked_into_terrain(self) -> None:
        mapper = AtlasMissionMapper(self.config, self.intrinsics)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        dynamic = np.zeros((4, 4), dtype=bool)
        dynamic[1:3, 1:3] = True
        accepted = mapper.integrate_capture(
            self.record(),
            image,
            dynamic_mask=dynamic,
        )
        self.assertTrue(accepted.accepted)

        located = mapper.project_object(
            "frame_0001.jpg",
            ImageObjectObservation.from_xywh(
                "track-7",
                "person",
                1.0,
                0.5,
                1.0,
                1.0,
                reference_camera_calibration_id=self.intrinsics.calibration_id,
                confidence=0.9,
            ),
            pixel_uncertainty_px=0.5,
        )
        self.assertTrue(located.accepted, located.reason)
        self.assertEqual(located.observation.label, "person")
        self.assertLess(located.observation.horizontal_uncertainty_m, 3.0)
        self.assertAlmostEqual(located.observation.ground_altitude_msl_m, 100.0, places=3)

        mismatched = mapper.project_object(
            "frame_0001.jpg",
            ImageObjectObservation(
                "wrong-space",
                "person",
                1.0,
                1.0,
                "different-camera-calibration",
            ),
        )
        self.assertFalse(mismatched.accepted)
        self.assertEqual(
            mismatched.reason, "object_reference_calibration_mismatch"
        )

    def test_explicit_time_scale_anchor_is_required(self) -> None:
        config = AtlasMissionConfig(
            mission_id="unanchored",
            origin_latitude_deg=34.0,
            origin_longitude_deg=-118.0,
            ground_elevation_msl_m=100.0,
            geoid_separation_m=-30.0,
            east_min_m=-10.0,
            east_max_m=10.0,
            north_min_m=-10.0,
            north_max_m=10.0,
            resolution_m=1.0,
            horizontal_crs_id="EPSG:4326",
            input_vertical_datum_id="TEST:MSL",
            geoid_model_id="TEST:constant-geoid",
            geoid_separation_uncertainty_m=0.05,
        )
        result = AtlasMissionMapper(config, self.intrinsics).integrate_capture(
            self.record(), np.zeros((4, 4, 3), dtype=np.uint8)
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "capture_time_has_no_explicit_tai_anchor")

    def test_bundle_is_inspectable_and_refuses_silent_overwrite(self) -> None:
        mapper = AtlasMissionMapper(self.config, self.intrinsics)
        image = np.full((4, 4, 3), 127, dtype=np.uint8)
        thermal = np.full((4, 4), 0.65, dtype=np.float32)
        self.assertTrue(
            mapper.integrate_capture(
                self.record(),
                image,
                thermal_normalized=thermal,
                thermal_contract=self.thermal_contract(),
            ).accepted
        )
        self.assertTrue(
            mapper.project_object(
                "frame_0001.jpg",
                ImageObjectObservation(
                    "track-1",
                    "vehicle",
                    1.5,
                    1.5,
                    self.intrinsics.calibration_id,
                ),
                pixel_uncertainty_px=0.5,
            ).accepted
        )

        with tempfile.TemporaryDirectory() as temporary:
            paths = mapper.save_bundle(temporary)
            for output in (
                paths.arrays,
                paths.metadata,
                paths.rgb_preview,
                paths.thermal_preview,
                paths.support_preview,
                paths.bounds_geojson,
                paths.objects_geojson,
                paths.odm_geo,
            ):
                self.assertTrue(Path(output).is_file(), output)
            with np.load(paths.arrays) as layers:
                self.assertIn("rgb", layers.files)
                self.assertIn("projected_uncertainty_m", layers.files)
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
            self.assertFalse(metadata["survey_grade"])
            self.assertEqual(metadata["accepted_capture_count"], 1)
            bounds = json.loads(paths.bounds_geojson.read_text(encoding="utf-8"))
            self.assertEqual(bounds["features"][0]["geometry"]["type"], "Polygon")
            objects = json.loads(paths.objects_geojson.read_text(encoding="utf-8"))
            self.assertEqual(objects["features"][0]["properties"]["label"], "vehicle")
            self.assertIsInstance(
                objects["features"][0]["properties"]["timestamp_tai_ns"], str
            )
            self.assertTrue(paths.odm_geo.read_text(encoding="utf-8").startswith("EPSG:4326\n"))
            with self.assertRaises(FileExistsError):
                mapper.save_bundle(temporary)

    def test_bundle_stages_atomically_and_normalizes_path_provenance(self) -> None:
        mapper = AtlasMissionMapper(self.config, self.intrinsics)
        self.assertTrue(
            mapper.integrate_capture(
                self.record(),
                np.zeros((4, 4, 3), dtype=np.uint8),
                provenance={"source_asset": Path("rgb/frame.jpg")},
            ).accepted
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = mapper.save_bundle(Path(temporary) / "path-provenance")
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
            caller = metadata["source_provenance"]["frame_0001.jpg"]["source"][
                "caller"
            ]
            self.assertEqual(caller["source_asset"], str(Path("rgb/frame.jpg")))

        invalid = AtlasMissionMapper(self.config, self.intrinsics)
        self.assertTrue(
            invalid.integrate_capture(
                self.record(),
                np.zeros((4, 4, 3), dtype=np.uint8),
                provenance={"invalid": float("nan")},
            ).accepted
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "must-remain-uncommitted"
            with self.assertRaisesRegex(ValueError, "NaN"):
                invalid.save_bundle(destination)
            self.assertFalse((destination / "atlas_layers.npz").exists())
            self.assertFalse((destination / "atlas_metadata.json").exists())

    def test_private_grid_bypass_is_detected_before_publication(self) -> None:
        mapper = AtlasMissionMapper(self.config, self.intrinsics)
        self.assertFalse(hasattr(mapper, "grid"))
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        self.assertTrue(mapper.integrate_capture(self.record(), image).accepted)
        _, timestamp, pose = mapper._accepted_context["frame_0001.jpg"]
        bypass = MappingFrame(
            frame_id="bypass.jpg",
            timestamp_tai_ns=timestamp,
            rgb=image,
            intrinsics=self.intrinsics,
            pose=pose,
        )
        self.assertTrue(mapper._grid.integrate(bypass).accepted)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "mission-gated"):
                mapper.save_bundle(temporary)

    def test_concurrent_publishers_cannot_both_win_without_overwrite(self) -> None:
        first = AtlasMissionMapper(self.config, self.intrinsics)
        second = AtlasMissionMapper(self.config, self.intrinsics)
        with tempfile.TemporaryDirectory() as temporary:
            def publish(mapper: AtlasMissionMapper) -> str:
                try:
                    mapper.save_bundle(temporary)
                    return "published"
                except FileExistsError:
                    return "rejected"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = tuple(executor.map(publish, (first, second)))
            self.assertEqual(outcomes.count("published"), 1)
            self.assertEqual(outcomes.count("rejected"), 1)


if __name__ == "__main__":
    unittest.main()
