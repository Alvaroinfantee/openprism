from __future__ import annotations

from dataclasses import replace
import json
import math
import sys
import unittest
from unittest.mock import patch

from openprism.pixhawk import (
    DECLARED_CAMERA_IMAGE_ATTITUDE_PROFILE,
    ImageReference,
    InterpolationError,
    MessageFormatError,
    OptionalDependencyError,
    PixhawkBridge,
    PixhawkBridgeConfig,
    TimingDomainError,
    TimingError,
    export_odm_geo_txt,
    iter_pymavlink_messages,
    match_capture_events,
    mavlink_ned_frd_to_openprism_enu_flu,
    mavlink_ned_frd_to_openprism_enu_optical,
)


def message(kind: str, **fields: object) -> dict[str, object]:
    if kind == "CAMERA_IMAGE_CAPTURED":
        fields.setdefault(
            "camera_attitude_profile",
            DECLARED_CAMERA_IMAGE_ATTITUDE_PROFILE,
        )
        fields.setdefault("camera_position_reference", "camera_optical_center")
    return {"mavpackettype": kind, **fields}


class CoordinateConventionTests(unittest.TestCase):
    def test_level_north_camera_converts_to_enu_flu(self) -> None:
        converted = mavlink_ned_frd_to_openprism_enu_flu((1.0, 0.0, 0.0, 0.0))
        root_half = math.sqrt(0.5)
        # A north-facing FLU x-axis points along ENU +Y: +90 degrees about +Z.
        for actual, expected in zip(converted, (root_half, 0.0, 0.0, root_half)):
            self.assertAlmostEqual(actual, expected, places=7)
        optical = mavlink_ned_frd_to_openprism_enu_optical((1.0, 0.0, 0.0, 0.0))
        for actual, expected in zip(optical, (root_half, -root_half, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=7)


class DirectCaptureTests(unittest.TestCase):
    @staticmethod
    def valid_record():
        return match_capture_events(
            [
                message(
                    "CAMERA_IMAGE_CAPTURED",
                    time_boot_ms=1_000,
                    lat=40_000_000,
                    lon=-30_000_000,
                    alt=10_000,
                    relative_alt=5_000,
                    q=[1.0, 0.0, 0.0, 0.0],
                    image_index=0,
                    capture_result=1,
                    fix_type=3,
                )
            ],
            ["captured.jpg"],
        )[0]

    def test_camera_image_captured_preserves_dual_time_and_quality(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_250,
            time_utc=1_700_000_000_250_000,
            clock_domain="pixhawk-A:boot-17",
            time_uncertainty_ns=250_000,
            lat=37_123_456,
            lon=-122_987_654,
            alt=123_456,
            relative_alt=44_500,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=7,
            capture_result=1,
            file_url="file:///DCIM/100MEDIA/PXH0007.JPG",
            horizontal_accuracy_m=0.025,
            vertical_accuracy_m=0.040,
            attitude_accuracy_deg=0.20,
            fix_type=6,
        )
        records = match_capture_events([captured], {7: "rgb/PXH0007.JPG"})
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.image_name, "rgb/PXH0007.JPG")
        self.assertEqual(record.image_index, 7)
        self.assertAlmostEqual(record.latitude_deg, 3.7123456)
        self.assertAlmostEqual(record.longitude_deg, -12.2987654)
        self.assertAlmostEqual(record.altitude_msl_m, 123.456)
        self.assertEqual(record.capture_monotonic_ns, 1_250_000_000)
        self.assertEqual(record.capture_utc_ns, 1_700_000_000_250_000_000)
        self.assertEqual(record.time_basis, "mavlink_system_boot+utc")
        self.assertEqual(record.clock_domain, "pixhawk-A:boot-17")
        self.assertEqual(record.fix_quality, "rtk_fixed")
        self.assertEqual(record.rtk_status, "fixed")
        self.assertEqual(record.position_source, "CAMERA_IMAGE_CAPTURED")
        self.assertEqual(record.attitude_source, "CAMERA_IMAGE_CAPTURED")
        self.assertIsNone(record.interpolation_span_ns)
        # The neutral record can enter a JSON event log without an encoder.
        encoded = json.dumps(record.as_dict())
        self.assertIn('"monotonic_ns": 1250000000', encoded)
        self.assertIn('"rtk_status": "fixed"', encoded)

    def test_camera_message_is_preferred_over_trigger(self) -> None:
        messages = [
            message("CAMERA_TRIGGER", time_usec=1_000_000, seq=99),
            message(
                "CAMERA_IMAGE_CAPTURED",
                time_boot_ms=1_000,
                lat=40_000_000,
                lon=-30_000_000,
                alt=10_000,
                q=[1.0, 0.0, 0.0, 0.0],
                image_index=4,
                capture_result=1,
            ),
        ]
        records = PixhawkBridge().parse(messages, {4: "preferred.jpg"})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_message, "CAMERA_IMAGE_CAPTURED")
        self.assertEqual(records[0].image_name, "preferred.jpg")

    def test_trigger_shutter_lag_does_not_shift_captured_exposure_record(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_000,
            lat=40_000_000,
            lon=-30_000_000,
            alt=10_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            capture_result=1,
        )
        record = match_capture_events(
            [captured],
            ["captured.jpg"],
            config=PixhawkBridgeConfig(
                shutter_lag_ms=125.0,
                shutter_lag_uncertainty_ms=3.0,
            ),
        )[0]
        self.assertEqual(record.capture_monotonic_ns, 1_000_000_000)
        self.assertEqual(record.event_monotonic_ns, 1_000_000_000)
        self.assertEqual(record.position_source, "CAMERA_IMAGE_CAPTURED")
        self.assertEqual(record.attitude_source, "CAMERA_IMAGE_CAPTURED")
        self.assertIsNone(record.interpolation_span_ns)

    def test_captured_clock_correction_preserves_authoritative_camera_pose(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_000,
            time_uncertainty_ns=100_000,
            lat=40_000_000,
            lon=-30_000_000,
            alt=10_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            capture_result=1,
        )
        record = match_capture_events(
            [captured],
            ["captured.jpg"],
            config=PixhawkBridgeConfig(
                captured_event_time_correction_ms=2.0,
                captured_event_time_uncertainty_ms=0.5,
            ),
        )[0]
        self.assertEqual(record.event_monotonic_ns, 1_000_000_000)
        self.assertEqual(record.capture_monotonic_ns, 1_002_000_000)
        self.assertEqual(record.time_uncertainty_ns, 600_000)
        self.assertEqual(record.position_source, "CAMERA_IMAGE_CAPTURED")
        self.assertEqual(record.attitude_source, "CAMERA_IMAGE_CAPTURED")

    def test_pose_record_rejects_invalid_numeric_and_timing_invariants(self) -> None:
        record = self.valid_record()
        invalid_changes = (
            {"image_index": -7},
            {"relative_altitude_m": float("nan")},
            {"capture_monotonic_ns": -5},
            {"event_monotonic_ns": -5},
            {"time_uncertainty_ns": -9},
            {"time_basis": "utc"},
            {
                "quaternion_camera_flu_to_enu_wxyz": (
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            },
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(record, **changes)

        spoofed = replace(record, fix_quality="rtk_fixed", rtk_status="fixed")
        self.assertEqual(spoofed.fix_quality, "3d_fix")
        self.assertEqual(spoofed.rtk_status, "not_rtk")

    def test_pose_record_normalizes_integer_source_identifiers(self) -> None:
        record = replace(self.valid_record(), system_id="1", component_id="100")
        self.assertEqual(record.system_id, 1)
        self.assertIsInstance(record.system_id, int)
        self.assertEqual(record.component_id, 100)

    def test_external_capture_identifiers_require_exact_nonnegative_integers(
        self,
    ) -> None:
        base = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_000,
            lat=40_000_000,
            lon=-30_000_000,
            alt=10_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            camera_id=1,
            capture_result=1,
        )
        for field in ("image_index", "camera_id", "camera_device_id"):
            for invalid in (True, 1.5, -1):
                captured = dict(base)
                if field == "camera_device_id":
                    captured.pop("camera_id")
                captured[field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(
                    MessageFormatError
                ):
                    match_capture_events([captured], ["captured.jpg"])

        for invalid in (True, 1.5, -1):
            telemetry = FallbackInterpolationTests.telemetry()
            telemetry[-1]["seq"] = invalid
            with self.subTest(field="seq", invalid=invalid), self.assertRaises(
                MessageFormatError
            ):
                match_capture_events(telemetry, ["triggered.jpg"])

        for invalid in (True, 1.5, -1):
            with self.subTest(reference_index=invalid), self.assertRaises(ValueError):
                ImageReference("frame.jpg", image_index=invalid)

    def test_unknown_capture_time_is_rejected_even_with_direct_pose(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_utc=0,
            lat=40_000_000,
            lon=-30_000_000,
            alt=10_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            capture_result=1,
        )
        with self.assertRaises(TimingError):
            match_capture_events([captured])

    def test_index_and_file_url_conflict_is_rejected_by_default(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_000,
            lat=40_000_000,
            lon=-30_000_000,
            alt=10_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=9,
            capture_result=1,
            file_url="right.jpg",
        )
        images = [
            ImageReference("wrong.jpg", image_index=9),
            ImageReference("right.jpg", image_index=8),
        ]
        with self.assertRaises(MessageFormatError):
            match_capture_events([captured], images)

    def test_multiple_vehicle_systems_require_explicit_selection(self) -> None:
        first = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_000,
            lat=40_000_000,
            lon=-30_000_000,
            alt=10_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=1,
            capture_result=1,
            _srcSystem=1,
        )
        second = dict(first, image_index=2, _srcSystem=2)
        with self.assertRaises(MessageFormatError):
            match_capture_events([first, second])

    def test_system_aliases_and_capture_components_are_selected_strictly(self) -> None:
        base = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_000,
            lat=40_000_000,
            lon=-30_000_000,
            alt=10_000,
            q=[1.0, 0.0, 0.0, 0.0],
            capture_result=1,
            camera_id=0,
        )
        systems = [
            dict(base, image_index=1, system_id=1),
            dict(base, image_index=2, system_id=2),
        ]
        with self.assertRaises(MessageFormatError):
            match_capture_events(systems)
        selected = match_capture_events(
            systems,
            config=PixhawkBridgeConfig(system_id=1),
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].system_id, 1)

        components = [
            dict(base, image_index=3, system_id=1, component_id=100),
            dict(base, image_index=4, system_id=1, component_id=101),
        ]
        with self.assertRaises(MessageFormatError):
            match_capture_events(components)
        selected = match_capture_events(
            components,
            config=PixhawkBridgeConfig(system_id=1, capture_component_id=100),
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].component_id, 100)

    def test_partial_capture_acknowledgements_do_not_drop_other_triggers(self) -> None:
        messages = [
            message(
                "CAMERA_IMAGE_CAPTURED",
                time_boot_ms=1_000,
                lat=40_000_000,
                lon=-30_000_000,
                alt=10_000,
                q=[1.0, 0.0, 0.0, 0.0],
                image_index=0,
                capture_result=1,
            ),
            message(
                "GLOBAL_POSITION_INT",
                time_boot_ms=2_000,
                lat=40_000_100,
                lon=-30_000_100,
                alt=10_500,
                horizontal_accuracy_m=0.1,
                vertical_accuracy_m=0.2,
                fix_type=3,
            ),
            message(
                "ATTITUDE_QUATERNION",
                time_boot_ms=2_000,
                q1=1.0,
                q2=0.0,
                q3=0.0,
                q4=0.0,
                attitude_accuracy_deg=0.2,
            ),
            message("CAMERA_TRIGGER", time_usec=2_000_000, seq=1),
        ]
        records = match_capture_events(
            messages,
            {0: "captured.jpg", 1: "triggered.jpg"},
        )
        self.assertEqual([record.source_message for record in records], [
            "CAMERA_IMAGE_CAPTURED",
            "CAMERA_TRIGGER",
        ])

    def test_implicit_boot_clock_reboot_is_rejected(self) -> None:
        first = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_000,
            lat=40_000_000,
            lon=-30_000_000,
            alt=10_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            capture_result=1,
        )
        second = dict(first, time_boot_ms=100, image_index=1)
        with self.assertRaisesRegex(TimingDomainError, "reset"):
            match_capture_events([first, second])

    def test_cross_component_boot_clock_requires_explicit_shared_domain(self) -> None:
        telemetry = [
            message(
                "GLOBAL_POSITION_INT",
                time_boot_ms=1_000,
                system_id=1,
                component_id=1,
                lat=40_000_000,
                lon=-30_000_000,
                alt=10_000,
                horizontal_accuracy_m=0.1,
                vertical_accuracy_m=0.2,
                fix_type=3,
            ),
            message(
                "ATTITUDE_QUATERNION",
                time_boot_ms=1_000,
                system_id=1,
                component_id=1,
                q1=1.0,
                q2=0.0,
                q3=0.0,
                q4=0.0,
                attitude_accuracy_deg=0.2,
            ),
            message(
                "CAMERA_TRIGGER",
                time_usec=1_000_000,
                system_id=1,
                component_id=100,
                seq=0,
            ),
        ]
        with self.assertRaisesRegex(TimingDomainError, "multiple MAVLink components"):
            match_capture_events(telemetry, ["frame.jpg"])

        synchronized = [dict(item, clock_domain="flight-epoch-7") for item in telemetry]
        record = match_capture_events(synchronized, ["frame.jpg"])[0]
        self.assertEqual(record.clock_domain, "flight-epoch-7")

    def test_navigation_component_and_gps_type_selectors_prevent_stream_mixing(
        self,
    ) -> None:
        telemetry = FallbackInterpolationTests.telemetry()
        for item in telemetry:
            if item["mavpackettype"] in {
                "GLOBAL_POSITION_INT",
                "ATTITUDE_QUATERNION",
            }:
                item["system_id"] = 1
                item["component_id"] = 1
            elif item["mavpackettype"] == "CAMERA_TRIGGER":
                item["system_id"] = 1
                item["component_id"] = 100
        conflicting_position = {
            **next(
                item
                for item in telemetry
                if item["mavpackettype"] == "GLOBAL_POSITION_INT"
            ),
            "component_id": 2,
            "lat": 99_000_000,
        }
        with self.assertRaisesRegex(MessageFormatError, "multiple MAVLink sources"):
            match_capture_events(
                [*telemetry, conflicting_position], ["frame.jpg"]
            )
        selected = match_capture_events(
            [*telemetry, conflicting_position],
            ["frame.jpg"],
            config=PixhawkBridgeConfig(navigation_component_id=1),
        )[0]
        self.assertAlmostEqual(selected.latitude_deg, 1.0)

        gps_first = message(
            "GPS_RAW_INT",
            time_usec=1_000_000,
            time_basis="boot",
            clock_domain="flight-boot",
            time_uncertainty_ns=0,
            system_id=1,
            component_id=1,
            fix_type=6,
        )
        gps_second = {
            **gps_first,
            "mavpackettype": "GPS2_RAW",
            "time_usec": 1_200_000,
            "fix_type": 3,
        }
        with self.assertRaisesRegex(MessageFormatError, "multiple message types"):
            match_capture_events(
                [*telemetry, gps_first, gps_second], ["frame.jpg"]
            )
        selected_gps = match_capture_events(
            [*telemetry, gps_first, gps_second],
            ["frame.jpg"],
            config=PixhawkBridgeConfig(gps_message_type="gps_raw_int"),
        )[0]
        self.assertEqual(selected_gps.fix_type, 6)

    def test_standard_trigger_requires_explicit_camera_routing_when_selected(self) -> None:
        messages = FallbackInterpolationTests.telemetry()
        with self.assertRaises(MessageFormatError):
            match_capture_events(
                messages,
                ["frame.jpg"],
                config=PixhawkBridgeConfig(camera_id=2),
            )
        record = match_capture_events(
            messages,
            ["frame.jpg"],
            config=PixhawkBridgeConfig(
                camera_id=2,
                unidentified_trigger_camera_id=2,
            ),
        )[0]
        self.assertEqual(record.camera_id, 2)


class FallbackInterpolationTests(unittest.TestCase):
    @staticmethod
    def telemetry(*, trigger_domain: str = "flight-boot") -> list[dict[str, object]]:
        root_half = math.sqrt(0.5)
        return [
            message(
                "GLOBAL_POSITION_INT",
                time_boot_ms=1_000,
                clock_domain="flight-boot",
                time_uncertainty_ns=0,
                lat=10_000_000,
                lon=20_000_000,
                alt=100_000,
                relative_alt=20_000,
                horizontal_accuracy_m=0.10,
                vertical_accuracy_m=0.20,
                fix_type=6,
            ),
            message(
                "GLOBAL_POSITION_INT",
                time_boot_ms=1_200,
                clock_domain="flight-boot",
                time_uncertainty_ns=0,
                lat=12_000_000,
                lon=24_000_000,
                alt=120_000,
                relative_alt=22_000,
                horizontal_accuracy_m=0.12,
                vertical_accuracy_m=0.25,
                fix_type=5,
            ),
            message(
                "ATTITUDE_QUATERNION",
                time_boot_ms=1_000,
                clock_domain="flight-boot",
                time_uncertainty_ns=0,
                q1=1.0,
                q2=0.0,
                q3=0.0,
                q4=0.0,
                attitude_accuracy_deg=0.3,
            ),
            message(
                "ATTITUDE_QUATERNION",
                time_boot_ms=1_200,
                clock_domain="flight-boot",
                time_uncertainty_ns=0,
                q1=root_half,
                q2=0.0,
                q3=0.0,
                q4=-root_half,
                attitude_accuracy_deg=0.5,
            ),
            message(
                "CAMERA_TRIGGER",
                time_usec=1_000_000,
                time_basis="boot",
                clock_domain=trigger_domain,
                time_uncertainty_ns=0,
                seq=0,
            ),
        ]

    def test_bounded_position_and_slerp_with_shutter_lag(self) -> None:
        config = PixhawkBridgeConfig(
            shutter_lag_ms=100.0,
            shutter_lag_uncertainty_ms=2.0,
            max_interpolation_gap_ms=250.0,
            position_acceleration_bound_mps2=0.0,
            angular_acceleration_bound_deg_s2=0.0,
        )
        record = match_capture_events(
            self.telemetry(), ["frame000.jpg"], config=config
        )[0]
        self.assertAlmostEqual(record.latitude_deg, 1.1)
        self.assertAlmostEqual(record.longitude_deg, 2.2)
        self.assertAlmostEqual(record.altitude_msl_m, 110.0)
        self.assertAlmostEqual(record.relative_altitude_m or 0.0, 21.0)
        self.assertAlmostEqual(record.yaw_deg, 45.0, places=5)
        self.assertAlmostEqual(record.pitch_deg, 0.0, places=5)
        self.assertAlmostEqual(record.roll_deg, 0.0, places=5)
        self.assertEqual(record.event_monotonic_ns, 1_000_000_000)
        self.assertEqual(record.capture_monotonic_ns, 1_100_000_000)
        self.assertEqual(record.time_uncertainty_ns, 2_000_000)
        self.assertEqual(record.interpolation_span_ns, 200_000_000)
        self.assertEqual(record.horizontal_accuracy_m, 0.12)
        self.assertEqual(record.vertical_accuracy_m, 0.25)
        self.assertEqual(record.attitude_accuracy_deg, 0.5)
        # Conservative pair quality chooses RTK float over RTK fixed.
        self.assertEqual(record.fix_type, 5)
        self.assertEqual(record.rtk_status, "float")
        self.assertIn("identity_mount_assumption", record.attitude_source)

    def test_motion_model_bounds_are_required_for_genuine_interpolation(self) -> None:
        with self.assertRaisesRegex(
            InterpolationError,
            "position_acceleration_bound_mps2",
        ):
            match_capture_events(
                self.telemetry(),
                ["frame000.jpg"],
                config=PixhawkBridgeConfig(shutter_lag_ms=100.0),
            )
        with self.assertRaisesRegex(
            InterpolationError,
            "angular_acceleration_bound_deg_s2",
        ):
            match_capture_events(
                self.telemetry(),
                ["frame000.jpg"],
                config=PixhawkBridgeConfig(
                    shutter_lag_ms=100.0,
                    position_acceleration_bound_mps2=0.0,
                ),
            )

    def test_curved_high_dynamics_inflate_pose_accuracy(self) -> None:
        telemetry = self.telemetry()
        # Identical endpoint poses do not prove a stationary path: the vehicle
        # can curve away and return between samples.
        telemetry[1].update(
            lat=telemetry[0]["lat"],
            lon=telemetry[0]["lon"],
            alt=telemetry[0]["alt"],
            relative_alt=telemetry[0]["relative_alt"],
        )
        telemetry[3].update(
            q1=telemetry[2]["q1"],
            q2=telemetry[2]["q2"],
            q3=telemetry[2]["q3"],
            q4=telemetry[2]["q4"],
        )
        record = match_capture_events(
            telemetry,
            ["frame000.jpg"],
            config=PixhawkBridgeConfig(
                shutter_lag_ms=100.0,
                position_acceleration_bound_mps2=40.0,
                angular_acceleration_bound_deg_s2=720.0,
            ),
        )[0]
        # At the midpoint of a 0.2 s bracket, a*T^2/8 contributes
        # 0.2 m and 3.6 degrees in addition to endpoint accuracy.
        self.assertAlmostEqual(record.horizontal_accuracy_m or 0.0, 0.32)
        self.assertAlmostEqual(record.vertical_accuracy_m or 0.0, 0.45)
        self.assertAlmostEqual(record.attitude_accuracy_deg or 0.0, 4.1)

    def test_cross_domain_boot_timing_is_rejected(self) -> None:
        with self.assertRaises(TimingDomainError):
            match_capture_events(
                self.telemetry(trigger_domain="camera-boot"),
                ["frame000.jpg"],
            )

    def test_unbounded_interpolation_is_rejected(self) -> None:
        with self.assertRaises(InterpolationError):
            match_capture_events(
                self.telemetry(),
                ["frame000.jpg"],
                config=PixhawkBridgeConfig(
                    shutter_lag_ms=100.0,
                    max_interpolation_gap_ms=100.0,
                ),
            )

    def test_timestamped_image_can_replace_missing_trigger(self) -> None:
        messages = [
            item for item in self.telemetry() if item["mavpackettype"] != "CAMERA_TRIGGER"
        ]
        image = ImageReference(
            "timed.jpg",
            image_index=3,
            capture_monotonic_ns=1_100_000_000,
            clock_domain="flight-boot",
        )
        record = match_capture_events(
            messages,
            [image],
            config=PixhawkBridgeConfig(
                position_acceleration_bound_mps2=0.0,
                angular_acceleration_bound_deg_s2=0.0,
            ),
        )[0]
        self.assertEqual(record.source_message, "IMAGE_REFERENCE")
        self.assertEqual(record.image_index, 3)
        self.assertAlmostEqual(record.latitude_deg, 1.1)

    def test_image_and_event_clock_domains_must_match(self) -> None:
        messages = self.telemetry()
        image = ImageReference(
            "mismatch.jpg",
            image_index=0,
            capture_monotonic_ns=1_000_000_000,
            clock_domain="camera-clock",
        )
        with self.assertRaises(TimingDomainError):
            match_capture_events(messages, [image])

    def test_conflicting_equal_time_telemetry_is_rejected_in_any_order(self) -> None:
        original = self.telemetry()[0]
        conflicting = {**original, "lat": 99_000_000}
        for duplicates in ((original, conflicting), (conflicting, original)):
            messages = [
                *self.telemetry(),
                *duplicates,
            ]
            with self.subTest(order=duplicates[0]["lat"]), self.assertRaisesRegex(
                MessageFormatError,
                r"conflicting GLOBAL_POSITION_INT samples at equal boot timestamp",
            ):
                match_capture_events(messages, ["frame000.jpg"])

    def test_quaternion_sign_duplicates_are_equivalent(self) -> None:
        messages = self.telemetry()
        attitude = messages[2]
        messages.append(
            {
                **attitude,
                "q1": -float(attitude["q1"]),
                "q2": -float(attitude["q2"]),
                "q3": -float(attitude["q3"]),
                "q4": -float(attitude["q4"]),
            }
        )
        record = match_capture_events(messages, ["frame000.jpg"])[0]
        self.assertAlmostEqual(record.latitude_deg, 1.0)

    def test_gps_quality_requires_a_bracket_and_uses_worst_endpoint(self) -> None:
        base = [
            item
            for item in self.telemetry()
            if item["mavpackettype"] != "GLOBAL_POSITION_INT"
        ]
        positions = [
            message(
                "GLOBAL_POSITION_INT",
                time_boot_ms=time_ms,
                clock_domain="flight-boot",
                time_uncertainty_ns=0,
                lat=latitude,
                lon=20_000_000,
                alt=100_000,
                relative_alt=20_000,
                horizontal_accuracy_m=0.10,
                vertical_accuracy_m=0.20,
            )
            for time_ms, latitude in ((1_000, 10_000_000), (1_200, 12_000_000))
        ]
        fixed = message(
            "GPS_RAW_INT",
            time_usec=1_000_000,
            time_basis="boot",
            clock_domain="flight-boot",
            time_uncertainty_ns=0,
            fix_type=6,
            h_acc=100,
            v_acc=200,
        )
        degraded = {
            **fixed,
            "time_usec": 1_200_000,
            "fix_type": 3,
            "h_acc": 500,
            "v_acc": 700,
        }
        bracketed = match_capture_events(
            [*positions, *base, fixed, degraded],
            ["frame000.jpg"],
            config=PixhawkBridgeConfig(
                shutter_lag_ms=100.0,
                position_acceleration_bound_mps2=0.0,
                angular_acceleration_bound_deg_s2=0.0,
            ),
        )[0]
        self.assertEqual(bracketed.fix_type, 3)
        self.assertEqual(bracketed.horizontal_accuracy_m, 0.5)
        self.assertEqual(bracketed.vertical_accuracy_m, 0.7)

        one_sided = match_capture_events(
            [*positions, *base, degraded],
            ["frame000.jpg"],
            config=PixhawkBridgeConfig(
                shutter_lag_ms=100.0,
                position_acceleration_bound_mps2=0.0,
                angular_acceleration_bound_deg_s2=0.0,
            ),
        )[0]
        self.assertIsNone(one_sided.fix_type)
        self.assertEqual(one_sided.fix_quality, "unknown")

    def test_direct_capture_fix_quality_remains_authoritative(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_100,
            clock_domain="flight-boot",
            time_uncertainty_ns=100_000,
            lat=10_000_000,
            lon=20_000_000,
            alt=100_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            capture_result=1,
            fix_type=6,
        )
        quality = [
            message(
                "GPS_RAW_INT",
                time_usec=time_us,
                time_basis="boot",
                clock_domain="flight-boot",
                time_uncertainty_ns=0,
                fix_type=3,
                h_acc=200,
                v_acc=300,
            )
            for time_us in (1_000_000, 1_200_000)
        ]
        record = match_capture_events(
            [captured, *quality], ["captured.jpg"]
        )[0]
        self.assertEqual(record.fix_type, 6)
        self.assertEqual(record.rtk_status, "fixed")
        self.assertEqual(record.horizontal_accuracy_m, 0.2)
        self.assertEqual(record.vertical_accuracy_m, 0.3)

    def test_partial_capture_uses_interpolated_position_quality(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_100,
            clock_domain="flight-boot",
            time_uncertainty_ns=100_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            capture_result=1,
            fix_type=6,
            horizontal_accuracy_m=0.02,
            vertical_accuracy_m=0.03,
        )
        positions = [
            message(
                "GLOBAL_POSITION_INT",
                time_boot_ms=time_ms,
                clock_domain="flight-boot",
                time_uncertainty_ns=0,
                lat=latitude,
                lon=20_000_000,
                alt=100_000,
                relative_alt=20_000,
                fix_type=3,
                horizontal_accuracy_m=0.5,
                vertical_accuracy_m=0.7,
            )
            for time_ms, latitude in ((1_000, 10_000_000), (1_200, 12_000_000))
        ]
        record = match_capture_events(
            [*positions, captured],
            ["captured.jpg"],
            config=PixhawkBridgeConfig(
                position_acceleration_bound_mps2=0.0,
            ),
        )[0]
        self.assertEqual(
            record.position_source,
            "GLOBAL_POSITION_INT:bounded_linear_interpolation",
        )
        self.assertEqual(record.attitude_source, "CAMERA_IMAGE_CAPTURED")
        self.assertAlmostEqual(record.latitude_deg, 1.1)
        self.assertEqual(record.fix_type, 3)
        self.assertEqual(record.fix_quality, "3d_fix")
        self.assertEqual(record.horizontal_accuracy_m, 0.5)
        self.assertEqual(record.vertical_accuracy_m, 0.7)

    def test_incomparable_optional_gps_does_not_invalidate_direct_quality(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_100,
            clock_domain="camera-boot",
            time_uncertainty_ns=100_000,
            lat=10_000_000,
            lon=20_000_000,
            alt=100_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            capture_result=1,
            fix_type=6,
        )
        gps = [
            message(
                "GPS_RAW_INT",
                time_usec=time_us,
                time_basis="boot",
                clock_domain="gps-boot",
                time_uncertainty_ns=50_000,
                fix_type=3,
                h_acc=200,
                v_acc=300,
            )
            for time_us in (1_000_000, 1_200_000)
        ]
        record = match_capture_events([captured, *gps], ["captured.jpg"])[0]
        self.assertEqual(record.fix_type, 6)
        self.assertEqual(record.rtk_status, "fixed")
        self.assertIsNone(record.horizontal_accuracy_m)
        self.assertIsNone(record.vertical_accuracy_m)
        self.assertEqual(record.time_uncertainty_ns, 100_000)

    def test_unknown_time_optional_gps_is_not_consumed(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=1_100,
            clock_domain="flight-boot",
            time_uncertainty_ns=100_000,
            lat=10_000_000,
            lon=20_000_000,
            alt=100_000,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            capture_result=1,
            fix_type=6,
        )
        gps = [
            message(
                "GPS_RAW_INT",
                time_usec=time_us,
                time_basis="boot",
                clock_domain="flight-boot",
                fix_type=3,
                h_acc=200,
                v_acc=300,
            )
            for time_us in (1_000_000, 1_200_000)
        ]
        record = match_capture_events([captured, *gps], ["captured.jpg"])[0]
        self.assertEqual(record.fix_type, 6)
        self.assertEqual(record.rtk_status, "fixed")
        self.assertIsNone(record.horizontal_accuracy_m)
        self.assertIsNone(record.vertical_accuracy_m)
        self.assertEqual(record.time_uncertainty_ns, 100_000)
        self.assertIsNone(record.interpolation_span_ns)

    def test_interpolation_endpoint_clock_uncertainty_is_propagated(self) -> None:
        messages = self.telemetry()
        endpoint_uncertainties = {
            "GLOBAL_POSITION_INT": iter((200_000, 400_000)),
            "ATTITUDE_QUATERNION": iter((300_000, 900_000)),
        }
        for item in messages:
            stream = str(item["mavpackettype"])
            if stream in endpoint_uncertainties:
                item["time_uncertainty_ns"] = next(
                    endpoint_uncertainties[stream]
                )
        trigger = next(
            item for item in messages if item["mavpackettype"] == "CAMERA_TRIGGER"
        )
        trigger["time_uncertainty_ns"] = 100_000
        record = match_capture_events(
            messages,
            ["frame000.jpg"],
            config=PixhawkBridgeConfig(
                shutter_lag_ms=100.0,
                shutter_lag_uncertainty_ms=0.05,
                position_acceleration_bound_mps2=0.0,
                angular_acceleration_bound_deg_s2=0.0,
            ),
        )[0]
        self.assertEqual(record.time_uncertainty_ns, 1_050_000)

        messages[0].pop("time_uncertainty_ns")
        unknown = match_capture_events(
            messages,
            ["frame000.jpg"],
            config=PixhawkBridgeConfig(
                shutter_lag_ms=100.0,
                shutter_lag_uncertainty_ms=0.05,
                position_acceleration_bound_mps2=0.0,
                angular_acceleration_bound_deg_s2=0.0,
            ),
        )[0]
        self.assertIsNone(unknown.time_uncertainty_ns)


class OpenDroneMapExportTests(unittest.TestCase):
    def test_geo_txt_has_projection_pose_and_accuracy(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=10,
            lat=515_000_000,
            lon=-1_250_000,
            alt=82_345,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            capture_result=1,
            horizontal_accuracy_m=0.02,
            vertical_accuracy_m=0.04,
        )
        record = match_capture_events([captured], ["image000.jpg"])[0]
        geo = export_odm_geo_txt([record], require_accuracy=True)
        lines = geo.splitlines()
        self.assertEqual(lines[0], "EPSG:4326")
        columns = lines[1].split()
        self.assertEqual(columns[0], "image000.jpg")
        # EPSG:4326 ODM rows are image, longitude/X, latitude/Y, altitude.
        self.assertEqual(columns[1:4], ["-0.1250000000", "51.5000000000", "82.345"])
        # Orientation/accuracy are deliberately omitted: geo.txt has no null
        # orientation fields, and 0/0/0 would be consumed as a real pose.
        self.assertEqual(len(columns), 4)
        self.assertNotIn("0.020", columns)

    def test_projected_header_is_rejected_without_coordinate_transform(self) -> None:
        captured = message(
            "CAMERA_IMAGE_CAPTURED",
            time_boot_ms=10,
            lat=515_000_000,
            lon=-1_250_000,
            alt=82_345,
            q=[1.0, 0.0, 0.0, 0.0],
            image_index=0,
            capture_result=1,
        )
        record = match_capture_events([captured])[0]
        with self.assertRaises(ValueError):
            export_odm_geo_txt([record], projection="EPSG:32630")


class OptionalPymavlinkTests(unittest.TestCase):
    def test_missing_pymavlink_has_actionable_lazy_error(self) -> None:
        # Importing openprism.pixhawk above succeeded without pymavlink. The
        # dependency is consulted only once the optional generator advances.
        with patch.dict(sys.modules, {"pymavlink": None}):
            with self.assertRaisesRegex(OptionalDependencyError, "pip install pymavlink"):
                next(iter_pymavlink_messages("flight.tlog", max_messages=1))


if __name__ == "__main__":
    unittest.main()
