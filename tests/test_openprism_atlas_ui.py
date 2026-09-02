from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from openprism.server import OperatorApplication


REPO_ROOT = Path(__file__).resolve().parents[1]


def _application_for(atlas_root: Path) -> OperatorApplication:
    # atlas_payload is deliberately independent from the dataset catalog.  Avoid
    # indexing the large replay datasets in these focused contract tests.
    application = object.__new__(OperatorApplication)
    application.atlas_root = atlas_root.resolve()
    return application


def _write_preview(path: Path, color: tuple[int, int, int], size=(7, 5)) -> None:
    pixels = np.empty((size[1], size[0], 3), dtype=np.uint8)
    pixels[...] = color
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def _write_bundle(root: Path) -> None:
    root.mkdir(parents=True)
    metadata = {
        "schema_version": "openprism.atlas-bundle/0.1",
        "mission_id": "flight-042",
        "product": "live_tactical_2.5d_mosaic",
        "survey_grade": False,
        "camera_calibration_id": "rig-cal-9",
        "data_provenance": {
            "real_sensor_data": True,
            "real_navigation_data": True,
        },
        "accepted_capture_count": 12,
        "rejected_capture_count": 2,
        "layers": ["rgb", "thermal_normalized", "support"],
        "source_ids": ["a", "b"],
        "private_operator_note": "must not cross the API boundary",
        "mosaic": {
            "coordinate_reference": {
                "type": "local_tangent_plane",
                "axes": "ENU",
                "horizontal_datum": "WGS84",
                "vertical_datum": "WGS84_ellipsoid",
                "frame_id": "flight-042:enu",
                "origin": {
                    "latitude_deg": 34.2,
                    "longitude_deg": -118.1,
                    "ellipsoid_height_m": 351.5,
                },
                "untrusted_extra": "not exposed",
            },
            "grid": {
                "north_up": True,
                "east_min_m": -10.0,
                "east_max_m": 11.0,
                "north_min_m": -7.5,
                "north_max_m": 7.5,
                "resolution_m": 3.0,
                "shape": [999, 999],
            },
        },
    }
    (root / "atlas_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _write_preview(root / "atlas_rgb.png", (20, 40, 30))
    _write_preview(root / "atlas_thermal.png", (180, 70, 10))
    _write_preview(root / "atlas_support.png", (10, 170, 80))


class AtlasOperatorPayloadTests(unittest.TestCase):
    def test_absent_bundle_is_truthfully_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "latest"
            payload = _application_for(missing).atlas_payload()

        self.assertFalse(payload["available"])
        self.assertFalse(payload["synthetic"])
        self.assertIn("No exported atlas", payload["reason"])
        self.assertEqual(payload["expected_bundle"], "output/openprism_atlas/latest")

    def test_valid_bundle_exposes_only_bounded_operator_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "latest"
            _write_bundle(root)
            payload = _application_for(root).atlas_payload()

        self.assertTrue(payload["available"])
        self.assertFalse(payload["synthetic"])
        self.assertEqual(payload["origin_status"], "captured_evidence")
        self.assertFalse(payload["meta"]["survey_grade"])
        self.assertEqual(payload["meta"]["mission_id"], "flight-042")
        self.assertEqual(payload["meta"]["width"], 7)
        self.assertEqual(payload["meta"]["height"], 5)
        self.assertEqual(payload["meta"]["grid"]["shape"], [5, 7])
        self.assertEqual(payload["meta"]["source_count"], 2)
        self.assertNotIn("private_operator_note", payload["meta"])
        self.assertNotIn("untrusted_extra", payload["meta"]["coordinate_reference"])
        self.assertEqual(set(payload["images"]), {"rgb", "thermal", "support"})
        self.assertTrue(
            all(value.startswith("data:image/png;base64,") for value in payload["images"].values())
        )
        self.assertFalse(payload["provenance"]["model_generated_pixels"])

    def test_synthetic_bundle_remains_visible_but_cannot_look_captured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "latest"
            _write_bundle(root)
            metadata_path = root / "atlas_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["mission_id"] = "SYNTHETIC-demo"
            metadata["camera_calibration_id"] = "SYNTHETIC:test"
            metadata.pop("data_provenance")
            metadata["demonstration"] = {
                "synthetic": True,
                "real_sensor_data": False,
                "real_gps_data": False,
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            payload = _application_for(root).atlas_payload()

        self.assertTrue(payload["available"])
        self.assertTrue(payload["synthetic"])
        self.assertEqual(payload["origin_status"], "synthetic_demo")
        self.assertEqual(payload["provenance"]["source"], "analytic_synthetic_atlas_demo")

    def test_missing_origin_declaration_is_unverified_not_captured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "latest"
            _write_bundle(root)
            metadata_path = root / "atlas_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("data_provenance")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            payload = _application_for(root).atlas_payload()

        self.assertTrue(payload["available"])
        self.assertFalse(payload["synthetic"])
        self.assertEqual(payload["origin_status"], "unverified")
        self.assertEqual(payload["provenance"]["source"], "unverified_atlas_bundle")

    def test_bundle_cannot_silently_claim_survey_grade(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "latest"
            _write_bundle(root)
            metadata_path = root / "atlas_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["survey_grade"] = True
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            payload = _application_for(root).atlas_payload()

        self.assertFalse(payload["available"])
        self.assertIn("survey_grade false", payload["reason"])

    def test_mismatched_preview_grids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "latest"
            _write_bundle(root)
            _write_preview(root / "atlas_support.png", (0, 255, 0), size=(6, 5))
            payload = _application_for(root).atlas_payload()

        self.assertFalse(payload["available"])
        self.assertIn("one pixel grid", payload["reason"])

    def test_transient_objects_and_stale_snapshot_state_cross_api_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "latest"
            _write_bundle(root)
            metadata_path = root / "atlas_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["publication"] = {
                "published_at_utc": "2000-01-01T00:00:00+00:00",
                "operator_freshness_ttl_s": 1.0,
                "temporal_role": "immutable_mission_snapshot",
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            (root / "atlas_objects.geojson").write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {
                                    "object_id": "track-7",
                                    "label": "person",
                                    "east_m": 2.5,
                                    "north_m": -1.25,
                                    "confidence": 0.9,
                                    "horizontal_uncertainty_m": 1.2,
                                    "timestamp_tai_ns": "1700000037000000000",
                                    "untrusted": "not exposed",
                                },
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [-118.1, 34.2],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = _application_for(root).atlas_payload()

        self.assertTrue(payload["available"])
        self.assertEqual(payload["meta"]["freshness_status"], "stale")
        self.assertEqual(payload["meta"]["object_count"], 1)
        self.assertEqual(payload["meta"]["track_count"], 1)
        self.assertEqual(payload["objects"][0]["label"], "person")
        self.assertNotIn("untrusted", payload["objects"][0])

    def test_future_dated_publication_is_flagged_instead_of_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "latest"
            _write_bundle(root)
            metadata_path = root / "atlas_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["publication"] = {
                "published_at_utc": "2999-01-01T00:00:00+00:00",
                "operator_freshness_ttl_s": 60.0,
                "temporal_role": "immutable_mission_snapshot",
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            application = _application_for(root)
            payload = application.atlas_payload()
            status = application.atlas_status_payload()

        self.assertTrue(payload["available"])
        self.assertEqual(payload["meta"]["freshness_status"], "future")
        self.assertLess(payload["meta"]["publication_age_s"], -5.0)
        self.assertEqual(
            payload["meta"]["future_publication_tolerance_s"], 5.0
        )
        self.assertTrue(status["available"])
        self.assertEqual(status["freshness_status"], "future")

    def test_lightweight_status_revision_changes_without_embedding_previews(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "latest"
            _write_bundle(root)
            application = _application_for(root)
            first = application.atlas_status_payload()
            metadata_path = root / "atlas_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["accepted_capture_count"] += 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            second = application.atlas_status_payload()

        self.assertTrue(first["available"])
        self.assertTrue(second["available"])
        self.assertNotEqual(first["revision_id"], second["revision_id"])
        self.assertNotIn("images", first)
        self.assertNotIn("images", second)


class AtlasOperatorStaticUiTests(unittest.TestCase):
    def test_atlas_is_a_fifth_non_synthetic_view(self) -> None:
        index = (REPO_ROOT / "openprism" / "web" / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "openprism" / "web" / "app.js").read_text(encoding="utf-8")
        server = (REPO_ROOT / "openprism" / "server.py").read_text(encoding="utf-8")

        self.assertIn('data-preset="atlas"', index)
        self.assertIn("TACTICAL 2.5D", index)
        self.assertIn("not survey-grade", index)
        self.assertIn('fetch("/api/atlas"', script)
        self.assertIn("No map is fabricated", script)
        self.assertIn("SYNTHETIC DEMO · NO REAL SENSOR OR GPS DATA", script)
        self.assertIn("UNVERIFIED DATA ORIGIN", script)
        self.assertIn('fetch("/api/atlas/status"', script)
        self.assertIn("automatic_control", script)
        self.assertIn("updateAIAdvisor", script)
        self.assertIn('id="autoFusionToggle"', index)
        self.assertIn('id="aiSummary"', index)
        self.assertIn("ATLAS_STATUS_POLL_MS", script)
        self.assertIn("atlasFreshness(meta)", script)
        self.assertIn("FUTURE-DATED ATLAS · CLOCKS INVALID", script)
        self.assertIn("BigInt(text)", script)
        self.assertIn("ATLAS_TRACK_MAX_GAP_NS", script)
        self.assertIn("track.sort(compareAtlasObservationTime)", script)
        self.assertIn("atlasTrackPointsAreContinuous(previous, item)", script)
        self.assertIn('parsed.path == "/api/atlas"', server)
        self.assertIn('parsed.path == "/api/atlas/status"', server)
        self.assertIn('parsed.path == "/api/ai/context"', server)


if __name__ == "__main__":
    unittest.main()
