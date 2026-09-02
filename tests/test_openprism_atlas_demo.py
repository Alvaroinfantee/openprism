from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tooling.build_openprism_atlas_demo import (
    DYNAMIC_CLASS_ID,
    SYNTHETIC_WARNING,
    build_demo,
    flight_plan_enu,
    render_synthetic_frame,
)


class SyntheticAtlasDemoTests(unittest.TestCase):
    def test_render_is_deterministic_and_contains_a_masked_dynamic_object(self) -> None:
        east, north = flight_plan_enu()[0]
        first = render_synthetic_frame(0, east, north)
        second = render_synthetic_frame(0, east, north)

        np.testing.assert_array_equal(first.rgb, second.rgb)
        np.testing.assert_array_equal(first.thermal_normalized, second.thermal_normalized)
        np.testing.assert_array_equal(first.semantic_class, second.semantic_class)
        np.testing.assert_array_equal(first.dynamic_mask, second.dynamic_mask)
        self.assertGreater(np.count_nonzero(first.dynamic_mask), 0)
        self.assertTrue(
            np.all(first.semantic_class[first.dynamic_mask] == DYNAMIC_CLASS_ID)
        )
        self.assertIn("SYNTHETIC", first.record.source_message)
        self.assertEqual(first.record.position_source, "analytic_synthetic_flight_plan")

    def test_builds_labelled_bundle_and_excludes_dynamics_from_static_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "atlas"
            result = build_demo(output)

            self.assertEqual(result.accepted_capture_count, len(flight_plan_enu()))
            self.assertEqual(result.rejected_capture_count, 0)
            self.assertGreater(result.dynamic_pixel_count, 0)
            self.assertTrue(result.notice.is_file())
            self.assertIn(SYNTHETIC_WARNING, result.notice.read_text(encoding="utf-8"))

            manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertTrue(manifest["synthetic"])
            self.assertFalse(manifest["data_provenance"]["real_sensor_data"])
            self.assertFalse(manifest["data_provenance"]["real_gps_data"])
            self.assertFalse(manifest["data_provenance"]["caltech_imagery_used"])
            self.assertEqual(manifest["data_provenance"]["external_datasets_used"], [])
            self.assertTrue(manifest["dynamic_object"]["excluded_from_static_atlas"])

            metadata = json.loads(result.atlas.metadata.read_text(encoding="utf-8"))
            self.assertTrue(metadata["demonstration"]["synthetic"])
            self.assertEqual(metadata["demonstration"]["warning"], SYNTHETIC_WARNING)
            self.assertEqual(
                metadata["accepted_capture_count"],
                len(flight_plan_enu()),
            )
            self.assertEqual(
                metadata["geolocated_object_count"],
                len(flight_plan_enu()),
            )
            for source in metadata["source_provenance"].values():
                self.assertGreater(source["source"]["dynamic_pixels_excluded"], 0)
                self.assertFalse(source["source"]["caller"]["real_gps_data"])

            with np.load(result.atlas.arrays) as layers:
                valid = layers["valid"].astype(bool)
                semantic = layers["semantic_class"]
                thermal = layers["thermal_normalized"]
                self.assertGreater(np.count_nonzero(valid), 0)
                self.assertFalse(np.any(semantic[valid] == DYNAMIC_CLASS_ID))
                self.assertTrue(np.all(np.isfinite(thermal[valid])))

            exact_sources = sorted(result.source_frames.glob("*_layers.npz"))
            self.assertEqual(len(exact_sources), len(flight_plan_enu()))
            with np.load(exact_sources[0]) as source:
                source_dynamic = source["dynamic_mask"].astype(bool)
                self.assertGreater(np.count_nonzero(source_dynamic), 0)
                self.assertTrue(
                    np.all(
                        source["semantic_class"][source_dynamic]
                        == DYNAMIC_CLASS_ID
                    )
                )

            objects = json.loads(
                result.atlas.objects_geojson.read_text(encoding="utf-8")
            )
            self.assertEqual(len(objects["features"]), len(flight_plan_enu()))
            self.assertTrue(
                all(
                    feature["properties"]["label"] == "synthetic_vehicle"
                    for feature in objects["features"]
                )
            )

            with self.assertRaises(FileExistsError):
                build_demo(output)
            overwritten = build_demo(output, overwrite=True)
            self.assertEqual(
                overwritten.accepted_capture_count,
                len(flight_plan_enu()),
            )


if __name__ == "__main__":
    unittest.main()
