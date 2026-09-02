from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from openprism.mapping import (
    CameraIntrinsics,
    GeodeticCoordinate,
    LocalENUFrame,
    MappingFrame,
    OrthoMosaicConfig,
    OrthoMosaicGrid,
    Quaternion,
    VehicleCameraPose,
    enu_to_geodetic,
    geodetic_to_enu,
    intersect_ground_plane,
    intersect_height_field,
    project_pixels_to_height_field,
    project_pixels_to_ground,
    projected_ground_uncertainty,
    sample_surface_elevation,
)


def _camera_rotations_nadir() -> tuple[Quaternion, Quaternion]:
    # Level FRD vehicle: body x -> north, y -> east, z -> down.
    enu_from_body = Quaternion.from_rotation_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
    )
    # Rectified camera: image right -> body right, image down -> body aft,
    # optical axis -> body down. The resulting image is north-up on the ground.
    body_from_camera = Quaternion.from_rotation_matrix(
        np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    return enu_from_body, body_from_camera


def _pose(
    origin: GeodeticCoordinate,
    *,
    timestamp: int | None = 1_000_000_000,
    pose_id: str | None = "pixhawk-ekf/1",
    attitude_sigma: float = 0.0,
    upward: bool = False,
) -> VehicleCameraPose:
    if upward:
        enu_from_body = Quaternion.identity()
        body_from_camera = Quaternion.identity()
    else:
        enu_from_body, body_from_camera = _camera_rotations_nadir()
    return VehicleCameraPose(
        position=GeodeticCoordinate(
            origin.latitude_deg,
            origin.longitude_deg,
            origin.ellipsoid_height_m + 10.0,
        ),
        enu_from_body=enu_from_body,
        body_from_camera=body_from_camera,
        timestamp_tai_ns=timestamp,
        pose_id=pose_id,
        position_sigma_enu_m=(0.0, 0.0, 0.0),
        attitude_sigma_rad=attitude_sigma,
    )


def _intrinsics(
    width: int = 1,
    height: int = 1,
    *,
    calibration_id: str | None = "camera/calibration/1",
) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx_px=10.0 if width > 1 else 1.0,
        fy_px=10.0 if height > 1 else 1.0,
        cx_px=(width - 1) / 2.0,
        cy_px=(height - 1) / 2.0,
        calibration_id=calibration_id,
    )


def _single_pixel_frame(
    origin: GeodeticCoordinate,
    frame_id: str,
    *,
    rgb: tuple[int, int, int] = (255, 0, 0),
    thermal: float = 0.2,
    semantic: int = 2,
    support: float = 1.0,
    timestamp: int | None = 1_000_000_000,
    intrinsics: CameraIntrinsics | None = None,
    pose: VehicleCameraPose | None = None,
) -> MappingFrame:
    return MappingFrame(
        frame_id=frame_id,
        timestamp_tai_ns=timestamp,
        rgb=np.asarray([[rgb]], dtype=np.uint8),
        intrinsics=_intrinsics() if intrinsics is None else intrinsics,
        pose=_pose(origin) if pose is None else pose,
        thermal_normalized=np.asarray([[thermal]], dtype=np.float32),
        semantic_class=np.asarray([[semantic]], dtype=np.int32),
        support=support,
        provenance={"source": "synthetic", "nested": {"immutable": True}},
    )


class GeodesyTests(unittest.TestCase):
    def test_wgs84_enu_axes_and_round_trip(self) -> None:
        origin = GeodeticCoordinate(0.0, 0.0, 25.0)
        local = LocalENUFrame(origin, frame_id="test-enu")

        east = geodetic_to_enu(GeodeticCoordinate(0.0, 0.001, 25.0), local)
        north = geodetic_to_enu(GeodeticCoordinate(0.001, 0.0, 25.0), local)
        up = geodetic_to_enu(GeodeticCoordinate(0.0, 0.0, 35.0), local)

        self.assertAlmostEqual(east[0], 111.3199, places=3)
        self.assertAlmostEqual(east[1], 0.0, places=6)
        self.assertAlmostEqual(north[1], 110.5747, places=3)
        self.assertAlmostEqual(north[0], 0.0, places=6)
        np.testing.assert_allclose(up, [0.0, 0.0, 10.0], atol=1e-8)

        target = GeodeticCoordinate(0.00073, -0.00042, 42.25)
        reconstructed = enu_to_geodetic(geodetic_to_enu(target, local), local)
        self.assertAlmostEqual(reconstructed.latitude_deg, target.latitude_deg, places=10)
        self.assertAlmostEqual(reconstructed.longitude_deg, target.longitude_deg, places=10)
        self.assertAlmostEqual(
            reconstructed.ellipsoid_height_m, target.ellipsoid_height_m, places=5
        )

    def test_contracts_and_snapshot_arrays_are_immutable(self) -> None:
        coordinate = GeodeticCoordinate(37.0, -122.0, 100.0)
        with self.assertRaises(FrozenInstanceError):
            coordinate.latitude_deg = 0.0  # type: ignore[misc]

        grid = OrthoMosaicGrid(
            LocalENUFrame(coordinate),
            OrthoMosaicConfig(-0.5, 0.5, -0.5, 0.5, 1.0),
        )
        self.assertTrue(grid.integrate(_single_pixel_frame(coordinate, "f0")).accepted)
        snapshot = grid.snapshot()
        with self.assertRaises(ValueError):
            snapshot.rgb[0, 0, 0] = 0.0
        with self.assertRaises(TypeError):
            snapshot.metadata["changed"] = True  # type: ignore[index]
        with self.assertRaises(TypeError):
            snapshot.provenance["f0"]["source"] = "changed"  # type: ignore[index]

    def test_camera_intrinsics_reject_fractional_dimensions_and_blank_calibration(self) -> None:
        with self.assertRaises(ValueError):
            CameraIntrinsics(1.9, 2, 3.0, 4.0, 0.0, 0.0, "cal")
        with self.assertRaises(ValueError):
            CameraIntrinsics(2, 2, 3.0, 4.0, 0.0, 0.0, "   ")

        intrinsics = CameraIntrinsics(2, 2, "3.0", "4.0", "0", "0", "cal")
        self.assertIsInstance(intrinsics.fx_px, float)
        self.assertAlmostEqual(intrinsics.focal_mean_px, np.sqrt(12.0))


class RotationAndProjectionTests(unittest.TestCase):
    def test_hamilton_quaternion_is_active_source_to_destination(self) -> None:
        rotation = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
        np.testing.assert_allclose(rotation.rotate(np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-12)

        reconstructed = Quaternion.from_rotation_matrix(rotation.as_rotation_matrix())
        np.testing.assert_allclose(
            reconstructed.as_rotation_matrix(), rotation.as_rotation_matrix(), atol=1e-12
        )

    def test_nadir_pinhole_projection_has_expected_ground_footprint(self) -> None:
        origin = GeodeticCoordinate(37.0, -122.0, 100.0)
        local = LocalENUFrame(origin)
        pixels = np.array(
            [
                [1.0, 1.0],  # principal point
                [0.0, 0.0],  # west, north
                [2.0, 2.0],  # east, south
            ]
        )
        projection = project_pixels_to_ground(
            pixels,
            _intrinsics(3, 3),
            _pose(origin),
            local,
            ground_elevation_enu_m=0.0,
        )
        self.assertTrue(np.all(projection.valid))
        np.testing.assert_allclose(projection.points_enu_m[0], [0.0, 0.0, 0.0], atol=2e-7)
        np.testing.assert_allclose(projection.points_enu_m[1], [-1.0, 1.0, 0.0], atol=2e-6)
        np.testing.assert_allclose(projection.points_enu_m[2], [1.0, -1.0, 0.0], atol=2e-6)

    def test_ground_intersection_rejects_upward_horizon_and_excess_range(self) -> None:
        projection = intersect_ground_plane(
            np.array([0.0, 0.0, 10.0]),
            np.array(
                [
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, -0.01],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, -0.2],
                ]
            ),
            min_downward_cosine=0.05,
            max_slant_range_m=40.0,
        )
        np.testing.assert_array_equal(projection.valid, [True, False, False, False])
        np.testing.assert_allclose(projection.points_enu_m[0], [0.0, 0.0, 0.0])
        self.assertTrue(np.all(np.isnan(projection.points_enu_m[1:])))

    def test_height_field_intersection_preserves_terrain_elevation(self) -> None:
        origin = GeodeticCoordinate(37.0, -122.0, 100.0)
        local = LocalENUFrame(origin)
        east_centres = np.arange(-2.0, 3.0)
        surface = np.tile(1.0 + 0.2 * east_centres, (5, 1))
        config = OrthoMosaicConfig(
            -2.5,
            2.5,
            -2.5,
            2.5,
            1.0,
            surface_elevation_enu_m=surface,
            surface_model_id="synthetic-slope-v1",
        )
        projection = project_pixels_to_height_field(
            np.array([[1.0, 1.0], [2.0, 1.0]]),
            _intrinsics(3, 3),
            _pose(origin),
            local,
            config,
        )
        self.assertTrue(np.all(projection.valid))
        self.assertAlmostEqual(float(projection.points_enu_m[0, 2]), 1.0, places=5)
        expected_east = float(projection.points_enu_m[1, 0])
        self.assertAlmostEqual(
            float(projection.points_enu_m[1, 2]),
            1.0 + 0.2 * expected_east,
            places=3,
        )

    def test_height_field_works_when_camera_is_below_global_dem_median(self) -> None:
        surface = np.full((5, 5), 20.0, dtype=np.float64)
        surface[2, 2] = 0.0
        config = OrthoMosaicConfig(
            -2.5,
            2.5,
            -2.5,
            2.5,
            1.0,
            surface_elevation_enu_m=surface,
            surface_model_id="valley-dem",
        )
        projection = intersect_height_field(
            np.array([0.0, 0.0, 10.0]),
            np.array([[0.0, 0.0, -1.0]]),
            config,
        )
        self.assertTrue(bool(projection.valid[0]))
        self.assertAlmostEqual(float(projection.points_enu_m[0, 2]), 0.0, places=4)

    def test_height_field_returns_first_visible_surface_crossing(self) -> None:
        surface = np.zeros((1, 10), dtype=np.float64)
        surface[0, 4] = 6.0
        surface[0, 7] = 8.0
        config = OrthoMosaicConfig(
            -4.5,
            5.5,
            -0.5,
            0.5,
            1.0,
            surface_elevation_enu_m=surface,
            surface_model_id="two-ridges",
            surface_ray_step_m=0.25,
        )
        projection = intersect_height_field(
            np.array([-4.0, 0.0, 10.0]),
            np.array([[1.0, 0.0, -1.0]]),
            config,
        )
        self.assertTrue(bool(projection.valid[0]))
        self.assertLess(float(projection.points_enu_m[0, 0]), 2.0)

    def test_ridge_visibility_transition_is_not_falsely_precise(self) -> None:
        # The nominal 45-degree ray is tangent to the near ridge at east=4 m.
        # A -0.5 degree perturbation misses it and first hits the farther ridge.
        # The planar estimate is ~5 cm, but the terrain-aware bound must expose
        # the multi-metre first-visible-surface branch change.
        surface = np.zeros((21, 10), dtype=np.float64)
        surface[:, 4] = 6.0
        surface[:, 7] = 8.0
        config = OrthoMosaicConfig(
            -0.5,
            9.5,
            -10.5,
            10.5,
            1.0,
            surface_elevation_enu_m=surface,
            surface_model_id="tangent-two-ridge-dem",
            surface_ray_step_m=0.1,
            min_downward_cosine=0.05,
        )
        projection = intersect_height_field(
            np.array([0.0, 0.0, 10.0]),
            np.array([[1.0, 0.0, -1.0]]),
            config,
        )
        self.assertTrue(bool(projection.valid[0]))
        self.assertAlmostEqual(float(projection.points_enu_m[0, 0]), 4.0, places=3)

        origin = GeodeticCoordinate(37.0, -122.0, 100.0)
        pose = _pose(origin, attitude_sigma=np.deg2rad(0.5))
        planar = projected_ground_uncertainty(
            projection,
            _intrinsics(),
            pose,
            frame_timestamp_tai_ns=pose.timestamp_tai_ns,
            ground_elevation_enu_m=projection.points_enu_m[:, 2],
        )
        terrain_aware = projected_ground_uncertainty(
            projection,
            _intrinsics(),
            pose,
            frame_timestamp_tai_ns=pose.timestamp_tai_ns,
            ground_elevation_enu_m=projection.points_enu_m[:, 2],
            config=config,
        )

        self.assertLess(float(planar[0]), 0.06)
        self.assertGreater(float(terrain_aware[0]), 2.0)

    def test_smooth_dem_uncertainty_remains_finite_and_uses_surface_sigma(self) -> None:
        surface = np.zeros((21, 21), dtype=np.float64)
        config = OrthoMosaicConfig(
            -0.5,
            20.5,
            -10.5,
            10.5,
            1.0,
            surface_elevation_enu_m=surface,
            surface_elevation_sigma_m=0.25,
            surface_model_id="smooth-flat-dem",
            surface_ray_step_m=0.1,
            min_downward_cosine=0.05,
        )
        projection = intersect_height_field(
            np.array([0.0, 0.0, 10.0]),
            np.array([[0.5, 0.0, -1.0]]),
            config,
        )
        origin = GeodeticCoordinate(37.0, -122.0, 100.0)
        pose = _pose(origin, attitude_sigma=np.deg2rad(0.5))
        without_dem_sigma = projected_ground_uncertainty(
            projection,
            _intrinsics(),
            pose,
            frame_timestamp_tai_ns=pose.timestamp_tai_ns,
            ground_elevation_enu_m=projection.points_enu_m[:, 2],
        )
        terrain_aware = projected_ground_uncertainty(
            projection,
            _intrinsics(),
            pose,
            frame_timestamp_tai_ns=pose.timestamp_tai_ns,
            ground_elevation_enu_m=projection.points_enu_m[:, 2],
            config=config,
        )

        self.assertTrue(np.isfinite(terrain_aware[0]))
        self.assertGreater(float(terrain_aware[0]), float(without_dem_sigma[0]))
        self.assertLess(float(terrain_aware[0]), 0.5)

    def test_public_surface_sampling_fails_closed_outside_dem(self) -> None:
        config = OrthoMosaicConfig(
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            surface_elevation_enu_m=np.array([[1.0, 2.0], [3.0, np.nan]]),
            surface_model_id="partial-dem",
        )
        elevation, valid = sample_surface_elevation(
            np.array([-0.5, 5.0, 0.5]),
            np.array([0.5, 0.5, -0.5]),
            config,
        )

        self.assertEqual(float(elevation[0]), 1.0)
        np.testing.assert_array_equal(valid, [True, False, False])
        self.assertTrue(np.all(np.isnan(elevation[1:])))
        with self.assertRaises(ValueError):
            elevation[0] = 99.0

    def test_height_field_never_projects_through_a_nodata_gap(self) -> None:
        surface = np.zeros((2, 10), dtype=np.float64)
        surface[:, 2:5] = np.nan
        config = OrthoMosaicConfig(
            0.0,
            10.0,
            0.0,
            2.0,
            1.0,
            surface_elevation_enu_m=surface,
            surface_model_id="dem:nodata-occluder-test",
            min_downward_cosine=0.1,
            max_slant_range_m=20.0,
        )
        projection = intersect_height_field(
            np.array([0.5, 1.0, 5.0]),
            np.array([[1.0, 0.0, -1.0]]),
            config,
        )
        self.assertFalse(bool(projection.valid[0]))
        self.assertTrue(np.all(np.isnan(projection.points_enu_m[0])))


class OrthoMosaicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = GeodeticCoordinate(37.0, -122.0, 100.0)
        self.local = LocalENUFrame(self.origin, frame_id="mission-enu")
        self.one_cell = OrthoMosaicConfig(
            east_min_m=-0.5,
            east_max_m=0.5,
            north_min_m=-0.5,
            north_max_m=0.5,
            resolution_m=1.0,
            max_projected_uncertainty_m=5.0,
        )

    def test_overlap_is_weighted_and_provenance_is_preserved(self) -> None:
        grid = OrthoMosaicGrid(self.local, self.one_cell)
        first = _single_pixel_frame(
            self.origin,
            "rgb-a",
            rgb=(255, 0, 0),
            thermal=0.2,
            support=1.0,
        )
        second = _single_pixel_frame(
            self.origin,
            "rgb-b",
            rgb=(0, 0, 255),
            thermal=1.0,
            support=0.25,
        )
        self.assertTrue(grid.integrate(first).accepted)
        self.assertTrue(grid.integrate(second).accepted)
        snapshot = grid.snapshot()

        np.testing.assert_allclose(snapshot.rgb[0, 0], [0.8, 0.0, 0.2], atol=1e-6)
        self.assertAlmostEqual(float(snapshot.thermal_normalized[0, 0]), 0.36, places=6)
        self.assertEqual(int(snapshot.coverage_count[0, 0]), 2)
        self.assertGreater(float(snapshot.support[0, 0]), 0.0)
        self.assertEqual(snapshot.source_ids, ("rgb-a", "rgb-b"))
        dominant = snapshot.source_ids[int(snapshot.dominant_source_index[0, 0])]
        self.assertEqual(dominant, "rgb-a")
        self.assertEqual(snapshot.provenance["rgb-a"]["calibration_id"], "camera/calibration/1")
        self.assertEqual(snapshot.metadata["coordinate_reference"]["axes"], "ENU")

        # A snapshot owns its storage and cannot be changed by later integration.
        frozen_rgb = snapshot.rgb.copy()
        grid.integrate(
            _single_pixel_frame(self.origin, "rgb-c", rgb=(0, 255, 0), support=1.0)
        )
        np.testing.assert_array_equal(snapshot.rgb, frozen_rgb)

    def test_cell_cap_is_enforced_before_grid_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeding max_cells=9999"):
            OrthoMosaicConfig(
                0.0,
                100.0,
                0.0,
                100.0,
                1.0,
                max_cells=9_999,
            )

    def test_peak_memory_budget_is_jointly_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "peak-memory estimate"):
            OrthoMosaicConfig(
                0.0,
                100.0,
                0.0,
                100.0,
                1.0,
                max_peak_memory_bytes=1_000_000,
            )

        config = OrthoMosaicConfig(
            -0.5,
            0.5,
            -0.5,
            0.5,
            1.0,
            max_peak_memory_bytes=580,
            max_sampled_pixels_per_frame=1,
        )
        grid = OrthoMosaicGrid(self.local, config)
        rejected = grid.integrate(
            _single_pixel_frame(self.origin, "memory-class", semantic=1)
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "semantic_memory_budget_exceeded")
        self.assertFalse(np.any(grid.snapshot().valid))

    def test_frame_work_and_semantic_dtype_are_fail_closed(self) -> None:
        config = OrthoMosaicConfig(
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            max_sampled_pixels_per_frame=1,
        )
        grid = OrthoMosaicGrid(self.local, config)
        frame = MappingFrame(
            frame_id="too-many-rays",
            timestamp_tai_ns=1_000_000_000,
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            intrinsics=_intrinsics(2, 2),
            pose=_pose(self.origin),
        )
        rejected = grid.integrate(frame)
        self.assertEqual(rejected.reason, "sampled_pixel_limit_exceeded")
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            MappingFrame(
                frame_id="float-semantics",
                timestamp_tai_ns=1,
                rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                intrinsics=_intrinsics(),
                pose=_pose(self.origin),
                semantic_class=np.array([[1.5]], dtype=np.float32),
            )

    def test_modality_weights_support_and_class_evidence_are_exported(self) -> None:
        grid = OrthoMosaicGrid(self.local, self.one_cell)
        self.assertTrue(
            grid.integrate(
                _single_pixel_frame(
                    self.origin,
                    "modalities",
                    thermal=0.7,
                    semantic=12,
                    support=0.5,
                )
            ).accepted
        )
        snapshot = grid.snapshot()
        layers = snapshot.array_layers

        self.assertIs(layers["rgb_weight_sum"], snapshot.weight_sum)
        self.assertIs(layers["rgb_support"], snapshot.support)
        np.testing.assert_allclose(layers["thermal_weight_sum"], snapshot.weight_sum)
        np.testing.assert_allclose(layers["semantic_weight_sum"], snapshot.weight_sum)
        np.testing.assert_allclose(layers["thermal_support"], snapshot.support)
        np.testing.assert_allclose(layers["semantic_support"], snapshot.support)
        np.testing.assert_allclose(
            layers["semantic_class_evidence_12"], snapshot.class_evidence[12]
        )
        with self.assertRaises(ValueError):
            layers["thermal_weight_sum"][0, 0] = 0.0

    def test_semantic_class_cap_rejects_frame_before_any_mutation(self) -> None:
        config = OrthoMosaicConfig(
            -0.5,
            0.5,
            -0.5,
            0.5,
            1.0,
            max_semantic_classes=1,
        )
        grid = OrthoMosaicGrid(self.local, config)
        self.assertTrue(
            grid.integrate(_single_pixel_frame(self.origin, "class-one", semantic=1)).accepted
        )
        before = grid.snapshot()
        rejected = grid.integrate(
            _single_pixel_frame(self.origin, "class-two", semantic=2)
        )
        after = grid.snapshot()

        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "semantic_class_limit_exceeded")
        np.testing.assert_array_equal(after.weight_sum, before.weight_sum)
        np.testing.assert_array_equal(after.rgb, before.rgb)
        self.assertEqual(after.source_ids, ("class-one",))
        self.assertEqual(tuple(after.class_evidence), (1,))

    def test_source_indexes_follow_stable_insertion_order(self) -> None:
        grid = OrthoMosaicGrid(self.local, self.one_cell)
        for frame_id in ("z-last-lexically", "a-first-lexically"):
            self.assertTrue(
                grid.integrate(_single_pixel_frame(self.origin, frame_id)).accepted
            )
        first = grid.snapshot()
        self.assertEqual(first.source_ids, ("z-last-lexically", "a-first-lexically"))

        self.assertTrue(
            grid.integrate(_single_pixel_frame(self.origin, "middle")).accepted
        )
        second = grid.snapshot()
        self.assertEqual(second.source_ids[:2], first.source_ids)
        self.assertEqual(second.source_ids[2], "middle")

    def test_semantics_use_accumulated_class_evidence_not_last_frame(self) -> None:
        grid = OrthoMosaicGrid(self.local, self.one_cell)
        frames = (
            _single_pixel_frame(self.origin, "class-4", semantic=4, support=0.8),
            _single_pixel_frame(self.origin, "class-7a", semantic=7, support=0.5),
            _single_pixel_frame(self.origin, "class-7b", semantic=7, support=0.5),
        )
        for frame in frames:
            self.assertTrue(grid.integrate(frame).accepted)
        snapshot = grid.snapshot()

        self.assertEqual(int(snapshot.semantic_class[0, 0]), 7)
        self.assertAlmostEqual(
            float(snapshot.semantic_confidence[0, 0]), 1.0 / 1.8, places=6
        )
        self.assertAlmostEqual(float(snapshot.class_evidence[4][0, 0]), 0.8, places=6)
        self.assertAlmostEqual(float(snapshot.class_evidence[7][0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(snapshot.height_enu_m[0, 0]), 0.0, places=6)

    def test_bilinear_splat_is_north_up_and_deterministic(self) -> None:
        config = OrthoMosaicConfig(-2.0, 2.0, -2.0, 2.0, 1.0)
        frame = MappingFrame(
            frame_id="footprint",
            timestamp_tai_ns=1_000_000_000,
            rgb=np.full((3, 3, 3), 127, dtype=np.uint8),
            intrinsics=_intrinsics(3, 3),
            pose=_pose(self.origin),
            semantic_class=np.arange(9, dtype=np.int32).reshape(3, 3),
        )
        first = OrthoMosaicGrid(self.local, config)
        second = OrthoMosaicGrid(self.local, config)
        self.assertTrue(first.integrate(frame).accepted)
        self.assertTrue(second.integrate(frame).accepted)
        a = first.snapshot()
        b = second.snapshot()

        np.testing.assert_array_equal(a.coverage_count, b.coverage_count)
        np.testing.assert_array_equal(a.weight_sum, b.weight_sum)
        np.testing.assert_array_equal(a.semantic_class, b.semantic_class)
        # The top-left image ray lands west and north, hence in the map's
        # north-west quadrant (small row and column indexes).
        self.assertGreater(int(a.coverage_count[0, 0]), 0)
        self.assertEqual(a.metadata["grid"]["north_up"], True)

    def test_required_evidence_and_uncertainty_are_hard_rejection_gates(self) -> None:
        grid = OrthoMosaicGrid(
            self.local,
            OrthoMosaicConfig(
                -0.5,
                0.5,
                -0.5,
                0.5,
                1.0,
                max_projected_uncertainty_m=1.0,
                max_pose_time_offset_ns=10,
            ),
        )

        unknown_time = _single_pixel_frame(self.origin, "unknown-time", timestamp=None)
        no_calibration = _single_pixel_frame(
            self.origin,
            "no-calibration",
            intrinsics=_intrinsics(calibration_id=None),
        )
        no_pose = MappingFrame(
            frame_id="no-pose",
            timestamp_tai_ns=1,
            rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            intrinsics=_intrinsics(),
            pose=None,
        )
        stale_pose = _single_pixel_frame(
            self.origin,
            "stale-pose",
            timestamp=1_000_000_100,
            pose=_pose(self.origin, timestamp=1_000_000_000),
        )
        uncertain = _single_pixel_frame(
            self.origin,
            "uncertain",
            pose=_pose(self.origin, attitude_sigma=0.2),
        )
        upward = _single_pixel_frame(
            self.origin,
            "upward",
            pose=_pose(self.origin, upward=True),
        )

        cases = (
            (unknown_time, "unknown_capture_time"),
            (no_calibration, "unknown_camera_calibration"),
            (no_pose, "unknown_vehicle_camera_pose"),
            (stale_pose, "pose_time_offset_exceeds_limit"),
            (uncertain, "projected_uncertainty_exceeds_limit"),
            (upward, "no_safe_ground_intersections"),
        )
        for frame, reason in cases:
            result = grid.integrate(frame)
            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, reason)

        snapshot = grid.snapshot()
        self.assertFalse(np.any(snapshot.valid))
        self.assertEqual(snapshot.metadata["accepted_frame_count"], 0)
        self.assertEqual(snapshot.metadata["rejected_frame_count"], len(cases))
        self.assertTrue(np.all(np.isnan(snapshot.rgb)))

    def test_duplicate_frame_id_is_rejected_without_double_counting(self) -> None:
        grid = OrthoMosaicGrid(self.local, self.one_cell)
        frame = _single_pixel_frame(self.origin, "same-id")
        self.assertTrue(grid.integrate(frame).accepted)
        duplicate = grid.integrate(frame)
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "duplicate_frame_id")
        self.assertEqual(int(grid.snapshot().coverage_count[0, 0]), 1)

    def test_concurrent_duplicate_delivery_commits_exactly_once(self) -> None:
        grid = OrthoMosaicGrid(self.local, self.one_cell)
        frame = _single_pixel_frame(self.origin, "concurrent-same-id")
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = tuple(executor.map(grid.integrate, (frame,) * 8))

        self.assertEqual(sum(result.accepted for result in results), 1)
        self.assertEqual(
            sum(result.reason == "duplicate_frame_id" for result in results), 7
        )
        snapshot = grid.snapshot()
        self.assertEqual(int(snapshot.coverage_count[0, 0]), 1)
        self.assertEqual(snapshot.source_ids, ("concurrent-same-id",))

    def test_height_field_is_integrated_as_a_real_2_5d_layer(self) -> None:
        surface = np.full((5, 5), 1.25, dtype=np.float64)
        config = OrthoMosaicConfig(
            -2.5,
            2.5,
            -2.5,
            2.5,
            1.0,
            surface_elevation_enu_m=surface,
            surface_model_id="dem:test:1",
        )
        grid = OrthoMosaicGrid(self.local, config)
        frame = MappingFrame(
            frame_id="height-field",
            timestamp_tai_ns=1_000_000_000,
            rgb=np.full((3, 3, 3), 127, dtype=np.uint8),
            intrinsics=_intrinsics(3, 3),
            pose=_pose(self.origin),
        )
        self.assertTrue(grid.integrate(frame).accepted)
        snapshot = grid.snapshot()
        self.assertTrue(np.any(snapshot.valid))
        np.testing.assert_allclose(snapshot.height_enu_m[snapshot.valid], 1.25, atol=0.05)
        self.assertEqual(snapshot.metadata["surface_model"]["kind"], "north_up_height_field")
        self.assertEqual(snapshot.metadata["surface_model"]["surface_model_id"], "dem:test:1")


if __name__ == "__main__":
    unittest.main()
