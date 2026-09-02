from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from openprism.pixhawk import CameraPoseRecord
from openprism.survey import (
    DockerUnavailableError,
    ImagePoseMatchError,
    ImmutableProjectError,
    OverlapPrerequisiteError,
    RoughOverlapContract,
    SurveyDatumContract,
    SurveyExecutionError,
    SurveyPreparationConfig,
    SurveyPreparationError,
    execute_odm_project,
    load_camera_pose_records,
    prepare_odm_survey_project,
)


def pose(name: str, index: int, *, latitude_offset_deg: float | None = None) -> CameraPoseRecord:
    offset = index * 0.00002 if latitude_offset_deg is None else latitude_offset_deg
    return CameraPoseRecord(
        image_name=name,
        image_index=index,
        latitude_deg=34.0 + offset,
        longitude_deg=-118.0,
        altitude_msl_m=155.0,
        relative_altitude_m=50.0,
        quaternion_camera_flu_to_enu_wxyz=(0.5, -0.5, 0.5, 0.5),
        quaternion_camera_optical_to_enu_wxyz=(0.0, 1.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=-90.0,
        roll_deg=0.0,
        capture_monotonic_ns=None,
        capture_utc_ns=1_750_000_000_000_000_000 + index * 1_000_000_000,
        event_monotonic_ns=None,
        event_utc_ns=1_750_000_000_000_000_000 + index * 1_000_000_000,
        clock_domain="test:UTC",
        time_basis="utc",
        time_uncertainty_ns=100_000,
        horizontal_accuracy_m=0.03,
        vertical_accuracy_m=0.05,
        attitude_accuracy_deg=0.02,
        fix_type=6,
        fix_quality="ignored_and_canonicalized",
        rtk_status="ignored_and_canonicalized",
        source_message="TEST_CAPTURE",
        position_source="test_camera_center",
        attitude_source="test_nadir_calibration",
        interpolation_span_ns=None,
        relative_altitude_reference="ground",
        position_reference="camera_optical_center",
        input_attitude_profile="test_calibrated_nadir_camera",
        image_match_basis="exact_test_name",
    )


def config(**overrides: object) -> SurveyPreparationConfig:
    values: dict[str, object] = {
        "project_name": "mission-alpha",
        "datum": SurveyDatumContract(
            horizontal_crs_id="EPSG:4326",
            vertical_datum_id="TEST:orthometric-MSL-v1",
            geoid_model_id="TEST:geoid-grid-v3",
            geoid_model_sha256="a" * 64,
        ),
        "overlap": RoughOverlapContract(
            nominal_agl_m=50.0,
            horizontal_fov_deg=60.0,
            vertical_fov_deg=50.0,
            planned_forward_overlap_fraction=0.80,
            planned_side_overlap_fraction=0.70,
        ),
        "minimum_image_count": 8,
        "require_position_accuracy": True,
        "require_camera_optical_center": True,
        "odm_image": "opendronemap/odm@sha256:" + "b" * 64,
        "odm_options": ("--dsm",),
    }
    values.update(overrides)
    return SurveyPreparationConfig(**values)  # type: ignore[arg-type]


class SurveyPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "projects"
        self.source.mkdir()
        self.images: list[Path] = []
        self.records: list[CameraPoseRecord] = []
        for index in range(8):
            image = self.source / f"frame_{index:03d}.jpg"
            Image.new("RGB", (12, 8), (index * 20, 80, 120)).save(image)
            self.images.append(image)
            self.records.append(pose(image.name, index))

    def tearDown(self) -> None:
        # Prepared inputs are intentionally read-only. Restore write permission
        # so TemporaryDirectory can remove them on Windows as well as POSIX.
        for path in self.root.rglob("*"):
            if path.is_file():
                path.chmod(path.stat().st_mode | stat.S_IWRITE)
        self.temporary.cleanup()

    def test_stages_deterministic_immutable_inputs_and_honest_plan(self) -> None:
        with patch("openprism.survey.subprocess.run") as process:
            result = prepare_odm_survey_project(
                self.images, self.records, self.output, config()
            )
        process.assert_not_called()
        self.assertFalse(result.odm_executed)
        self.assertTrue(result.project_directory.is_dir())
        self.assertEqual(result.project_directory.name, result.project_id)
        self.assertIn(str(self.output.resolve()), result.docker_argv[4])
        self.assertIn("--geo", result.docker_argv)
        self.assertEqual(result.docker_argv[0:3], ("docker", "run", "--rm"))

        for source in self.images:
            staged = result.images_directory / source.name
            self.assertEqual(staged.read_bytes(), source.read_bytes())
            self.assertEqual(staged.stat().st_mode & stat.S_IWUSR, 0)

        geo_lines = result.geo_txt.read_text(encoding="utf-8").splitlines()
        self.assertEqual(geo_lines[0], "EPSG:4326")
        self.assertEqual(len(geo_lines), 9)
        self.assertTrue(all(len(line.split()) == 4 for line in geo_lines[1:]))

        plan = json.loads(result.plan_json.read_text(encoding="utf-8"))
        self.assertFalse(plan["geo_txt_contract"]["orientation_serialized"])
        self.assertFalse(plan["geo_txt_contract"]["accuracy_serialized"])
        self.assertIn("GPS proximity does not prove", " ".join(
            plan["rough_overlap_gate"]["limitations"]
        ))
        self.assertIn("not terrain depth", " ".join(plan["limitations"]))
        self.assertEqual(plan["docker"]["argv"], list(result.docker_argv))
        self.assertTrue(plan["docker"]["image_is_digest_pinned"])

        manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
        self.assertFalse(manifest["claims"]["terrain_reconstructed"])
        self.assertFalse(manifest["claims"]["gps_only_mapping"])
        self.assertFalse(manifest["claims"]["survey_grade"])
        for entry in manifest["files"]:
            path = result.project_directory / entry["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])

        repeated = prepare_odm_survey_project(
            reversed(self.images), reversed(self.records), self.output, config()
        )
        self.assertTrue(repeated.reused_existing)
        self.assertEqual(repeated.project_id, result.project_id)

    def test_rejects_ambiguous_or_non_basename_image_matching(self) -> None:
        duplicate_directory = self.root / "other"
        duplicate_directory.mkdir()
        duplicate = duplicate_directory / self.images[0].name
        Image.new("RGB", (12, 8), (255, 0, 0)).save(duplicate)
        with self.assertRaisesRegex(ImagePoseMatchError, "basenames must be unique"):
            prepare_odm_survey_project(
                [*self.images, duplicate],
                self.records,
                self.output,
                config(minimum_image_count=3),
            )

    def test_rejects_non_image_and_non_rgb_inputs(self) -> None:
        broken = self.root / "broken.jpg"
        broken.write_bytes(b"not an image")
        with self.assertRaisesRegex(ImagePoseMatchError, "not a decodable image"):
            prepare_odm_survey_project(
                [broken, *self.images],
                [pose(broken.name, 99), *self.records],
                self.output,
                config(),
            )

        grayscale = self.root / "gray.png"
        Image.new("L", (12, 8), 127).save(grayscale)
        with self.assertRaisesRegex(ImagePoseMatchError, "mode 'L'"):
            prepare_odm_survey_project(
                [grayscale, *self.images],
                [pose(grayscale.name, 99), *self.records],
                self.output,
                config(),
            )

        unsafe = list(self.records)
        unsafe[0] = pose("nested/frame_000.jpg", 0)
        with self.assertRaisesRegex(ImagePoseMatchError, "no directory component"):
            prepare_odm_survey_project(
                self.images,
                unsafe,
                self.output,
                config(),
            )

    def test_rejects_missing_extra_case_mismatch_and_duplicate_indexes(self) -> None:
        with self.assertRaisesRegex(ImagePoseMatchError, "exactly one-to-one"):
            prepare_odm_survey_project(
                self.images,
                self.records[:-1],
                self.output,
                config(),
            )

        wrong_case = list(self.records)
        wrong_case[0] = pose("FRAME_000.JPG", 0)
        with self.assertRaisesRegex(ImagePoseMatchError, "case must match exactly"):
            prepare_odm_survey_project(
                self.images,
                wrong_case,
                self.output,
                config(),
            )

        duplicate_index = list(self.records)
        duplicate_index[-1] = pose("frame_007.jpg", 0)
        with self.assertRaisesRegex(ImagePoseMatchError, "duplicate non-null image_index"):
            prepare_odm_survey_project(
                self.images,
                duplicate_index,
                self.output,
                config(),
            )

    def test_gates_count_declared_overlap_spacing_and_stationary_capture(self) -> None:
        with self.assertRaisesRegex(OverlapPrerequisiteError, "at least 8"):
            prepare_odm_survey_project(
                self.images[:7], self.records[:7], self.output, config()
            )
        with self.assertRaisesRegex(OverlapPrerequisiteError, "forward overlap"):
            RoughOverlapContract(
                nominal_agl_m=50,
                horizontal_fov_deg=60,
                vertical_fov_deg=50,
                planned_forward_overlap_fraction=0.50,
                planned_side_overlap_fraction=0.70,
            )

        sparse = [pose(path.name, index, latitude_offset_deg=index * 0.001) for index, path in enumerate(self.images)]
        with self.assertRaisesRegex(OverlapPrerequisiteError, "too sparse"):
            prepare_odm_survey_project(self.images, sparse, self.output, config())

        stationary = [pose(path.name, index, latitude_offset_deg=0.0) for index, path in enumerate(self.images)]
        with self.assertRaisesRegex(OverlapPrerequisiteError, "spatial extent"):
            prepare_odm_survey_project(self.images, stationary, self.output, config())

    def test_requires_explicit_datum_and_geoid_identity(self) -> None:
        with self.assertRaisesRegex(SurveyPreparationError, "vertical_datum_id"):
            SurveyDatumContract("EPSG:4326", "unknown", "EGM2008")
        with self.assertRaisesRegex(SurveyPreparationError, "geoid_model_id"):
            SurveyDatumContract("EPSG:4326", "MSL:test", "unspecified")
        with self.assertRaisesRegex(SurveyPreparationError, "requires horizontal"):
            SurveyDatumContract("EPSG:32611", "MSL:test", "EGM2008")

    def test_rejects_missing_accuracy_and_non_camera_center_when_required(self) -> None:
        values = self.records[0].__class__(
            **{
                **{field: getattr(self.records[0], field) for field in self.records[0].__dataclass_fields__},
                "horizontal_accuracy_m": None,
            }
        )
        missing_accuracy = [values, *self.records[1:]]
        with self.assertRaisesRegex(SurveyPreparationError, "position accuracy"):
            prepare_odm_survey_project(self.images, missing_accuracy, self.output, config())

        vehicle = self.records[0].__class__(
            **{
                **{field: getattr(self.records[0], field) for field in self.records[0].__dataclass_fields__},
                "position_reference": "vehicle_navigation_origin",
            }
        )
        with self.assertRaisesRegex(SurveyPreparationError, "optical-center"):
            prepare_odm_survey_project(
                self.images,
                [vehicle, *self.records[1:]],
                self.output,
                config(),
            )

    def test_tampered_content_addressed_project_is_not_reused(self) -> None:
        result = prepare_odm_survey_project(
            self.images, self.records, self.output, config()
        )
        staged = result.images_directory / self.images[0].name
        staged.chmod(staged.stat().st_mode | stat.S_IWRITE)
        staged.write_bytes(b"tampered")
        with self.assertRaisesRegex(ImmutableProjectError, "hash mismatch"):
            prepare_odm_survey_project(
                self.images, self.records, self.output, config()
            )

        # Rewriting the untrusted manifest to bless altered staged bytes does
        # not defeat the source-derived content contract.
        manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
        relative = f"images/{self.images[0].name}"
        for entry in manifest["files"]:
            if entry["path"] == relative:
                entry["sha256"] = hashlib.sha256(staged.read_bytes()).hexdigest()
                entry["size_bytes"] = staged.stat().st_size
                break
        result.manifest_json.chmod(result.manifest_json.stat().st_mode | stat.S_IWRITE)
        result.manifest_json.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with self.assertRaisesRegex(ImmutableProjectError, "source contract"):
            prepare_odm_survey_project(
                self.images, self.records, self.output, config()
            )

    def test_pose_json_round_trip_preserves_records(self) -> None:
        source = self.root / "poses.json"
        source.write_text(
            json.dumps({"records": [record.as_dict() for record in self.records]}),
            encoding="utf-8",
        )
        loaded = load_camera_pose_records(source)
        self.assertEqual(loaded, tuple(self.records))


class SurveyExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        self.images = []
        self.records = []
        for index in range(8):
            image = source / f"frame_{index:03d}.jpg"
            Image.new("RGB", (12, 8), (index * 20, 80, 120)).save(image)
            self.images.append(image)
            self.records.append(pose(image.name, index))
        self.result = prepare_odm_survey_project(
            self.images, self.records, self.root / "projects", config()
        )

    def tearDown(self) -> None:
        for path in self.root.rglob("*"):
            if path.is_file():
                path.chmod(path.stat().st_mode | stat.S_IWRITE)
        self.temporary.cleanup()

    def test_execution_requires_docker_and_explicit_call(self) -> None:
        with patch("openprism.survey.shutil.which", return_value=None), patch(
            "openprism.survey.subprocess.run"
        ) as process:
            with self.assertRaises(DockerUnavailableError):
                execute_odm_project(self.result)
        process.assert_not_called()

    def test_explicit_execution_uses_argv_without_shell_and_checks_exit(self) -> None:
        with patch("openprism.survey.shutil.which", return_value="docker.exe"), patch(
            "openprism.survey.subprocess.run",
            return_value=SimpleNamespace(returncode=0),
        ) as process:
            executed = execute_odm_project(self.result)
        self.assertTrue(executed.odm_executed)
        self.assertEqual(executed.odm_returncode, 0)
        process.assert_called_once_with(
            list(self.result.docker_argv),
            cwd=self.result.project_directory.parent,
            check=False,
        )

        with patch("openprism.survey.shutil.which", return_value="docker.exe"), patch(
            "openprism.survey.subprocess.run",
            return_value=SimpleNamespace(returncode=7),
        ):
            with self.assertRaisesRegex(SurveyExecutionError, "exit code 7"):
                execute_odm_project(self.result)


if __name__ == "__main__":
    unittest.main()
