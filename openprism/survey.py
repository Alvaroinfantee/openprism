"""Deterministic post-flight handoff from OpenPRISM to OpenDroneMap.

This module prepares inputs for a real photogrammetric reconstruction.  It
does not reconstruct terrain itself: Pixhawk positions are written only as
camera-position priors, while depth, the point cloud, DSM, and orthomosaic
must be estimated by OpenDroneMap from overlapping image observations.

The prepared project is content addressed.  Source image bytes and the pose,
datum, overlap, and processing contracts determine its directory name.  Input
files are copied into that directory, hashed, and made read-only.  A later ODM
run may add derived products, but it cannot silently replace the recorded
input manifest without failing verification on the next preparation call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from .pixhawk import CameraPoseRecord, GEODETIC_POSITION_FRAME, export_odm_geo_txt


SURVEY_SCHEMA_VERSION = "openprism.survey-preparation/1.0"
SUPPORTED_RGB_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
_PLACEHOLDER_IDENTIFIERS = frozenset(
    {"", "unknown", "unspecified", "none", "n/a", "na", "tbd"}
)
_PROJECT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class SurveyPreparationError(ValueError):
    """Base class for a survey project that cannot be prepared safely."""


class ImagePoseMatchError(SurveyPreparationError):
    """The supplied image set and pose records are not an exact safe match."""


class OverlapPrerequisiteError(SurveyPreparationError):
    """The mission fails the declared, coarse image-overlap prerequisites."""


class ImmutableProjectError(SurveyPreparationError):
    """An existing content-addressed project no longer matches its manifest."""


class DockerUnavailableError(RuntimeError):
    """ODM execution was requested but the Docker executable is unavailable."""


class SurveyExecutionError(RuntimeError):
    """The explicitly requested ODM process returned a non-zero status."""


def _declared_identifier(name: str, value: Any) -> str:
    identifier = str(value).strip()
    if identifier.casefold() in _PLACEHOLDER_IDENTIFIERS:
        raise SurveyPreparationError(f"{name} must be an explicit, non-placeholder identity")
    if any(character in identifier for character in ("\0", "\r", "\n")):
        raise SurveyPreparationError(f"{name} contains a forbidden control character")
    return identifier


def _finite(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SurveyPreparationError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise SurveyPreparationError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class SurveyDatumContract:
    """Horizontal and vertical reference identities for the supplied poses.

    ``CameraPoseRecord`` stores longitude/latitude and MSL altitude.  The
    position-only ODM file therefore uses EPSG:4326 as its horizontal header,
    while the otherwise-unrepresentable vertical datum and geoid model are
    pinned here and in the project manifest.
    """

    horizontal_crs_id: str
    vertical_datum_id: str
    geoid_model_id: str
    geoid_model_sha256: str | None = None

    def __post_init__(self) -> None:
        horizontal = _declared_identifier("horizontal_crs_id", self.horizontal_crs_id)
        if horizontal.upper().replace(" ", "") not in {"EPSG:4326", "EPSG4326"}:
            raise SurveyPreparationError(
                "CameraPoseRecord contains WGS84 longitude/latitude; the survey "
                "handoff currently requires horizontal_crs_id='EPSG:4326'"
            )
        object.__setattr__(self, "horizontal_crs_id", "EPSG:4326")
        object.__setattr__(
            self,
            "vertical_datum_id",
            _declared_identifier("vertical_datum_id", self.vertical_datum_id),
        )
        object.__setattr__(
            self,
            "geoid_model_id",
            _declared_identifier("geoid_model_id", self.geoid_model_id),
        )
        if self.geoid_model_sha256 is not None:
            digest = str(self.geoid_model_sha256).strip().lower()
            if not _SHA256_PATTERN.fullmatch(digest):
                raise SurveyPreparationError(
                    "geoid_model_sha256 must be a 64-character lowercase SHA-256 digest"
                )
            object.__setattr__(self, "geoid_model_sha256", digest)


@dataclass(frozen=True, slots=True)
class RoughOverlapContract:
    """Declared mission geometry used for a coarse pre-SfM overlap gate.

    The proximity test is intentionally modest.  It can reject obviously
    sparse or stationary acquisitions, but it cannot prove feature overlap,
    cross-track lane spacing, parallax quality, sharpness, or reconstruction
    success.  ODM's matching and reconstruction diagnostics remain decisive.
    """

    nominal_agl_m: float
    horizontal_fov_deg: float
    vertical_fov_deg: float
    planned_forward_overlap_fraction: float
    planned_side_overlap_fraction: float
    minimum_forward_overlap_fraction: float = 0.70
    minimum_side_overlap_fraction: float = 0.60
    minimum_neighbor_pass_fraction: float = 0.80
    minimum_spatial_extent_fraction: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "nominal_agl_m",
            "horizontal_fov_deg",
            "vertical_fov_deg",
            "planned_forward_overlap_fraction",
            "planned_side_overlap_fraction",
            "minimum_forward_overlap_fraction",
            "minimum_side_overlap_fraction",
            "minimum_neighbor_pass_fraction",
            "minimum_spatial_extent_fraction",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.nominal_agl_m <= 0.0:
            raise SurveyPreparationError("nominal_agl_m must be positive")
        for name in ("horizontal_fov_deg", "vertical_fov_deg"):
            if not 0.0 < getattr(self, name) < 180.0:
                raise SurveyPreparationError(f"{name} must be in (0, 180)")
        for name in (
            "planned_forward_overlap_fraction",
            "planned_side_overlap_fraction",
            "minimum_forward_overlap_fraction",
            "minimum_side_overlap_fraction",
            "minimum_neighbor_pass_fraction",
            "minimum_spatial_extent_fraction",
        ):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise SurveyPreparationError(f"{name} must be in [0, 1)")
        if self.planned_forward_overlap_fraction < self.minimum_forward_overlap_fraction:
            raise OverlapPrerequisiteError(
                "declared forward overlap is below the configured prerequisite"
            )
        if self.planned_side_overlap_fraction < self.minimum_side_overlap_fraction:
            raise OverlapPrerequisiteError(
                "declared side overlap is below the configured prerequisite"
            )

    @property
    def footprint_width_m(self) -> float:
        return 2.0 * self.nominal_agl_m * math.tan(
            math.radians(self.horizontal_fov_deg) / 2.0
        )

    @property
    def footprint_height_m(self) -> float:
        return 2.0 * self.nominal_agl_m * math.tan(
            math.radians(self.vertical_fov_deg) / 2.0
        )


@dataclass(frozen=True, slots=True)
class SurveyPreparationConfig:
    """Policy and reproducibility contract for one ODM project."""

    project_name: str
    datum: SurveyDatumContract
    overlap: RoughOverlapContract
    minimum_image_count: int = 8
    require_position_accuracy: bool = True
    require_camera_optical_center: bool = False
    odm_image: str = "opendronemap/odm:latest"
    odm_options: tuple[str, ...] = ("--dsm",)

    def __post_init__(self) -> None:
        if not isinstance(self.datum, SurveyDatumContract):
            raise SurveyPreparationError("datum must be a SurveyDatumContract")
        if not isinstance(self.overlap, RoughOverlapContract):
            raise SurveyPreparationError("overlap must be a RoughOverlapContract")
        project_name = str(self.project_name).strip()
        if not _PROJECT_NAME_PATTERN.fullmatch(project_name) or project_name in {".", ".."}:
            raise SurveyPreparationError(
                "project_name must be 1-64 basename-safe letters, digits, '.', '_' or '-'"
            )
        object.__setattr__(self, "project_name", project_name)
        if (
            isinstance(self.minimum_image_count, bool)
            or not isinstance(self.minimum_image_count, int)
            or self.minimum_image_count < 3
        ):
            raise SurveyPreparationError("minimum_image_count must be an integer of at least 3")
        odm_image = str(self.odm_image).strip()
        if not odm_image or any(character.isspace() for character in odm_image):
            raise SurveyPreparationError("odm_image must be one Docker image reference")
        if any(character in odm_image for character in ("\0", "\r", "\n")):
            raise SurveyPreparationError("odm_image contains a forbidden control character")
        object.__setattr__(self, "odm_image", odm_image)
        options = tuple(str(option) for option in self.odm_options)
        forbidden = {"--project-path", "--geo"}
        if any(
            not option
            or any(character in option for character in ("\0", "\r", "\n"))
            for option in options
        ):
            raise SurveyPreparationError("ODM options must be non-empty argv tokens")
        if any(option.split("=", 1)[0] in forbidden for option in options):
            raise SurveyPreparationError(
                "odm_options may not override --project-path or --geo"
            )
        object.__setattr__(self, "odm_options", options)


@dataclass(frozen=True, slots=True)
class SurveyPreparationResult:
    """Prepared paths and the exact non-shell Docker invocation."""

    project_id: str
    project_directory: Path
    images_directory: Path
    geo_txt: Path
    pose_records_json: Path
    plan_json: Path
    manifest_json: Path
    docker_argv: tuple[str, ...]
    docker_command_powershell: str
    reused_existing: bool
    odm_executed: bool = False
    odm_returncode: int | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_document_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_document_bytes(value))


def _safe_basename(value: str, *, label: str) -> str:
    name = str(value).strip()
    if (
        not name
        or name in {".", ".."}
        or PurePosixPath(name).name != name
        or PureWindowsPath(name).name != name
        or any(character.isspace() for character in name)
        or any(character in name for character in ("\0", "\r", "\n"))
    ):
        raise ImagePoseMatchError(
            f"{label} must be a non-whitespace basename with no directory component: {value!r}"
        )
    return name


def _normalized_images(images: Iterable[str | Path]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    names: dict[str, Path] = {}
    sources: set[Path] = set()
    for supplied in images:
        path = Path(supplied).expanduser().resolve()
        if not path.is_file():
            raise ImagePoseMatchError(f"RGB capture is not a regular file: {path}")
        name = _safe_basename(path.name, label="RGB capture name")
        if path.suffix.casefold() not in SUPPORTED_RGB_SUFFIXES:
            raise ImagePoseMatchError(
                f"unsupported RGB capture suffix for {name}; expected one of "
                + ", ".join(sorted(SUPPORTED_RGB_SUFFIXES))
            )
        if path.stat().st_size <= 0:
            raise ImagePoseMatchError(f"RGB capture is empty: {path}")
        _rgb_image_metadata(path)
        folded = name.casefold()
        if folded in names:
            raise ImagePoseMatchError(
                "RGB capture basenames must be unique even on case-insensitive filesystems: "
                f"{names[folded]} and {path}"
            )
        if path in sources:
            raise ImagePoseMatchError(f"RGB capture was supplied more than once: {path}")
        names[folded] = path
        sources.add(path)
        resolved.append(path)
    return tuple(sorted(resolved, key=lambda item: (item.name.casefold(), item.name)))


def _rgb_image_metadata(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise ImagePoseMatchError(f"RGB capture is not a decodable image: {path}") from error
    if width <= 0 or height <= 0:
        raise ImagePoseMatchError(f"RGB capture has invalid dimensions: {path}")
    if mode not in {"RGB", "RGBA"}:
        raise ImagePoseMatchError(
            f"RGB capture {path.name} has mode {mode!r}; supply the original RGB/RGBA image, "
            "not a thermal/grayscale band"
        )
    return {
        "width_px": width,
        "height_px": height,
        "mode": mode,
        "format": image_format,
    }


def _matched_records(
    images: Sequence[Path], records: Iterable[CameraPoseRecord]
) -> tuple[CameraPoseRecord, ...]:
    by_name: dict[str, CameraPoseRecord] = {}
    indexes: dict[int, str] = {}
    for record in records:
        if not isinstance(record, CameraPoseRecord):
            raise ImagePoseMatchError("every pose must be a CameraPoseRecord")
        name = _safe_basename(record.image_name, label="CameraPoseRecord.image_name")
        folded = name.casefold()
        if folded in by_name:
            raise ImagePoseMatchError(f"duplicate pose record image name: {name}")
        if record.image_index is not None:
            if record.image_index in indexes:
                raise ImagePoseMatchError(
                    f"duplicate non-null image_index {record.image_index}: "
                    f"{indexes[record.image_index]} and {name}"
                )
            indexes[record.image_index] = name
        by_name[folded] = record

    image_names = {path.name.casefold(): path.name for path in images}
    missing = sorted(image_names[key] for key in image_names.keys() - by_name.keys())
    extra = sorted(by_name[key].image_name for key in by_name.keys() - image_names.keys())
    if missing or extra:
        raise ImagePoseMatchError(
            f"RGB/pose matching must be exactly one-to-one; missing poses={missing}, "
            f"poses without images={extra}"
        )
    matched: list[CameraPoseRecord] = []
    for image in images:
        record = by_name[image.name.casefold()]
        if record.image_name != image.name:
            raise ImagePoseMatchError(
                "image and pose filename case must match exactly for the Linux ODM "
                f"container: {image.name!r} != {record.image_name!r}"
            )
        matched.append(record)
    return tuple(matched)


def _distance_m(first: CameraPoseRecord, second: CameraPoseRecord) -> float:
    mean_latitude = math.radians((first.latitude_deg + second.latitude_deg) / 2.0)
    north = math.radians(second.latitude_deg - first.latitude_deg) * 6_378_137.0
    east = (
        math.radians(second.longitude_deg - first.longitude_deg)
        * 6_378_137.0
        * math.cos(mean_latitude)
    )
    return math.hypot(east, north)


def _rough_overlap_report(
    records: Sequence[CameraPoseRecord], contract: RoughOverlapContract
) -> dict[str, Any]:
    if len(records) < 2:
        raise OverlapPrerequisiteError("at least two distinct positions are needed")
    nearest: list[float] = []
    maximum_baseline = 0.0
    for index, record in enumerate(records):
        distances = [
            _distance_m(record, other)
            for other_index, other in enumerate(records)
            if other_index != index
        ]
        nearest.append(min(distances))
        maximum_baseline = max(maximum_baseline, max(distances))

    allowable_forward = contract.footprint_height_m * (
        1.0 - contract.minimum_forward_overlap_fraction
    )
    allowable_side = contract.footprint_width_m * (
        1.0 - contract.minimum_side_overlap_fraction
    )
    # Without a declared camera-to-flight-line rotation or lane identifiers,
    # a neighbor cannot truthfully be labelled along- versus cross-track.  The
    # smaller threshold is a conservative orientation-independent proximity
    # check; planned forward/side overlaps are separately gated above.
    allowable_neighbor = min(allowable_forward, allowable_side)
    passing = sum(distance <= allowable_neighbor for distance in nearest)
    neighbor_fraction = passing / len(nearest)
    required_extent = min(
        contract.footprint_width_m, contract.footprint_height_m
    ) * contract.minimum_spatial_extent_fraction
    if maximum_baseline < required_extent:
        raise OverlapPrerequisiteError(
            "capture positions have insufficient spatial extent for a terrain reconstruction"
        )
    if neighbor_fraction < contract.minimum_neighbor_pass_fraction:
        raise OverlapPrerequisiteError(
            "capture positions are too sparse for the configured rough overlap gate: "
            f"{neighbor_fraction:.3f} < {contract.minimum_neighbor_pass_fraction:.3f}"
        )
    return {
        "status": "coarse_prerequisites_passed_not_photogrammetry_verified",
        "nominal_agl_m": contract.nominal_agl_m,
        "declared_horizontal_fov_deg": contract.horizontal_fov_deg,
        "declared_vertical_fov_deg": contract.vertical_fov_deg,
        "estimated_nadir_footprint_width_m": contract.footprint_width_m,
        "estimated_nadir_footprint_height_m": contract.footprint_height_m,
        "declared_planned_forward_overlap_fraction": (
            contract.planned_forward_overlap_fraction
        ),
        "declared_planned_side_overlap_fraction": contract.planned_side_overlap_fraction,
        "minimum_forward_overlap_fraction": contract.minimum_forward_overlap_fraction,
        "minimum_side_overlap_fraction": contract.minimum_side_overlap_fraction,
        "orientation_independent_neighbor_limit_m": allowable_neighbor,
        "nearest_neighbor_distance_m": nearest,
        "neighbor_pass_fraction": neighbor_fraction,
        "minimum_neighbor_pass_fraction": contract.minimum_neighbor_pass_fraction,
        "maximum_position_baseline_m": maximum_baseline,
        "minimum_required_spatial_extent_m": required_extent,
        "lateral_lane_evidence": "declared_only_not_inferred_from_the_trajectory",
        "limitations": [
            "GPS proximity does not prove shared visual features or usable parallax.",
            "This gate does not measure cross-track flight-line separation.",
            "It does not detect blur, rolling-shutter distortion, lighting change, "
            "repeated texture, occlusion, moving objects, or weak bundle geometry.",
            "ODM matching, bundle adjustment, component checks, reprojection residuals, "
            "and independent checkpoints must decide whether outputs are usable.",
        ],
    }


def _powershell_token(token: str) -> str:
    return "'" + token.replace("'", "''") + "'"


def _docker_argv(
    dataset_root: Path, project_id: str, config: SurveyPreparationConfig
) -> tuple[str, ...]:
    return (
        "docker",
        "run",
        "--rm",
        "-v",
        f"{dataset_root}:/datasets",
        config.odm_image,
        "--project-path",
        "/datasets",
        project_id,
        "--geo",
        f"/datasets/{project_id}/geo.txt",
        *config.odm_options,
    )


def _file_entry(path: Path, root: Path, *, role: str, immutable_input: bool) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "role": role,
        "immutable_input": immutable_input,
    }


def _make_read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _remove_private_staging(path: Path, parent: Path) -> None:
    try:
        if path.parent.resolve() != parent.resolve() or not path.name.startswith(".staging-"):
            return
    except OSError:
        return

    def make_writable_then_retry(function: Any, failing_path: str, _: Any) -> None:
        os.chmod(failing_path, stat.S_IWRITE)
        function(failing_path)

    shutil.rmtree(path, onerror=make_writable_then_retry)


def _safe_manifest_path(project: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ImmutableProjectError(f"manifest contains unsafe relative path: {relative!r}")
    candidate = project.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(project.resolve())
    except ValueError as error:
        raise ImmutableProjectError(
            f"manifest path escapes the project directory: {relative!r}"
        ) from error
    return candidate


def _verify_existing_project(
    project: Path,
    expected_fingerprint: str,
    expected_input_hashes: Mapping[str, str],
) -> None:
    manifest_path = project / "survey_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImmutableProjectError(
            f"existing project has no readable survey manifest: {project}"
        ) from error
    if manifest.get("schema_version") != SURVEY_SCHEMA_VERSION:
        raise ImmutableProjectError("existing project manifest schema is incompatible")
    if manifest.get("project_fingerprint") != expected_fingerprint:
        raise ImmutableProjectError("existing project fingerprint does not match its path")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ImmutableProjectError("existing project manifest has no file inventory")
    inventory: dict[str, Mapping[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ImmutableProjectError("existing project manifest file entry is invalid")
        relative = str(entry.get("path", ""))
        if relative in inventory:
            raise ImmutableProjectError(
                f"existing project manifest repeats a file path: {relative!r}"
            )
        inventory[relative] = entry
        path = _safe_manifest_path(project, relative)
        if not path.is_file():
            raise ImmutableProjectError(f"immutable project file is missing: {path}")
        expected = str(entry.get("sha256", ""))
        if not _SHA256_PATTERN.fullmatch(expected) or _sha256_file(path) != expected:
            raise ImmutableProjectError(f"immutable project file hash mismatch: {path}")
    for relative, expected_hash in expected_input_hashes.items():
        entry = inventory.get(relative)
        if entry is None or entry.get("sha256") != expected_hash:
            raise ImmutableProjectError(
                f"existing project input no longer matches its source contract: {relative}"
            )
        path = _safe_manifest_path(project, relative)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ImmutableProjectError(
                f"existing project input no longer matches its source contract: {relative}"
            )


def _result_from_project(
    project: Path,
    project_id: str,
    dataset_root: Path,
    config: SurveyPreparationConfig,
    *,
    reused: bool,
) -> SurveyPreparationResult:
    argv = _docker_argv(dataset_root, project_id, config)
    return SurveyPreparationResult(
        project_id=project_id,
        project_directory=project,
        images_directory=project / "images",
        geo_txt=project / "geo.txt",
        pose_records_json=project / "camera_pose_records.json",
        plan_json=project / "survey_plan.json",
        manifest_json=project / "survey_manifest.json",
        docker_argv=argv,
        docker_command_powershell=" ".join(_powershell_token(token) for token in argv),
        reused_existing=reused,
    )


def prepare_odm_survey_project(
    images: Iterable[str | Path],
    records: Iterable[CameraPoseRecord],
    output_root: str | Path,
    config: SurveyPreparationConfig,
    *,
    run_odm: bool = False,
) -> SurveyPreparationResult:
    """Validate and stage a content-addressed ODM project.

    ``run_odm`` defaults to ``False``.  Setting it to ``True`` is the explicit
    authorization boundary for launching Docker; staging alone never starts a
    process or downloads an ODM image.
    """

    normalized_images = _normalized_images(images)
    if len(normalized_images) < config.minimum_image_count:
        raise OverlapPrerequisiteError(
            f"need at least {config.minimum_image_count} RGB captures; "
            f"received {len(normalized_images)}"
        )
    matched_records = _matched_records(normalized_images, records)
    if config.require_position_accuracy:
        missing_accuracy = [
            record.image_name
            for record in matched_records
            if record.horizontal_accuracy_m is None or record.vertical_accuracy_m is None
        ]
        if missing_accuracy:
            raise SurveyPreparationError(
                "position accuracy is required but absent for: "
                + ", ".join(missing_accuracy)
            )
    if config.require_camera_optical_center:
        wrong_reference = [
            record.image_name
            for record in matched_records
            if record.position_reference != "camera_optical_center"
        ]
        if wrong_reference:
            raise SurveyPreparationError(
                "camera optical-center positions are required but absent for: "
                + ", ".join(wrong_reference)
            )

    overlap_report = _rough_overlap_report(matched_records, config.overlap)
    image_inventory = [
        {
            "name": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            **_rgb_image_metadata(path),
        }
        for path in normalized_images
    ]
    poses = [record.as_dict() for record in matched_records]
    geo_text = export_odm_geo_txt(
        matched_records,
        projection=config.datum.horizontal_crs_id,
        require_accuracy=config.require_position_accuracy,
    )
    pose_document = {
        "schema_version": "openprism.camera-pose-record-set/1.0",
        "records": poses,
    }
    fingerprint_material = {
        "schema_version": SURVEY_SCHEMA_VERSION,
        "project_name": config.project_name,
        "datum": asdict(config.datum),
        "overlap": asdict(config.overlap),
        "minimum_image_count": config.minimum_image_count,
        "require_position_accuracy": config.require_position_accuracy,
        "require_camera_optical_center": config.require_camera_optical_center,
        "odm_image": config.odm_image,
        "odm_options": list(config.odm_options),
        "images": image_inventory,
        "camera_pose_records": poses,
    }
    fingerprint = hashlib.sha256(_canonical_json_bytes(fingerprint_material)).hexdigest()
    project_id = f"{config.project_name}-{fingerprint[:16]}"
    dataset_root = Path(output_root).expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"output_root is not a directory: {dataset_root}")
    project = dataset_root / project_id
    expected_input_hashes = {
        **{
            f"images/{entry['name']}": str(entry["sha256"])
            for entry in image_inventory
        },
        "geo.txt": hashlib.sha256(geo_text.encode("utf-8")).hexdigest(),
        "camera_pose_records.json": hashlib.sha256(
            _json_document_bytes(pose_document)
        ).hexdigest(),
    }

    if project.exists():
        if not project.is_dir():
            raise ImmutableProjectError(f"content-addressed project path is not a directory: {project}")
        _verify_existing_project(project, fingerprint, expected_input_hashes)
        result = _result_from_project(
            project, project_id, dataset_root, config, reused=True
        )
        return execute_odm_project(result) if run_odm else result

    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=dataset_root))
    try:
        image_directory = staging / "images"
        image_directory.mkdir()
        for source, inventory in zip(normalized_images, image_inventory, strict=True):
            destination = image_directory / source.name
            shutil.copyfile(source, destination)
            if _sha256_file(destination) != inventory["sha256"]:
                raise ImmutableProjectError(f"copied image failed hash verification: {source}")

        geo_path = staging / "geo.txt"
        geo_path.write_text(
            geo_text,
            encoding="utf-8",
            newline="\n",
        )
        pose_path = staging / "camera_pose_records.json"
        _write_json(pose_path, pose_document)

        final_argv = _docker_argv(dataset_root, project_id, config)
        limitations = [
            "The geo.txt file contains longitude, latitude, and MSL altitude only.",
            "Yaw, pitch, roll, and pose accuracy are deliberately not serialized because "
            "ODM would consume placeholder angles as real orientation constraints.",
            "Pixhawk/GNSS coordinates are camera-position priors, not terrain depth, "
            "pixel-to-ground correspondences, or an accuracy certificate.",
            "Terrain geometry exists only if ODM obtains a valid multi-view reconstruction "
            "from overlapping image features.",
            "A DSM is not a bare-earth DTM and no output is survey-grade until independent "
            "checkpoints, datum handling, residuals, completeness, and scale are validated.",
        ]
        position_references = sorted(
            {record.position_reference for record in matched_records}
        )
        plan = {
            "schema_version": SURVEY_SCHEMA_VERSION,
            "project_id": project_id,
            "project_fingerprint": fingerprint,
            "workflow": "post_flight_rgb_structure_from_motion_and_multi_view_stereo",
            "execution_state": "prepared_not_executed",
            "docker": {
                "argv": list(final_argv),
                "powershell_command": " ".join(
                    _powershell_token(token) for token in final_argv
                ),
                "image_reference": config.odm_image,
                "image_is_digest_pinned": "@sha256:" in config.odm_image,
                "mutable_tag_warning": (
                    None
                    if "@sha256:" in config.odm_image
                    else "ODM image is not digest-pinned; resolve and pin a tested digest for repeatability."
                ),
            },
            "datum_contract": asdict(config.datum),
            "geo_txt_contract": {
                "horizontal_header": "EPSG:4326",
                "vertical_values": "orthometric_msl_m_as_declared_by_vertical_datum_contract",
                "row_fields": ["image_name", "longitude_deg", "latitude_deg", "altitude_msl_m"],
                "orientation_serialized": False,
                "accuracy_serialized": False,
            },
            "capture_count": len(matched_records),
            "position_references": position_references,
            "camera_center_warning": (
                None
                if position_references == ["camera_optical_center"]
                else "Some positions are not declared camera optical centers; account for antenna/body/camera lever arms."
            ),
            "rough_overlap_gate": overlap_report,
            "expected_odm_products": [
                "odm_orthophoto/odm_orthophoto.tif",
                "odm_dem/dsm.tif",
                "odm_georeferencing/odm_georeferenced_model.laz",
                "odm_texturing/odm_textured_model_geo.obj",
            ],
            "limitations": limitations,
            "required_acceptance_checks": [
                "connected reconstruction component and registered-image count",
                "tie-point distribution, bundle residuals, and obvious warped geometry",
                "independent horizontal and vertical checkpoints in the declared datums",
                "DSM completeness and artifacts over vegetation, water, shadows, and moving objects",
                "camera model, rolling-shutter, time synchronization, and lever-arm validity",
            ],
        }
        plan_path = staging / "survey_plan.json"
        _write_json(plan_path, plan)

        files: list[dict[str, Any]] = []
        for path in sorted(image_directory.iterdir(), key=lambda item: item.name.casefold()):
            files.append(
                _file_entry(path, staging, role="immutable_rgb_capture", immutable_input=True)
            )
        files.extend(
            (
                _file_entry(geo_path, staging, role="position_only_odm_geolocation", immutable_input=True),
                _file_entry(pose_path, staging, role="full_pose_provenance", immutable_input=True),
                _file_entry(plan_path, staging, role="execution_and_validation_plan", immutable_input=True),
            )
        )
        manifest = {
            "schema_version": SURVEY_SCHEMA_VERSION,
            "project_id": project_id,
            "project_fingerprint": fingerprint,
            "input_contract_sha256": hashlib.sha256(
                _canonical_json_bytes(fingerprint_material)
            ).hexdigest(),
            "content_addressed": True,
            "capture_count": len(matched_records),
            "files": files,
            "claims": {
                "terrain_reconstructed": False,
                "gps_only_mapping": False,
                "survey_grade": False,
                "state": "inputs_staged_for_external_photogrammetric_reconstruction",
            },
        }
        manifest_path = staging / "survey_manifest.json"
        _write_json(manifest_path, manifest)

        for entry in files:
            _make_read_only(_safe_manifest_path(staging, entry["path"]))
        _make_read_only(manifest_path)

        try:
            staging.rename(project)
        except FileExistsError:
            _remove_private_staging(staging, dataset_root)
            _verify_existing_project(project, fingerprint, expected_input_hashes)
        result = _result_from_project(
            project, project_id, dataset_root, config, reused=False
        )
    except Exception:
        if staging.exists():
            _remove_private_staging(staging, dataset_root)
        raise

    return execute_odm_project(result) if run_odm else result


def execute_odm_project(result: SurveyPreparationResult) -> SurveyPreparationResult:
    """Run the prepared command after an explicit caller request.

    The subprocess is invoked with an argv list and never through a shell.
    Docker/ODM errors are surfaced; a non-zero process is not represented as a
    completed reconstruction.
    """

    if shutil.which("docker") is None:
        raise DockerUnavailableError(
            "ODM execution was explicitly requested, but 'docker' is not on PATH; "
            "the verified project remains staged and can be run later"
        )
    completed = subprocess.run(
        list(result.docker_argv),
        cwd=result.project_directory.parent,
        check=False,
    )
    if completed.returncode != 0:
        raise SurveyExecutionError(
            f"OpenDroneMap Docker process failed with exit code {completed.returncode}; "
            "do not treat partial outputs as a completed terrain reconstruction"
        )
    return replace(result, odm_executed=True, odm_returncode=0)


def camera_pose_record_from_dict(payload: Mapping[str, Any]) -> CameraPoseRecord:
    """Decode either a flat constructor mapping or ``CameraPoseRecord.as_dict``."""

    if not isinstance(payload, Mapping):
        raise SurveyPreparationError("camera pose JSON entries must be objects")
    if "geodetic" not in payload:
        try:
            return CameraPoseRecord(**dict(payload))
        except (TypeError, ValueError) as error:
            raise SurveyPreparationError(f"invalid flat CameraPoseRecord: {error}") from error

    try:
        geodetic = payload["geodetic"]
        camera_pose = payload["camera_pose"]
        capture_time = payload["capture_time"]
        uncertainty = payload["uncertainty"]
        quality = payload["quality"]
        provenance = payload["provenance"]
        if not all(
            isinstance(section, Mapping)
            for section in (
                geodetic,
                camera_pose,
                capture_time,
                uncertainty,
                quality,
                provenance,
            )
        ):
            raise TypeError("nested pose sections must be objects")
        return CameraPoseRecord(
            image_name=payload["image_name"],
            image_index=payload.get("image_index"),
            latitude_deg=geodetic["latitude_deg"],
            longitude_deg=geodetic["longitude_deg"],
            altitude_msl_m=geodetic["altitude_msl_m"],
            relative_altitude_m=geodetic.get("relative_altitude_m"),
            quaternion_camera_flu_to_enu_wxyz=camera_pose[
                "quaternion_camera_flu_to_enu_wxyz"
            ],
            quaternion_camera_optical_to_enu_wxyz=camera_pose[
                "quaternion_camera_optical_to_enu_wxyz"
            ],
            yaw_deg=camera_pose["yaw_deg"],
            pitch_deg=camera_pose["pitch_deg"],
            roll_deg=camera_pose["roll_deg"],
            capture_monotonic_ns=capture_time.get("monotonic_ns"),
            capture_utc_ns=capture_time.get("utc_ns"),
            event_monotonic_ns=capture_time.get("event_monotonic_ns"),
            event_utc_ns=capture_time.get("event_utc_ns"),
            clock_domain=capture_time["clock_domain"],
            time_basis=capture_time["basis"],
            time_uncertainty_ns=capture_time.get("uncertainty_ns"),
            horizontal_accuracy_m=uncertainty.get("horizontal_m"),
            vertical_accuracy_m=uncertainty.get("vertical_m"),
            attitude_accuracy_deg=uncertainty.get("attitude_deg"),
            fix_type=quality.get("fix_type"),
            fix_quality=quality["fix_quality"],
            rtk_status=quality["rtk_status"],
            source_message=provenance["source_message"],
            position_source=provenance["position_source"],
            attitude_source=provenance["attitude_source"],
            interpolation_span_ns=provenance.get("interpolation_span_ns"),
            position_frame=geodetic.get("frame", GEODETIC_POSITION_FRAME),
            relative_altitude_reference=geodetic.get(
                "relative_altitude_reference", "unspecified"
            ),
            position_reference=geodetic.get("position_reference", "unspecified"),
            input_attitude_convention=provenance.get(
                "input_attitude_convention",
                "Hamilton wxyz coordinate rotation MAV_FRAME_LOCAL_NED -> MAV_FRAME_BODY_FRD/camera_FRD",
            ),
            input_attitude_profile=provenance.get(
                "input_attitude_profile", "externally_constructed_record"
            ),
            image_match_basis=provenance["image_match_basis"],
            system_id=provenance.get("system_id"),
            component_id=provenance.get("component_id"),
            camera_id=provenance.get("camera_id"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SurveyPreparationError(f"invalid nested CameraPoseRecord: {error}") from error


def load_camera_pose_records(path: str | Path) -> tuple[CameraPoseRecord, ...]:
    """Load a list or ``{"records": [...]}`` pose JSON document."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SurveyPreparationError(f"cannot read camera pose JSON {source}: {error}") from error
    entries = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, list):
        raise SurveyPreparationError("camera pose JSON must be a list or an object with a records list")
    return tuple(camera_pose_record_from_dict(entry) for entry in entries)


__all__ = [
    "DockerUnavailableError",
    "ImagePoseMatchError",
    "ImmutableProjectError",
    "OverlapPrerequisiteError",
    "RoughOverlapContract",
    "SURVEY_SCHEMA_VERSION",
    "SUPPORTED_RGB_SUFFIXES",
    "SurveyDatumContract",
    "SurveyExecutionError",
    "SurveyPreparationConfig",
    "SurveyPreparationError",
    "SurveyPreparationResult",
    "camera_pose_record_from_dict",
    "execute_odm_project",
    "load_camera_pose_records",
    "prepare_odm_survey_project",
]
