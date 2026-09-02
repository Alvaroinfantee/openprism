"""Dependency-light HTTP reference server for the OpenPRISM operator canvas."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, UnidentifiedImageError

from .autonomy import AdaptiveFusionController
from .datasets import DEFAULT_DATA_ROOT, DatasetCatalog
from .fusion import (
    EvidenceFusionEngine,
    FusionConfig,
    support_colormap,
    semantic_color,
)
from .rendering import image_data_url


WEB_ROOT = Path(__file__).resolve().parent / "web"
DEFAULT_ATLAS_ROOT = (
    Path.cwd() / "output" / "openprism_atlas" / "latest"
)
_ATLAS_METADATA_LIMIT_BYTES = 1_000_000
_ATLAS_PREVIEW_LIMIT_BYTES = 32_000_000
_ATLAS_PREVIEW_LIMIT_PIXELS = 40_000_000
_ATLAS_OBJECTS_LIMIT_BYTES = 8_000_000
_ATLAS_OBJECT_LIMIT = 10_000
# Small negative ages can occur while two correctly synchronized machines are
# a few seconds apart. Anything farther in the future is operator-significant
# clock error and must never be presented as a fresh tactical product.
_ATLAS_FUTURE_PUBLICATION_TOLERANCE_S = 5.0


class OperatorApplication:
    """Own the catalog, fusion engine, and bounded rendered-frame cache."""

    def __init__(
        self,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        atlas_root: Path | str = DEFAULT_ATLAS_ROOT,
    ) -> None:
        self.catalog = DatasetCatalog(data_root)
        self.engine = EvidenceFusionEngine()
        self.controller = AdaptiveFusionController()
        self.atlas_root = Path(atlas_root).resolve()

    def catalog_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "openprism.operator-api/0.1",
            "datasets": self.catalog.datasets(),
        }

    @staticmethod
    def _atlas_unavailable(reason: str) -> dict[str, Any]:
        return {
            "schema_version": "openprism.operator-atlas/0.1",
            "available": False,
            "reason": reason,
            "expected_bundle": "output/openprism_atlas/latest",
            "synthetic": False,
        }

    def _atlas_bundle_root(self) -> tuple[Path, dict[str, Any] | None]:
        """Resolve one atomically published immutable generation or legacy root."""

        root = self.atlas_root.resolve(strict=True)
        pointer = root / "CURRENT"
        if not pointer.exists():
            return root, None
        if not pointer.is_file() or pointer.stat().st_size > 4_096:
            raise ValueError("atlas CURRENT pointer has an unsafe size")
        current = json.loads(pointer.read_text(encoding="utf-8"))
        if not isinstance(current, dict) or current.get("schema_version") != (
            "openprism.atlas-current/0.1"
        ):
            raise ValueError("atlas CURRENT pointer has an unsupported schema")
        generation_id = current.get("generation_id")
        if not isinstance(generation_id, str) or re.fullmatch(
            r"[0-9a-f]{32}", generation_id
        ) is None:
            raise ValueError("atlas CURRENT generation id is invalid")
        generation = (root / ".generations" / generation_id).resolve(strict=True)
        try:
            generation.relative_to(root)
        except ValueError as exc:
            raise ValueError("atlas CURRENT generation escapes its root") from exc
        if not generation.is_dir():
            raise ValueError("atlas CURRENT generation is not a directory")
        manifest_path = generation / "bundle_manifest.json"
        if not manifest_path.is_file() or manifest_path.stat().st_size > 1_000_000:
            raise ValueError("atlas generation manifest is missing or unsafe")
        manifest_bytes = manifest_path.read_bytes()
        expected_manifest_hash = current.get("manifest_sha256")
        if (
            not isinstance(expected_manifest_hash, str)
            or hashlib.sha256(manifest_bytes).hexdigest()
            != expected_manifest_hash
        ):
            raise ValueError("atlas generation manifest hash does not match CURRENT")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != "openprism.atlas-manifest/0.1"
            or manifest.get("generation_id") != generation_id
            or not isinstance(manifest.get("files"), dict)
        ):
            raise ValueError("atlas generation manifest is invalid")
        return generation, manifest["files"]

    def _atlas_file(
        self,
        filename: str,
        maximum_bytes: int,
        *,
        bundle_root: Path | None = None,
        manifest_files: dict[str, Any] | None = None,
    ) -> Path:
        """Resolve one fixed bundle member without allowing path escape."""

        try:
            root = (
                self.atlas_root.resolve(strict=True)
                if bundle_root is None
                else bundle_root.resolve(strict=True)
            )
            candidate = (root / filename).resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"atlas bundle member is missing: {filename}") from exc
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("atlas bundle member escapes the configured directory") from exc
        if not candidate.is_file():
            raise ValueError(f"atlas bundle member is not a regular file: {filename}")
        size = candidate.stat().st_size
        if size <= 0 or size > maximum_bytes:
            raise ValueError(f"atlas bundle member has an unsafe size: {filename}")
        if manifest_files is not None:
            entry = manifest_files.get(filename)
            if not isinstance(entry, dict):
                raise ValueError(f"atlas manifest omits bundle member: {filename}")
            if entry.get("size_bytes") != size:
                raise ValueError(f"atlas bundle member size mismatch: {filename}")
            expected_hash = entry.get("sha256")
            if (
                not isinstance(expected_hash, str)
                or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_hash
            ):
                raise ValueError(f"atlas bundle member hash mismatch: {filename}")
        return candidate

    def _atlas_preview(
        self,
        filename: str,
        *,
        bundle_root: Path | None = None,
        manifest_files: dict[str, Any] | None = None,
    ) -> tuple[str, tuple[int, int]]:
        path = self._atlas_file(
            filename,
            _ATLAS_PREVIEW_LIMIT_BYTES,
            bundle_root=bundle_root,
            manifest_files=manifest_files,
        )
        payload = path.read_bytes()
        try:
            with Image.open(BytesIO(payload)) as preview:
                width, height = preview.size
                image_format = preview.format
                if (
                    width <= 0
                    or height <= 0
                    or width * height > _ATLAS_PREVIEW_LIMIT_PIXELS
                ):
                    raise ValueError(f"atlas preview has unsafe dimensions: {filename}")
                if image_format != "PNG":
                    raise ValueError(f"atlas preview must be PNG: {filename}")
                preview.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"atlas preview is not a valid PNG: {filename}") from exc
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:image/png;base64,{encoded}", (width, height)

    @staticmethod
    def _atlas_number(value: Any) -> float | int | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(float(value)):
            return None
        return value

    @staticmethod
    def _atlas_text(value: Any, *, maximum_length: int = 240) -> str | None:
        if not isinstance(value, str):
            return None
        text = "".join(
            character
            for character in value
            if character >= " " and character != "\x7f"
        ).strip()
        if not text:
            return None
        return text[:maximum_length]

    @classmethod
    def _atlas_publication_state(cls, publication: Any) -> dict[str, Any]:
        """Validate publication timing and derive a fail-closed freshness state."""

        source = publication if isinstance(publication, dict) else {}
        published_at_utc = cls._atlas_text(
            source.get("published_at_utc"), maximum_length=80
        )
        freshness_ttl_s = cls._atlas_number(
            source.get("operator_freshness_ttl_s")
        )
        freshness_status = "unverified"
        age_s: float | None = None
        if published_at_utc is not None:
            try:
                published = datetime.fromisoformat(
                    published_at_utc.replace("Z", "+00:00")
                )
                if published.tzinfo is None:
                    raise ValueError("publication timestamp has no timezone")
                raw_age_s = (
                    datetime.now(timezone.utc) - published.astimezone(timezone.utc)
                ).total_seconds()
                if raw_age_s < -_ATLAS_FUTURE_PUBLICATION_TOLERANCE_S:
                    # Preserve the negative age for diagnostics, but never let
                    # max(0, age) turn a future-dated bundle into "fresh".
                    age_s = raw_age_s
                    freshness_status = "future"
                else:
                    age_s = max(0.0, raw_age_s)
                    if freshness_ttl_s is None:
                        freshness_status = "snapshot"
                    elif freshness_ttl_s <= 0:
                        raise ValueError("atlas freshness TTL must be positive")
                    else:
                        freshness_status = (
                            "fresh" if age_s <= freshness_ttl_s else "stale"
                        )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("atlas publication timestamp is invalid") from exc
        return {
            "published_at_utc": published_at_utc,
            "freshness_ttl_s": freshness_ttl_s,
            "freshness_status": freshness_status,
            "publication_age_s": age_s,
            "future_publication_tolerance_s": (
                _ATLAS_FUTURE_PUBLICATION_TOLERANCE_S
            ),
        }

    def atlas_status_payload(self) -> dict[str, Any]:
        """Return a lightweight, integrity-checked atlas generation status.

        The operator polls this endpoint instead of repeatedly downloading the
        base64 previews. A changed revision causes one full ``/api/atlas`` load.
        """

        unavailable = {
            "schema_version": "openprism.operator-atlas-status/0.1",
            "available": False,
            "revision_id": None,
        }
        if not self.atlas_root.is_dir():
            return {
                **unavailable,
                "reason": "No exported atlas bundle is available yet.",
            }
        try:
            bundle_root, manifest_files = self._atlas_bundle_root()
            metadata_path = self._atlas_file(
                "atlas_metadata.json",
                _ATLAS_METADATA_LIMIT_BYTES,
                bundle_root=bundle_root,
                manifest_files=manifest_files,
            )
            metadata_bytes = metadata_path.read_bytes()
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("atlas metadata must be a JSON object")
            if metadata.get("schema_version") not in {
                "openprism.atlas-bundle/0.1",
                "openprism.atlas-bundle/0.2",
            }:
                raise ValueError("atlas metadata uses an unsupported schema")
            if metadata.get("survey_grade") is not False:
                raise ValueError("atlas bundle must explicitly declare survey_grade false")
            product = metadata.get("product")
            if product not in {
                "live_tactical_2.5d_mosaic",
                "live_tactical_2.5d_height_field_mosaic",
                "live_tactical_flat_ground_mosaic",
            }:
                raise ValueError("atlas metadata does not describe a tactical mosaic")
            revision_id = (
                bundle_root.name
                if manifest_files is not None
                else hashlib.sha256(metadata_bytes).hexdigest()
            )
            publication_state = self._atlas_publication_state(
                metadata.get("publication")
            )
            return {
                "schema_version": "openprism.operator-atlas-status/0.1",
                "available": True,
                "revision_id": revision_id,
                "server_time_utc": datetime.now(timezone.utc).isoformat(),
                **publication_state,
            }
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            return {
                **unavailable,
                "reason": f"Atlas bundle is incomplete or invalid: {exc}",
            }
        except OSError:
            return {
                **unavailable,
                "reason": "Atlas bundle could not be read safely.",
            }

    def atlas_payload(self) -> dict[str, Any]:
        """Return the latest tactical atlas with an explicit evidence-origin state.

        Only a small allow-list of metadata is exposed.  Preview files are fixed
        names, path-contained, size-bounded, decoded as PNG, and returned as data
        URLs; arbitrary paths or HTML from the bundle are never served.
        """

        if not self.atlas_root.is_dir():
            return self._atlas_unavailable(
                "No exported atlas bundle is available yet."
            )
        try:
            bundle_root, manifest_files = self._atlas_bundle_root()
            metadata_path = self._atlas_file(
                "atlas_metadata.json",
                _ATLAS_METADATA_LIMIT_BYTES,
                bundle_root=bundle_root,
                manifest_files=manifest_files,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("atlas metadata must be a JSON object")
            if metadata.get("schema_version") not in {
                "openprism.atlas-bundle/0.1",
                "openprism.atlas-bundle/0.2",
            }:
                raise ValueError("atlas metadata uses an unsupported schema")
            product = metadata.get("product")
            if product not in {
                "live_tactical_2.5d_mosaic",
                "live_tactical_2.5d_height_field_mosaic",
                "live_tactical_flat_ground_mosaic",
            }:
                raise ValueError("atlas metadata does not describe a tactical mosaic")
            if metadata.get("survey_grade") is not False:
                raise ValueError("atlas bundle must explicitly declare survey_grade false")

            mission_id = self._atlas_text(metadata.get("mission_id"))
            calibration_id = self._atlas_text(metadata.get("camera_calibration_id"))
            if mission_id is None or calibration_id is None:
                raise ValueError("atlas metadata is missing mission or calibration identity")
            demonstration = metadata.get("demonstration")
            demonstration = demonstration if isinstance(demonstration, dict) else {}
            data_provenance = metadata.get("data_provenance")
            data_provenance = (
                data_provenance if isinstance(data_provenance, dict) else {}
            )
            real_sensor_data = data_provenance.get(
                "real_sensor_data", demonstration.get("real_sensor_data")
            )
            real_navigation_data = data_provenance.get(
                "real_navigation_data",
                data_provenance.get(
                    "real_gps_data", demonstration.get("real_gps_data")
                ),
            )
            synthetic = (
                demonstration.get("synthetic") is True
                or real_sensor_data is False
                or real_navigation_data is False
                or mission_id.upper().startswith("SYNTHETIC")
                or calibration_id.upper().startswith("SYNTHETIC")
                or (bundle_root / "SYNTHETIC_DEMO_NOTICE.txt").is_file()
            )
            captured = real_sensor_data is True and real_navigation_data is True
            if synthetic and captured:
                raise ValueError("atlas origin declarations conflict")
            origin_status = (
                "synthetic_demo"
                if synthetic
                else "captured_evidence"
                if captured
                else "unverified"
            )

            images: dict[str, str] = {}
            dimensions: set[tuple[int, int]] = set()
            for key, filename in (
                ("rgb", "atlas_rgb.png"),
                ("thermal", "atlas_thermal.png"),
                ("support", "atlas_support.png"),
            ):
                images[key], size = self._atlas_preview(
                    filename,
                    bundle_root=bundle_root,
                    manifest_files=manifest_files,
                )
                dimensions.add(size)
            if len(dimensions) != 1:
                raise ValueError("atlas previews do not share one pixel grid")
            width, height = dimensions.pop()

            mosaic = metadata.get("mosaic")
            mosaic = mosaic if isinstance(mosaic, dict) else {}
            grid_source = mosaic.get("grid")
            grid_source = grid_source if isinstance(grid_source, dict) else {}
            if grid_source.get("north_up") is not True:
                raise ValueError("atlas grid must explicitly declare north_up true")
            coordinate_source = mosaic.get("coordinate_reference")
            coordinate_source = (
                coordinate_source if isinstance(coordinate_source, dict) else {}
            )
            origin_source = coordinate_source.get("origin")
            origin_source = origin_source if isinstance(origin_source, dict) else {}

            grid: dict[str, Any] = {
                "north_up": grid_source.get("north_up") is True,
                "shape": [height, width],
            }
            for key in (
                "row_zero_north_m",
                "east_min_m",
                "east_max_m",
                "north_min_m",
                "north_max_m",
                "ground_elevation_enu_m",
                "resolution_m",
            ):
                value = self._atlas_number(grid_source.get(key))
                if value is not None:
                    grid[key] = value

            coordinate_reference: dict[str, Any] = {}
            for key in (
                "type",
                "axes",
                "horizontal_datum",
                "vertical_datum",
                "frame_id",
            ):
                value = self._atlas_text(coordinate_source.get(key))
                if value is not None:
                    coordinate_reference[key] = value
            origin: dict[str, Any] = {}
            for key in ("latitude_deg", "longitude_deg", "ellipsoid_height_m"):
                value = self._atlas_number(origin_source.get(key))
                if value is not None:
                    origin[key] = value
            if origin:
                coordinate_reference["origin"] = origin

            objects: list[dict[str, Any]] = []
            objects_declared = (
                manifest_files is not None
                and "atlas_objects.geojson" in manifest_files
            ) or (bundle_root / "atlas_objects.geojson").is_file()
            if objects_declared:
                objects_path = self._atlas_file(
                    "atlas_objects.geojson",
                    _ATLAS_OBJECTS_LIMIT_BYTES,
                    bundle_root=bundle_root,
                    manifest_files=manifest_files,
                )
                object_collection = json.loads(
                    objects_path.read_text(encoding="utf-8")
                )
                if (
                    not isinstance(object_collection, dict)
                    or object_collection.get("type") != "FeatureCollection"
                    or not isinstance(object_collection.get("features"), list)
                ):
                    raise ValueError("atlas object layer is not valid GeoJSON")
                for feature in object_collection["features"][:_ATLAS_OBJECT_LIMIT]:
                    if not isinstance(feature, dict):
                        continue
                    properties = feature.get("properties")
                    properties = properties if isinstance(properties, dict) else {}
                    east_m = self._atlas_number(properties.get("east_m"))
                    north_m = self._atlas_number(properties.get("north_m"))
                    label = self._atlas_text(properties.get("label"), maximum_length=80)
                    object_id = self._atlas_text(
                        properties.get("object_id"), maximum_length=120
                    )
                    if east_m is None or north_m is None or label is None or object_id is None:
                        continue
                    item: dict[str, Any] = {
                        "object_id": object_id,
                        "label": label,
                        "east_m": east_m,
                        "north_m": north_m,
                    }
                    for key in ("confidence", "horizontal_uncertainty_m"):
                        value = self._atlas_number(properties.get(key))
                        if value is not None:
                            item[key] = value
                    timestamp = self._atlas_text(
                        properties.get("timestamp_tai_ns"), maximum_length=32
                    )
                    if timestamp is not None and timestamp.isdecimal():
                        item["timestamp_tai_ns"] = timestamp
                    objects.append(item)

            publication = metadata.get("publication")
            publication = publication if isinstance(publication, dict) else {}
            publication_state = self._atlas_publication_state(publication)
            revision_id = (
                bundle_root.name
                if manifest_files is not None
                else hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            )

            layers = metadata.get("layers")
            safe_layers = []
            if isinstance(layers, list):
                for layer in layers[:64]:
                    value = self._atlas_text(layer, maximum_length=80)
                    if value is not None:
                        safe_layers.append(value)

            def safe_count(key: str) -> int:
                value = metadata.get(key)
                if isinstance(value, bool) or not isinstance(value, int):
                    return 0
                return max(0, value)

            source_ids = metadata.get("source_ids")
            source_count = (
                min(len(source_ids), 1_000_000)
                if isinstance(source_ids, list)
                else 0
            )
            provenance_source = (
                "analytic_synthetic_atlas_demo"
                if synthetic
                else "exported_openprism_atlas_bundle"
                if captured
                else "unverified_atlas_bundle"
            )
            pixel_generation = (
                "analytic_synthetic_demo_layers"
                if synthetic
                else "weighted_projection_of_captured_sensor_evidence"
                if captured
                else "weighted_projection_from_unverified_bundle_inputs"
            )
            dynamic_exclusion = data_provenance.get(
                "dynamic_objects_excluded_from_static_atlas",
                demonstration.get(
                    "dynamic_pixels_excluded_from_static_atlas"
                ),
            )
            dynamic_exclusion = (
                dynamic_exclusion if isinstance(dynamic_exclusion, bool) else None
            )

            return {
                "schema_version": "openprism.operator-atlas/0.1",
                "available": True,
                "synthetic": synthetic,
                "origin_status": origin_status,
                "meta": {
                    "mission_id": mission_id,
                    "product": product,
                    "survey_grade": False,
                    "synthetic": synthetic,
                    "origin_status": origin_status,
                    "real_sensor_data": (
                        real_sensor_data if isinstance(real_sensor_data, bool) else None
                    ),
                    "real_navigation_data": (
                        real_navigation_data
                        if isinstance(real_navigation_data, bool)
                        else None
                    ),
                    "display_role": "north_up_tactical_2.5d_mosaic",
                    "camera_calibration_id": calibration_id,
                    "accepted_capture_count": safe_count("accepted_capture_count"),
                    "rejected_capture_count": safe_count("rejected_capture_count"),
                    "source_count": source_count,
                    "width": width,
                    "height": height,
                    "coordinate_reference": coordinate_reference,
                    "grid": grid,
                    "layers": safe_layers,
                    "dynamic_objects_excluded_from_static_atlas": dynamic_exclusion,
                    "object_count": len(objects),
                    "track_count": len(
                        {item["object_id"] for item in objects}
                    ),
                    "revision_id": revision_id,
                    **publication_state,
                    "temporal_role": self._atlas_text(
                        publication.get("temporal_role"), maximum_length=80
                    ),
                },
                "images": images,
                "objects": objects,
                "provenance": {
                    "source": provenance_source,
                    "pixel_generation": pixel_generation,
                    "model_generated_pixels": False,
                    "dynamic_objects_baked_into_terrain": (
                        False if dynamic_exclusion is True else None
                    ),
                },
            }
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            return self._atlas_unavailable(f"Atlas bundle is incomplete or invalid: {exc}")
        except OSError:
            return self._atlas_unavailable("Atlas bundle could not be read safely.")

    @lru_cache(maxsize=16)
    def frame_payload(
        self,
        dataset: str,
        split: str,
        index: int,
        thermal_gain: float,
        automatic_control: bool = False,
    ) -> dict[str, Any]:
        count = self.catalog.count(dataset, split)
        if count <= 0:
            raise ValueError("the requested split is empty")
        if index < 0 or index >= count:
            raise IndexError(f"index must be between 0 and {count - 1}")
        gain = round(float(thermal_gain), 2)
        if not 0.0 <= gain <= 2.5:
            raise ValueError("thermal_gain must be in [0, 2.5]")

        record = self.catalog.record(dataset, split, index)
        frame = self.catalog.load(dataset, split, index)
        recommendation, policy_features, probe = self.controller.recommend(
            frame, self.engine
        )
        applied_gain = recommendation.thermal_gain if automatic_control else gain
        output = (
            probe
            if applied_gain == 1.0
            else self.engine.fuse(frame, FusionConfig(thermal_gain=applied_gain))
        )
        ai_digest = self.controller.scene_digest(
            frame,
            output,
            recommendation,
            policy_features,
            automatic_control=automatic_control,
            applied_thermal_gain=applied_gain,
        )
        registration_mean = float(np.mean(output.registration_support))
        fusion_mean = float(np.mean(output.fusion_support))
        annotation_source = str(frame.provenance.get("annotation_source", "none"))
        terrain_classes = []
        if frame.semantic_mask is not None:
            class_ids, counts = np.unique(frame.semantic_mask, return_counts=True)
            total = float(frame.semantic_mask.size)
            for class_id, pixels in zip(class_ids, counts):
                red, green, blue = semantic_color(int(class_id))
                terrain_classes.append(
                    {
                        "id": int(class_id),
                        "label": frame.semantic_classes.get(
                            int(class_id), f"class {int(class_id)}"
                        ),
                        "color": f"#{red:02x}{green:02x}{blue:02x}",
                        "coverage": float(pixels) / total,
                    }
                )

        return {
            "schema_version": "openprism.operator-frame/0.1",
            "meta": {
                "dataset": dataset,
                "split": split,
                "index": index,
                "count": count,
                "sample_id": record.sample_id,
                "frame_id": frame.frame_id,
                "width": int(output.operator_rgb.shape[1]),
                "height": int(output.operator_rgb.shape[0]),
                "registration_confidence": None,
                "registration_support_score": registration_mean,
                "registration_support_prior": registration_mean,
                "registration_evidence_kind": "publisher_declared_prior",
                "fusion_confidence": None,
                "fusion_support_score": fusion_mean,
                "fusion_evidence_kind": "deterministic_heuristic_support",
                "registration_status": "declared_rectified_not_measured",
                "pixel_fusion_applied": output.pixel_fusion_applied,
                "synchronization_state": output.synchronization_state,
                "synchronization_basis": frame.synchronization.basis,
                "physical_timing_uncertainty_ns": (
                    output.physical_timing_uncertainty_ns
                ),
                "machine_channels": list(output.channel_names),
                "annotation_source": annotation_source,
                "thermal_units": "relative thermal intensity",
                "thermal_display_transform": "per_frame_robust_percentile_normalized",
                "capture_time_status": "unavailable_in_extracted_pair",
                "source_mode": "dataset_replay",
                "fusion_mode": output.provenance["fusion_mode"],
                "non_hallucinatory": True,
                "scene_group": record.scene_group,
            },
            "images": {
                "fused": image_data_url(output.operator_rgb),
                "visible": image_data_url(output.visible_view),
                "thermal": image_data_url(output.thermal_view),
                "support": image_data_url(
                    support_colormap(output.fusion_support), image_format="PNG"
                ),
                "semantic": (
                    image_data_url(output.semantic_view)
                    if output.semantic_view is not None
                    else None
                ),
            },
            "detections": [detection.as_dict() for detection in frame.detections],
            "terrain_classes": terrain_classes,
            "ai": ai_digest,
            "provenance": {
                "algorithm": output.provenance["algorithm"],
                "source_sensors": list(output.provenance["source_sensors"]),
                "registration": dict(output.provenance["registration"]),
                "non_hallucinatory": True,
            },
        }

    def ai_context_payload(
        self,
        dataset: str,
        split: str,
        index: int,
    ) -> dict[str, Any]:
        """Return a compact model-facing digest without image data URLs."""

        payload = self.frame_payload(
            dataset,
            split,
            index,
            thermal_gain=1.0,
            automatic_control=True,
        )
        meta = payload["meta"]
        return {
            "schema_version": "openprism.ai-context-envelope/1.0",
            "dataset": meta["dataset"],
            "split": meta["split"],
            "index": meta["index"],
            "sample_id": meta["sample_id"],
            "digest": payload["ai"],
        }


def build_handler(application: OperatorApplication) -> type[BaseHTTPRequestHandler]:
    class OperatorHandler(BaseHTTPRequestHandler):
        server_version = "OpenPRISM/0.1"

        def log_message(self, format: str, *args: object) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

        def _send_bytes(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(
            self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._send_bytes(encoded, "application/json; charset=utf-8", status)

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"error": message, "status": int(status)}, status)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._send_json(
                    {
                        "status": "ok",
                        "version": "0.1.0",
                        "datasets": len(application.catalog.datasets()),
                    }
                )
                return
            if parsed.path == "/api/catalog":
                self._send_json(application.catalog_payload())
                return
            if parsed.path == "/api/atlas":
                self._send_json(application.atlas_payload())
                return
            if parsed.path == "/api/atlas/status":
                self._send_json(application.atlas_status_payload())
                return
            if parsed.path == "/api/ai/context":
                self._ai_context(parse_qs(parsed.query))
                return
            if parsed.path == "/api/frame":
                self._frame(parse_qs(parsed.query))
                return
            if parsed.path == "/favicon.ico":
                self._send_bytes(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
                return

            static_files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            }
            item = static_files.get(parsed.path)
            if item is None:
                self._error(HTTPStatus.NOT_FOUND, "route not found")
                return
            filename, content_type = item
            path = WEB_ROOT / filename
            if not path.is_file():
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "operator UI is not installed")
                return
            self._send_bytes(path.read_bytes(), content_type)

        def _frame(self, query: dict[str, list[str]]) -> None:
            try:
                dataset = query.get("dataset", ["llvip"])[0]
                split = query.get("split", ["train"])[0]
                index = int(query.get("index", ["0"])[0])
                thermal_gain = float(query.get("thermal_gain", ["1.0"])[0])
                automatic_text = query.get("automatic_control", ["false"])[0].strip().lower()
                if automatic_text not in {"true", "false"}:
                    raise ValueError("automatic_control must be true or false")
                payload = application.frame_payload(
                    dataset,
                    split,
                    index,
                    round(thermal_gain, 2),
                    automatic_text == "true",
                )
            except KeyError:
                self._error(HTTPStatus.BAD_REQUEST, "unknown dataset or split")
                return
            except (ValueError, IndexError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except (OSError, RuntimeError) as exc:
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
                return
            self._send_json(payload)

        def _ai_context(self, query: dict[str, list[str]]) -> None:
            try:
                dataset = query.get("dataset", ["llvip"])[0]
                split = query.get("split", ["train"])[0]
                index = int(query.get("index", ["0"])[0])
                payload = application.ai_context_payload(dataset, split, index)
            except KeyError:
                self._error(HTTPStatus.BAD_REQUEST, "unknown dataset or split")
                return
            except (ValueError, IndexError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except (OSError, RuntimeError) as exc:
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
                return
            self._send_json(payload)

    return OperatorHandler


def make_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    atlas_root: Path | str = DEFAULT_ATLAS_ROOT,
) -> ThreadingHTTPServer:
    application = OperatorApplication(data_root, atlas_root)
    return ThreadingHTTPServer((host, port), build_handler(application))


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    atlas_root: Path | str = DEFAULT_ATLAS_ROOT,
) -> None:
    server = make_server(
        host=host,
        port=port,
        data_root=data_root,
        atlas_root=atlas_root,
    )
    print(f"OpenPRISM operator canvas: http://{host}:{server.server_port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
