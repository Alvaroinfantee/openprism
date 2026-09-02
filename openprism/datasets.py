"""Adapters from the staged research datasets into ``PrismFrame`` evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .contracts import Detection, PrismFrame, SensorObservation, Timestamp


# Resolve archives relative to the directory in which the operator starts the
# application.  This remains correct both from a source checkout and after a
# wheel installation; package-relative defaults would otherwise point inside
# ``site-packages``.
DEFAULT_DATA_ROOT = Path.cwd() / "data"

MSRS_CLASSES = {
    0: "unlabelled",
    1: "car",
    2: "person",
    3: "bike",
    4: "curve",
    5: "car stop",
    6: "guardrail",
    7: "cone",
    8: "bump",
}

CALTECH_CLASSES = {
    0: "unknown",
    1: "background",
    2: "bare ground",
    3: "rocky terrain",
    4: "developed structures",
    5: "road",
    6: "shrubs",
    7: "trees",
    8: "sky",
    9: "water",
    10: "vehicles",
    11: "person",
}


@dataclass(frozen=True, slots=True)
class SampleRecord:
    dataset: str
    split: str
    sample_id: str
    visible_path: Path
    thermal_path: Path
    thermal16_path: Path | None = None
    annotation_path: Path | None = None
    annotation_kind: str = "none"
    scene_group: str | None = None


def _image(path: Path, mode: str | None = None) -> np.ndarray:
    # Pillow identifies image content from the payload. This matters because the
    # Caltech archive contains PNG payloads with .jpg filenames.
    with Image.open(path) as source:
        if mode is not None:
            source = source.convert(mode)
        source.load()
        return np.array(source)


def _caltech_id(stem: str) -> str:
    return re.sub(r"_(?:eo|thermal|mask)-", "_frame-", stem)


def _llvip_detections(path: Path | None, width: int, height: int) -> tuple[Detection, ...]:
    if path is None or not path.is_file():
        return ()
    root = ET.parse(path).getroot()
    detections: list[Detection] = []
    for item in root.findall("object"):
        box = item.find("bndbox")
        if box is None:
            continue
        try:
            xmin = max(0.0, float(box.findtext("xmin", "0")))
            ymin = max(0.0, float(box.findtext("ymin", "0")))
            xmax = min(float(width), float(box.findtext("xmax", str(width))))
            ymax = min(float(height), float(box.findtext("ymax", str(height))))
        except ValueError:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        detections.append(
            Detection(
                label=item.findtext("name", "object"),
                confidence=1.0,
                x=xmin / width,
                y=ymin / height,
                width=(xmax - xmin) / width,
                height=(ymax - ymin) / height,
                source="dataset_ground_truth",
            )
        )
    return tuple(detections)


def _msrs_detections(path: Path | None) -> tuple[Detection, ...]:
    if path is None or not path.is_file():
        return ()
    labels = ("person", "bicycle", "car")
    detections: list[Detection] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            class_id = int(fields[0])
            center_x, center_y, width, height = map(float, fields[1:])
        except ValueError:
            continue
        x = max(0.0, center_x - width / 2.0)
        y = max(0.0, center_y - height / 2.0)
        width = min(width, 1.0 - x)
        height = min(height, 1.0 - y)
        if width <= 0.0 or height <= 0.0:
            continue
        detections.append(
            Detection(
                label=labels[class_id] if 0 <= class_id < len(labels) else f"class {class_id}",
                confidence=1.0,
                x=x,
                y=y,
                width=width,
                height=height,
                source="dataset_ground_truth",
            )
        )
    return tuple(detections)


class DatasetCatalog:
    """Index LLVIP, MSRS, and Caltech without copying their source files."""

    labels = {
        "llvip": "LLVIP",
        "msrs": "MSRS",
        "caltech": "Caltech Aerial RGB-T",
    }

    split_labels = {
        "train": "Train",
        "test": "Test",
        "detection": "Detection",
        "all": "All labeled pairs",
    }

    def __init__(self, data_root: Path | str = DEFAULT_DATA_ROOT) -> None:
        self.data_root = Path(data_root).resolve()
        self._records: dict[tuple[str, str], tuple[SampleRecord, ...]] = {}
        self._index()

    def _index(self) -> None:
        self._index_llvip()
        self._index_msrs()
        self._index_caltech()
        if not self._records:
            raise FileNotFoundError(f"no supported datasets found under {self.data_root}")

    def _index_llvip(self) -> None:
        root = self.data_root / "LLVIP"
        for split in ("train", "test"):
            visible_dir = root / "visible" / split
            thermal_dir = root / "infrared" / split
            if not visible_dir.is_dir() or not thermal_dir.is_dir():
                continue
            thermal = {path.stem: path for path in thermal_dir.iterdir() if path.is_file()}
            records = []
            for visible_path in sorted(path for path in visible_dir.iterdir() if path.is_file()):
                thermal_path = thermal.get(visible_path.stem)
                if thermal_path is None:
                    continue
                annotation = root / "Annotations" / f"{visible_path.stem}.xml"
                records.append(
                    SampleRecord(
                        dataset="llvip",
                        split=split,
                        sample_id=visible_path.stem,
                        visible_path=visible_path,
                        thermal_path=thermal_path,
                        annotation_path=annotation if annotation.is_file() else None,
                        annotation_kind="voc_boxes",
                    )
                )
            self._records[("llvip", split)] = tuple(records)

    def _index_msrs(self) -> None:
        root = self.data_root / "MSRS"
        for split in ("train", "test"):
            visible_dir = root / split / "vi"
            thermal_dir = root / split / "ir"
            mask_dir = root / split / "Segmentation_labels"
            if not visible_dir.is_dir() or not thermal_dir.is_dir():
                continue
            thermal = {path.stem: path for path in thermal_dir.iterdir() if path.is_file()}
            masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
            self._records[("msrs", split)] = tuple(
                SampleRecord(
                    dataset="msrs",
                    split=split,
                    sample_id=visible_path.stem,
                    visible_path=visible_path,
                    thermal_path=thermal[visible_path.stem],
                    annotation_path=masks.get(visible_path.stem),
                    annotation_kind="semantic_mask",
                )
                for visible_path in sorted(path for path in visible_dir.iterdir() if path.is_file())
                if visible_path.stem in thermal
            )

        visible_dir = root / "detection" / "vi"
        thermal_dir = root / "detection" / "ir"
        labels_dir = root / "detection" / "labels"
        if visible_dir.is_dir() and thermal_dir.is_dir():
            thermal = {path.stem: path for path in thermal_dir.iterdir() if path.is_file()}
            self._records[("msrs", "detection")] = tuple(
                SampleRecord(
                    dataset="msrs",
                    split="detection",
                    sample_id=visible_path.stem,
                    visible_path=visible_path,
                    thermal_path=thermal[visible_path.stem],
                    annotation_path=(labels_dir / f"{visible_path.stem}.txt"),
                    annotation_kind="yolo_boxes",
                )
                for visible_path in sorted(path for path in visible_dir.iterdir() if path.is_file())
                if visible_path.stem in thermal
            )

    def _index_caltech(self) -> None:
        root = self.data_root / "Caltech_Aerial_RGBT"
        visible_dir = root / "color"
        if not visible_dir.is_dir():
            return

        def keyed(directory: str) -> dict[str, Path]:
            path = root / directory
            return {
                _caltech_id(item.stem): item
                for item in path.iterdir()
                if item.is_file()
            }

        thermal8 = keyed("thermal8")
        thermal16 = keyed("thermal16")
        annotations = keyed("annotations")
        records = []
        for visible_path in sorted(path for path in visible_dir.iterdir() if path.is_file()):
            sample_id = _caltech_id(visible_path.stem)
            if sample_id not in thermal8:
                continue
            scene_group = re.sub(r"_frame-\d+$", "", sample_id)
            records.append(
                SampleRecord(
                    dataset="caltech",
                    split="all",
                    sample_id=sample_id,
                    visible_path=visible_path,
                    thermal_path=thermal8[sample_id],
                    thermal16_path=thermal16.get(sample_id),
                    annotation_path=annotations.get(sample_id),
                    annotation_kind="semantic_mask",
                    scene_group=scene_group,
                )
            )
        self._records[("caltech", "all")] = tuple(records)

    def datasets(self) -> list[dict[str, object]]:
        result = []
        for dataset in self.labels:
            splits = []
            for (candidate, split), records in self._records.items():
                if candidate == dataset:
                    splits.append(
                        {
                            "id": split,
                            "label": self.split_labels.get(split, split.title()),
                            "count": len(records),
                        }
                    )
            if splits:
                result.append(
                    {"id": dataset, "label": self.labels[dataset], "splits": splits}
                )
        return result

    def count(self, dataset: str, split: str) -> int:
        return len(self._records[(dataset, split)])

    def record(self, dataset: str, split: str, index: int) -> SampleRecord:
        return self._records[(dataset, split)][index]

    def load(self, dataset: str, split: str, index: int) -> PrismFrame:
        record = self.record(dataset, split, index)
        visible_data = _image(record.visible_path, "RGB")
        thermal_data = _image(record.thermal_path)
        height, width = visible_data.shape[:2]
        if thermal_data.shape[:2] != (height, width):
            raise ValueError(f"unaligned geometry in {record.sample_id}")

        timestamp = Timestamp(
            tai_ns=None,
            clock_id=f"{dataset}_archive",
            uncertainty_ns=None,
        )
        observations: dict[str, SensorObservation] = {
            "visible": SensorObservation(
                sensor_id="visible",
                modality="visible_rgb",
                frame_id="visible_optical",
                timestamp=timestamp,
                data=visible_data,
                encoding="rgb8",
                units="srgb_code_value",
                source_path=record.visible_path,
                metadata={"content_sniffed": True},
            ),
            "thermal": SensorObservation(
                sensor_id="thermal",
                modality="thermal_lwir_8",
                frame_id="thermal_optical",
                timestamp=timestamp,
                data=thermal_data,
                encoding="mono8" if thermal_data.ndim == 2 else "rgb8_grayscale",
                units="display_code_value",
                source_path=record.thermal_path,
                metadata={"content_sniffed": True, "temperature_calibrated": False},
            ),
        }
        if record.thermal16_path is not None:
            thermal16 = _image(record.thermal16_path)
            if thermal16.dtype != np.uint16:
                raise ValueError(f"expected uint16 thermal data: {record.thermal16_path}")
            observations["thermal16"] = SensorObservation(
                sensor_id="thermal16",
                modality="thermal_lwir_16",
                frame_id="thermal_optical",
                timestamp=timestamp,
                data=thermal16,
                encoding="mono16",
                units="raw_sensor_count",
                source_path=record.thermal16_path,
                metadata={"temperature_calibrated": False},
            )

        detections: tuple[Detection, ...] = ()
        semantic_mask = None
        semantic_classes: dict[int, str] = {}
        if record.annotation_kind == "voc_boxes":
            detections = _llvip_detections(record.annotation_path, width, height)
        elif record.annotation_kind == "yolo_boxes":
            detections = _msrs_detections(record.annotation_path)
        elif record.annotation_kind == "semantic_mask" and record.annotation_path:
            semantic_mask = _image(record.annotation_path)
            if semantic_mask.ndim == 3:
                semantic_mask = semantic_mask[..., 0]
            semantic_classes = CALTECH_CLASSES if dataset == "caltech" else MSRS_CLASSES

        return PrismFrame(
            frame_id=f"{dataset}:{split}:{record.sample_id}",
            timestamp=timestamp,
            reference_frame="visible_optical",
            observations=observations,
            transforms={"thermal_optical->visible_optical": np.eye(3)},
            detections=detections,
            semantic_mask=semantic_mask,
            semantic_classes=semantic_classes,
            provenance={
                "dataset": dataset,
                "split": split,
                "sample_id": record.sample_id,
                "scene_group": record.scene_group,
                "alignment": "publisher_provided_rectification",
                "annotation_source": (
                    "dataset_ground_truth"
                    if record.annotation_kind != "none"
                    else "none"
                ),
            },
        )
