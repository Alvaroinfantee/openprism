"""Run reviewed external fusion code without vendoring it into OpenPRISM.

Only the model definition file is imported.  The external repository's test
script is not executed, and checkpoint deserialization uses ``weights_only``.
Every output is accompanied by a content-addressed run manifest.
"""

from __future__ import annotations

import argparse
import ast
from collections import namedtuple
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import ModuleType
from typing import Callable, Iterable

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load model definition from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _selected_classes_module(path: Path, name: str, classes: set[str]) -> ModuleType:
    """Execute exact selected class nodes from a reviewed upstream module."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "layers_fusion"
        )
        or (isinstance(node, ast.ClassDef) and node.name in classes)
    ]
    found = {node.name for node in selected if isinstance(node, ast.ClassDef)}
    if found != classes:
        raise RuntimeError(f"missing reviewed classes in {path}: {sorted(classes - found)}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    code = compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec")
    exec(code, module.__dict__)
    return module


def _image(path: Path, mode: str) -> Tensor:
    with Image.open(path) as source:
        value = np.asarray(source.convert(mode), dtype=np.float32) / 255.0
    if value.ndim == 2:
        value = value[..., None]
    return torch.from_numpy(np.moveaxis(value, -1, 0).copy())[None]


def _minmax(value: Tensor) -> Tensor:
    return (value - value.min()) / (value.max() - value.min()).clamp_min(1e-8)


def _save(value: Tensor, path: Path) -> None:
    array = value.detach().float().cpu().clamp(0.0, 1.0)[0].numpy()
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    image = (np.moveaxis(array, 0, -1) * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def _seafusion(
    source_root: Path, weights: Path, device: torch.device
) -> tuple[Callable[[Tensor, Tensor], Tensor], dict[str, object]]:
    module = _module(source_root / "FusionNet.py", "openprism_external_seafusion")
    model = module.FusionNet(output=1).to(device).eval()
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)

    def infer(visible: Tensor, thermal: Tensor) -> Tensor:
        visible = visible.to(device)
        thermal = thermal.to(device)
        red, green, blue = visible[:, 0:1], visible[:, 1:2], visible[:, 2:3]
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        cb = ((blue - luminance) * 0.564 + 0.5).clamp(0.0, 1.0)
        cr = ((red - luminance) * 0.713 + 0.5).clamp(0.0, 1.0)
        fused = model(luminance, thermal)
        red_out = fused + 1.403 * (cr - 0.5)
        green_out = fused - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
        blue_out = fused + 1.773 * (cb - 0.5)
        # The official save path applies global min--max normalization to the
        # reconstructed RGB tensor. Reproduce that declared behavior exactly.
        reconstructed = torch.cat((red_out, green_out, blue_out), dim=1)
        return _minmax(reconstructed.clamp(0.0, 1.0))

    return infer, {
        "model_definition": "FusionNet.py:FusionNet",
        "preprocessing": "RGB/IR uint8 to [0,1]; RGB->Y; official RGB clamp and global output minmax; lossless PNG",
    }


def _load_parallel(module: nn.Module, state: dict[str, Tensor]) -> nn.Module:
    wrapped = nn.DataParallel(module)
    wrapped.load_state_dict(state)
    return wrapped


def _cddfuse(
    source_root: Path, weights: Path, device: torch.device
) -> tuple[Callable[[Tensor, Tensor], Tensor], dict[str, object]]:
    module = _module(source_root / "net.py", "openprism_external_cddfuse")
    checkpoint = torch.load(weights, map_location=device, weights_only=True)
    # The official checkpoint uses the historical DIDF key names while its
    # current test script requests CDDF keys. Accept either, and record which
    # schema was used rather than editing the external repository.
    encoder_key = "CDDF_Encoder" if "CDDF_Encoder" in checkpoint else "DIDF_Encoder"
    decoder_key = "CDDF_Decoder" if "CDDF_Decoder" in checkpoint else "DIDF_Decoder"
    encoder = _load_parallel(module.Restormer_Encoder(), checkpoint[encoder_key]).to(device).eval()
    decoder = _load_parallel(module.Restormer_Decoder(), checkpoint[decoder_key]).to(device).eval()
    base = _load_parallel(
        module.BaseFeatureExtraction(dim=64, num_heads=8), checkpoint["BaseFuseLayer"]
    ).to(device).eval()
    detail = _load_parallel(
        module.DetailFeatureExtraction(num_layers=1), checkpoint["DetailFuseLayer"]
    ).to(device).eval()

    def infer(visible: Tensor, thermal: Tensor) -> Tensor:
        visible_y = (
            0.299 * visible[:, 0:1]
            + 0.587 * visible[:, 1:2]
            + 0.114 * visible[:, 2:3]
        ).to(device)
        thermal = thermal.to(device)
        visible_base, visible_detail, _ = encoder(visible_y)
        thermal_base, thermal_detail, _ = encoder(thermal)
        fused_base = base(visible_base + thermal_base)
        fused_detail = detail(visible_detail + thermal_detail)
        fused, _ = decoder(visible_y, fused_base, fused_detail)
        return _minmax(fused).repeat(1, 3, 1, 1)

    return infer, {
        "model_definition": "net.py:Restormer_Encoder/Decoder + Base/DetailFeatureExtraction",
        "checkpoint_key_schema": f"{encoder_key}/{decoder_key}",
        "preprocessing": "RGB/IR uint8 to [0,1]; RGB->grayscale luminance; official output minmax",
    }


def _paif_module(source_root: Path) -> ModuleType:
    """Load PAIF's fusion module while isolating unused missing dependencies.

    The published checkout imports its segmentation stack at module load and
    references an ``antialias`` helper that is absent from the repository.
    Neither path is reached by the released fusion genotype.  Temporary stubs
    therefore fail loudly if they are ever used, while the fusion definitions
    and operations continue to execute directly from the pinned checkout.
    """

    class _Unavailable(nn.Module):
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("PAIF adapter reached an unavailable, unused dependency")

    package = ModuleType("core")
    package.__path__ = [str(source_root / "core")]  # type: ignore[attr-defined]
    segmentation = ModuleType("core.segformer_head")
    segmentation.SegFormerHead = _Unavailable  # type: ignore[attr-defined]
    transformer = ModuleType("core.mix_transformer")
    antialias = ModuleType("antialias")
    antialias.Downsample = _Unavailable  # type: ignore[attr-defined]
    temporary = {
        "core": package,
        "core.segformer_head": segmentation,
        "core.mix_transformer": transformer,
        "antialias": antialias,
    }
    previous = {name: sys.modules.get(name) for name in temporary}
    sys.path.insert(0, str(source_root))
    sys.modules.update(temporary)
    try:
        return _module(
            source_root / "core" / "model_fusion_auto.py",
            "core.model_fusion_auto",
        )
    finally:
        sys.path.remove(str(source_root))
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _paif(
    source_root: Path, weights: Path, device: torch.device
) -> tuple[Callable[[Tensor, Tensor], Tensor], dict[str, object]]:
    module = _paif_module(source_root)
    genotype_type = namedtuple(
        "Genotype",
        "normal_1 normal_1_concat normal_2 normal_2_concat normal_3 normal_3_concat",
    )
    genotype = genotype_type(
        normal_1=[("Denseblocks_3_1", 0), ("DilConv_3_2", 1)],
        normal_1_concat=[1, 2],
        normal_2=[("Denseblocks_3_1", 0), ("Denseblocks_3_1", 1)],
        normal_2_concat=[1, 2],
        normal_3=[("ECAattention_3", 0), ("Residualblocks_7_1", 1)],
        normal_3_concat=[1, 2],
    )
    model = module.Network_Fusion_Searched(32, None, genotype).to(device).eval()
    checkpoint = torch.load(weights, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("PAIF checkpoint must be a state dictionary")
    prefix = "enhance_net."
    fusion_state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }
    if not fusion_state:
        raise ValueError("PAIF checkpoint contains no enhance_net fusion weights")
    model.load_state_dict(fusion_state, strict=True)

    def infer(visible: Tensor, thermal: Tensor) -> Tensor:
        visible = visible.to(device)
        thermal = thermal.to(device)
        red, green, blue = visible[:, 0:1], visible[:, 1:2], visible[:, 2:3]
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        cr = (red - luminance) * 0.713 + 0.5
        cb = (blue - luminance) * 0.564 + 0.5
        fused = model(thermal, luminance)
        red_out = fused + 1.403 * (cr - 0.5)
        green_out = fused - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
        blue_out = fused + 1.773 * (cb - 0.5)
        reconstructed = torch.cat((red_out, green_out, blue_out), dim=1).clamp(0.0, 1.0)
        # Match the official script's uint8 conversion before global min--max.
        quantized = torch.floor(reconstructed * 255.0) / 255.0
        return _minmax(quantized)

    return infer, {
        "model_definition": "core/model_fusion_auto.py:Network_Fusion_Searched",
        "checkpoint_key_schema": "enhance_net.* (fusion subnetwork only)",
        "genotype_source": "test_original.py",
        "preprocessing": "RGB/IR uint8 to [0,1]; RGB->YCrCb; official RGB clamp, uint8 quantization, and global output minmax; lossless PNG",
        "adapter_note": "unused missing antialias and segmentation imports are fail-closed stubs",
    }


def _c2rf(
    source_root: Path, weights: Path, device: torch.device
) -> tuple[Callable[[Tensor, Tensor], Tensor], dict[str, object]]:
    if not weights.is_dir():
        raise ValueError("C2RF requires --weights to name its four-file checkpoint directory")
    if device.type != "cuda":
        raise ValueError("the audited C2RF revision hard-codes CUDA tensors")

    package = ModuleType("modules")
    package.__path__ = [str(source_root / "modules")]  # type: ignore[attr-defined]
    previous = {name: sys.modules.get(name) for name in ("modules",)}
    sys.path.insert(0, str(source_root))
    sys.modules["modules"] = package
    try:
        fusion_module = _selected_classes_module(
            source_root / "modules" / "FusionNet.py",
            "modules.FusionNet_selected",
            {"AE_Encoder", "AE_Decoder", "Fusion_layer"},
        )
        registration_module = _module(
            source_root / "modules" / "RegNet.py",
            "modules.RegNet",
        )
    finally:
        sys.path.remove(str(source_root))
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old

    encoder = fusion_module.AE_Encoder().to(device).eval()
    decoder = fusion_module.AE_Decoder().to(device).eval()
    fusion = fusion_module.Fusion_layer().to(device).eval()
    registration = registration_module.registration_net().to(device).eval()
    transformer = registration_module.SpatialTransformer(256, 256, True).to(device).eval()
    components = {
        "Encoder.pth": encoder,
        "Decoder.pth": decoder,
        "Fusion_layer.pth": fusion,
        "RegNet.pth": registration,
    }
    for filename, component in components.items():
        checkpoint = weights / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        component.load_state_dict(state, strict=True)

    def infer(visible: Tensor, thermal: Tensor) -> Tensor:
        visible = visible.to(device)
        red, green, blue = visible[:, 0:1], visible[:, 1:2], visible[:, 2:3]
        visible_y = (0.299 * red + 0.587 * green + 0.114 * blue).clamp(0.0, 1.0)
        infrared_y = thermal.to(device)
        (
            _,
            infrared_level_2,
            infrared_base,
            infrared_detail,
            _,
            visible_level_2,
            visible_base,
            visible_detail,
        ) = encoder(infrared_y, visible_y)
        deformation = registration(infrared_base, visible_base, type="bi")
        displacement = deformation["vis2ir"]
        visible_registered, _ = transformer(visible, displacement)
        visible_level_2, _ = transformer(visible_level_2, displacement)
        visible_base, _ = transformer(visible_base, displacement)
        visible_detail, _ = transformer(visible_detail, displacement)
        common, detail, cross = fusion(
            infrared_base,
            visible_base,
            infrared_detail,
            visible_detail,
            infrared_level_2,
            visible_level_2,
        )
        fused_y = decoder(common, detail, cross)
        registered_red = visible_registered[:, 0:1]
        registered_green = visible_registered[:, 1:2]
        registered_blue = visible_registered[:, 2:3]
        registered_y = (
            0.299 * registered_red
            + 0.587 * registered_green
            + 0.114 * registered_blue
        )
        registered_cr = ((registered_red - registered_y) * 0.713 + 0.5).clamp(0.0, 1.0)
        registered_cb = ((registered_blue - registered_y) * 0.564 + 0.5).clamp(0.0, 1.0)
        red_out = fused_y + 1.403 * (registered_cr - 0.5)
        green_out = fused_y - 0.714 * (registered_cr - 0.5) - 0.344 * (registered_cb - 0.5)
        blue_out = fused_y + 1.773 * (registered_cb - 0.5)
        return torch.cat((red_out, green_out, blue_out), dim=1).clamp(0.0, 1.0)

    return infer, {
        "model_definition": "selected exact AE_Encoder/AE_Decoder/Fusion_layer nodes from modules/FusionNet.py plus modules/RegNet.py",
        "checkpoint_key_schema": "Encoder.pth, Decoder.pth, Fusion_layer.pth, RegNet.pth",
        "preprocessing": "RGB/IR uint8 to [0,1]; aligned visible is supplied as vi_warp and chroma is taken from the registered visible output; lossless PNG",
        "adapter_note": "unused VGG contrastive-training classes are not executed; audited code requires CUDA",
    }


_LOADERS = {
    "seafusion": _seafusion,
    "cddfuse": _cddfuse,
    "paif": _paif,
    "c2rf": _c2rf,
}


def _weight_manifest(path: Path) -> tuple[str, dict[str, str]]:
    if path.is_file():
        digest = _sha256(path)
        return digest, {path.name: digest}
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {path}")
    members = {candidate.relative_to(path).as_posix(): _sha256(candidate) for candidate in files}
    aggregate = hashlib.sha256()
    for relative, digest in members.items():
        aggregate.update(f"{relative}\0{digest}\n".encode("utf-8"))
    return aggregate.hexdigest(), members


def run_external_fusion(
    baseline: str,
    source_root: Path,
    weights: Path,
    visible_dir: Path,
    thermal_dir: Path,
    output_dir: Path,
    *,
    expected_revision: str,
    expected_weights_sha256: str | None = None,
    device_name: str = "auto",
    overwrite: bool = False,
) -> dict[str, object]:
    if baseline not in _LOADERS:
        raise ValueError(f"unsupported external baseline: {baseline}")
    revision = _revision(source_root)
    if revision != expected_revision:
        raise ValueError(
            f"external revision mismatch: expected {expected_revision}, found {revision}"
        )
    weights_sha256, weight_files = _weight_manifest(weights)
    if (
        expected_weights_sha256 is not None
        and weights_sha256 != expected_weights_sha256.lower()
    ):
        raise ValueError(
            "checkpoint checksum mismatch: expected "
            f"{expected_weights_sha256.lower()}, found {weights_sha256}"
        )
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_dir.iterdir() if path.is_file()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --overwrite explicitly"
        )

    visible_paths = {
        path.stem: path
        for path in visible_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    }
    thermal_paths = {
        path.stem: path
        for path in thermal_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    }
    if set(visible_paths) != set(thermal_paths) or not visible_paths:
        raise ValueError("visible and thermal directories must contain the same non-empty ID set")

    torch.manual_seed(20260902)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260902)
    torch.use_deterministic_algorithms(True, warn_only=True)
    infer, adapter = _LOADERS[baseline](source_root, weights, device)
    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for sample_id in sorted(visible_paths):
            visible = _image(visible_paths[sample_id], "RGB")
            thermal = _image(thermal_paths[sample_id], "L")
            if visible.shape[2:] != thermal.shape[2:]:
                raise ValueError(f"unaligned geometry for {sample_id}")
            before = time.perf_counter()
            fused = infer(visible, thermal)
            destination = output_dir / f"{sample_id}.png"
            _save(fused, destination)
            outputs.append({
                "sample_id": sample_id,
                "path": destination.name,
                "sha256": _sha256(destination),
                "elapsed_seconds": time.perf_counter() - before,
            })
    report: dict[str, object] = {
        "schema_version": "openprism.external-fusion-run/1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "external_outputs_require_downstream_evaluation",
        "baseline": baseline,
        "repository": str(source_root.resolve()),
        "revision": revision,
        "weights": str(weights.resolve()),
        "weights_sha256": weights_sha256,
        "weight_files": weight_files,
        "adapter_source_sha256": _sha256(Path(__file__)),
        "adapter": adapter,
        "inputs": {
            "visible_directory": str(visible_dir.resolve()),
            "thermal_directory": str(thermal_dir.resolve()),
            "paired_ids_sorted": sorted(visible_paths),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
        },
        "input_count": len(visible_paths),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": outputs,
    }
    manifest = output_dir / "run_manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=tuple(_LOADERS), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--visible-dir", type=Path, required=True)
    parser.add_argument("--thermal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-weights-sha256")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = run_external_fusion(
        args.baseline,
        args.source_root,
        args.weights,
        args.visible_dir,
        args.thermal_dir,
        args.output_dir,
        expected_revision=args.expected_revision,
        expected_weights_sha256=args.expected_weights_sha256,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "baseline": report["baseline"],
        "revision": report["revision"],
        "weights_sha256": report["weights_sha256"],
        "input_count": report["input_count"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
