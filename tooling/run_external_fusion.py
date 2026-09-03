"""Run reviewed external fusion code without vendoring it into OpenPRISM.

Only the model definition file is imported.  The external repository's test
script is not executed, and checkpoint deserialization uses ``weights_only``.
Every output is accompanied by a content-addressed run manifest.
The manifest also records globally de-duplicated inference-parameter counts,
exact checkpoint bytes, and every per-sample failure. Validation failures exit
non-zero after an attempt-complete manifest is persisted. During the
irreversibly claimed final suite, fully accounted per-sample failures return
control to the controller so unchanged downstream evaluators can record them
as scientific outcomes; integrity or process failures still terminate the
suite.
"""

from __future__ import annotations

import argparse
import ast
from collections import namedtuple
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
from pathlib import PurePosixPath
import re
import stat
import subprocess
import sys
import time
from types import ModuleType
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import PIL
from PIL import Image
import torch
from torch import Tensor, nn


_FINAL_LEDGER_SCHEMA = "openprism.final-test-ledger/1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _git_bytes(source_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _decode_git_path(value: bytes) -> str:
    return value.decode("utf-8", errors="surrogateescape").replace("\\", "/")


def _status_entries(source_root: Path) -> tuple[tuple[str, str], ...]:
    """Return unquoted Git porcelain entries, including ignored files.

    Rename/copy records have a second NUL-terminated path.  They are retained
    as one entry because every tracked rename/copy is rejected by the source
    attestation regardless of either path's contents.
    """

    payload = _git_bytes(
        source_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
    )
    fields = payload.split(b"\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(fields) and fields[index]:
        record = fields[index]
        index += 1
        if len(record) < 4 or record[2:3] != b" ":
            raise RuntimeError("unexpected git status porcelain record")
        status = record[:2].decode("ascii", errors="strict")
        path = _decode_git_path(record[3:])
        entries.append((status, path))
        if "R" in status or "C" in status:
            if index >= len(fields) or not fields[index]:
                raise RuntimeError("truncated git rename/copy status record")
            index += 1
    return tuple(entries)


def _tracked_paths(source_root: Path) -> frozenset[str]:
    values = _git_bytes(source_root, "ls-files", "-z").split(b"\0")
    return frozenset(_decode_git_path(value) for value in values if value)


def _is_safe_python_cache(path: str) -> bool:
    candidate = PurePosixPath(path.rstrip("/"))
    return "__pycache__" in candidate.parts and (
        candidate.name == "__pycache__" or candidate.suffix in {".pyc", ".pyo"}
    )


def _case_collision_groups(tracked: Iterable[str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for path in tracked:
        grouped.setdefault(path.casefold(), []).append(path)
    return {
        key: tuple(sorted(paths))
        for key, paths in grouped.items()
        if len(set(paths)) > 1
    }


def _documentation_collision_exceptions(
    tracked: frozenset[str], requested: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Validate narrow, explicit case-collision exceptions for documentation.

    Some upstream repositories track both ``README.md`` and ``readme.md``.
    Windows cannot materialize both paths, so Git necessarily reports one as
    dirty.  Only explicitly named, non-code documentation groups are eligible;
    broad dirty-tree waivers are intentionally unsupported.
    """

    collisions = _case_collision_groups(tracked)
    allowed: dict[str, tuple[str, ...]] = {}
    for raw in requested:
        normalized = str(raw).strip().replace("\\", "/").strip("/")
        if not normalized or normalized not in tracked:
            raise ValueError(
                f"case-collision exception is not an exact tracked path: {raw!r}"
            )
        group = collisions.get(normalized.casefold())
        if group is None:
            raise ValueError(
                f"case-collision exception has no differently-cased tracked peer: {normalized}"
            )
        if any(PurePosixPath(path).suffix.lower() not in {".md", ".rst", ".txt"} for path in group):
            raise ValueError(
                f"case-collision exception is not documentation-only: {list(group)}"
            )
        allowed[normalized.casefold()] = group
    return allowed


def _source_worktree_attestation(
    source_root: Path,
    *,
    allowed_case_collision_paths: Sequence[str] = (),
) -> tuple[dict[str, object], frozenset[str]]:
    """Fail closed unless the exact Git tree is clean and reproducible."""

    root = source_root.resolve()
    top_level = Path(
        _git_bytes(root, "rev-parse", "--show-toplevel")
        .decode("utf-8", errors="surrogateescape")
        .strip()
    ).resolve()
    if root != top_level:
        raise ValueError(
            f"source_root must be the Git worktree root: expected {top_level}, got {root}"
        )
    revision = _revision(root)
    tracked = _tracked_paths(root)
    collision_groups = _documentation_collision_exceptions(
        tracked, allowed_case_collision_paths
    )
    safe_cache: list[dict[str, str]] = []
    allowed_collisions: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    used_collision_groups: set[str] = set()
    for status, path in _status_entries(root):
        if status in {"??", "!!"} and _is_safe_python_cache(path):
            safe_cache.append({"status": status, "path": path})
            continue
        group = collision_groups.get(path.casefold())
        if group is not None:
            if status not in {" M", " D"}:
                rejected.append({"status": status, "path": path})
                continue
            materialized = root / PurePosixPath(path)
            if not materialized.is_file():
                rejected.append({"status": status, "path": path})
                continue
            working_hash = _sha256(materialized)
            matching_peers = []
            for peer in group:
                head_blob = _git_bytes(root, "rev-parse", f"HEAD:{peer}").strip()
                materialized_blob = _git_bytes(
                    root,
                    "hash-object",
                    "--filters",
                    f"--path={peer}",
                    str(materialized),
                ).strip()
                if materialized_blob == head_blob:
                    matching_peers.append(peer)
            if not matching_peers:
                rejected.append({"status": status, "path": path})
                continue
            used_collision_groups.add(path.casefold())
            allowed_collisions.append(
                {
                    "status": status,
                    "path": path,
                    "tracked_case_peers": list(group),
                    "materialized_sha256": working_hash,
                    "matches_head_blob_for": matching_peers,
                }
            )
            continue
        rejected.append({"status": status, "path": path})
    unused = sorted(set(collision_groups) - used_collision_groups)
    if unused:
        raise ValueError(
            "case-collision exception was requested but no matching dirty path exists: "
            + ", ".join(collision_groups[key][0] for key in unused)
        )
    if rejected:
        rendered = ", ".join(f"{item['status']} {item['path']}" for item in rejected)
        raise ValueError(f"external source worktree is not clean: {rendered}")
    return (
        {
            "git_worktree_root": str(root),
            "revision": revision,
            "policy": (
                "tracked tree clean except explicit documentation-only NTFS case collisions; "
                "only ignored/untracked Python bytecode caches allowed"
            ),
            "safe_python_cache_entries": safe_cache,
            "documented_case_collision_exceptions": allowed_collisions,
        },
        tracked,
    )


_ACTIVE_UPSTREAM_SOURCES: ContextVar[set[Path] | None] = ContextVar(
    "openprism_active_upstream_sources", default=None
)


def _record_upstream_source(path: Path) -> None:
    active = _ACTIVE_UPSTREAM_SOURCES.get()
    if active is not None:
        active.add(path.resolve())


@contextmanager
def _track_upstream_source_execution(source_root: Path) -> Iterator[set[Path]]:
    """Compile upstream Python directly from source and record every file.

    Ignoring bytecode here makes the permitted ``__pycache__`` exception safe:
    no untracked cached bytecode under the upstream worktree is executed.
    """

    root = source_root.resolve()
    executed: set[Path] = set()
    token = _ACTIVE_UPSTREAM_SOURCES.set(executed)
    original_get_code = importlib.machinery.SourceFileLoader.get_code

    def source_only_get_code(
        loader: importlib.machinery.SourceFileLoader, fullname: str
    ) -> object:
        candidate = Path(loader.path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return original_get_code(loader, fullname)
        if candidate.suffix.lower() != ".py":
            raise RuntimeError(f"upstream import is not Python source: {candidate}")
        _record_upstream_source(candidate)
        source = loader.get_data(loader.path)
        return loader.source_to_code(source, loader.path)

    importlib.machinery.SourceFileLoader.get_code = source_only_get_code  # type: ignore[method-assign]
    try:
        yield executed
    finally:
        # Include already-cached dependencies from this worktree.  A CLI run
        # starts clean, but the public function can be called repeatedly in one
        # interpreter and must still attest dependencies reused from sys.modules.
        for module in tuple(sys.modules.values()):
            raw_path = getattr(module, "__file__", None)
            if not raw_path:
                continue
            candidate = Path(raw_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.suffix.lower() == ".py":
                executed.add(candidate)
        importlib.machinery.SourceFileLoader.get_code = original_get_code  # type: ignore[method-assign]
        _ACTIVE_UPSTREAM_SOURCES.reset(token)


def _executed_source_manifest(
    source_root: Path,
    executed: Iterable[Path],
    tracked: frozenset[str],
) -> dict[str, str]:
    root = source_root.resolve()
    manifest: dict[str, str] = {}
    for source in sorted({path.resolve() for path in executed}):
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"executed upstream source escapes worktree: {source}") from error
        if relative not in tracked:
            raise ValueError(f"executed upstream source is not tracked at the pinned revision: {relative}")
        if source.suffix.lower() != ".py" or not source.is_file():
            raise ValueError(f"executed upstream source is not a readable .py file: {relative}")
        manifest[relative] = _sha256(source)
    if not manifest:
        raise ValueError("external adapter did not attest any executed upstream source")
    return manifest


def _module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load model definition from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _selected_classes_module(path: Path, name: str, classes: set[str]) -> ModuleType:
    """Execute exact selected class nodes from a reviewed upstream module."""

    _record_upstream_source(path)
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


def _parameter_inventory(modules: dict[str, nn.Module]) -> dict[str, object]:
    """Count inference parameters once by ``nn.Parameter`` object identity.

    PyTorch permits one parameter object to be referenced by multiple modules
    (weight tying).  Summing component totals would double-count that storage.
    The aggregate therefore de-duplicates globally by Python object identity,
    while the per-component values use PyTorch's normal within-module
    de-duplication. Buffers are excluded from parameter counts.
    """

    if not modules:
        raise ValueError("at least one inference module is required for parameter inventory")
    seen: set[int] = set()
    total = 0
    trainable = 0
    unique_tensors = 0
    observed_references = 0
    components: dict[str, dict[str, int]] = {}
    for name, module in modules.items():
        if not name or not isinstance(module, nn.Module):
            raise TypeError("parameter inventory requires named torch modules")
        parameters = tuple(module.parameters())
        components[name] = {
            "total_parameters": int(sum(parameter.numel() for parameter in parameters)),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
            ),
            "parameter_tensors": len(parameters),
        }
        for parameter in parameters:
            observed_references += 1
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            unique_tensors += 1
            total += int(parameter.numel())
            if parameter.requires_grad:
                trainable += int(parameter.numel())
    return {
        "counting_policy": (
            "sum numel over unique nn.Parameter object identities across every "
            "inference module; tied/shared object references count once; buffers excluded"
        ),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "unique_parameter_tensors": unique_tensors,
        "observed_component_parameter_references": observed_references,
        "shared_parameter_references_deduplicated": observed_references - unique_tensors,
        "components": components,
    }


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
        "parameter_inventory": _parameter_inventory({"fusion": model}),
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
        "parameter_inventory": _parameter_inventory(
            {
                "encoder": encoder,
                "decoder": decoder,
                "base_fusion": base,
                "detail_fusion": detail,
            }
        ),
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
        "parameter_inventory": _parameter_inventory({"fusion": model}),
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
        "parameter_inventory": _parameter_inventory(
            {
                "encoder": encoder,
                "decoder": decoder,
                "fusion": fusion,
                "registration": registration,
                "spatial_transformer": transformer,
            }
        ),
    }


_LOADERS = {
    "seafusion": _seafusion,
    "cddfuse": _cddfuse,
    "paif": _paif,
    "c2rf": _c2rf,
}

_REVIEWED_UPSTREAM_FILES = {
    "seafusion": ("FusionNet.py",),
    "cddfuse": ("net.py",),
    # The genotype below is transcribed from this reviewed, non-executed script.
    "paif": ("core/model_fusion_auto.py", "test_original.py"),
    "c2rf": ("modules/FusionNet.py", "modules/RegNet.py"),
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


def _weight_byte_manifest(path: Path) -> tuple[int, dict[str, int]]:
    """Return exact checkpoint bytes using the same member policy as its hash."""

    if path.is_file():
        size = int(path.stat().st_size)
        return size, {path.name: size}
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {path}")
    members = {
        candidate.relative_to(path).as_posix(): int(candidate.stat().st_size)
        for candidate in files
    }
    return sum(members.values()), members


def _reviewed_source_manifest(
    baseline: str, source_root: Path, tracked: frozenset[str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _REVIEWED_UPSTREAM_FILES[baseline]:
        if relative not in tracked:
            raise ValueError(f"reviewed upstream source is not tracked: {relative}")
        path = source_root / PurePosixPath(relative)
        if not path.is_file():
            raise FileNotFoundError(f"reviewed upstream source is absent: {path}")
        result[relative] = _sha256(path)
    return result


def _runtime_environment(device: torch.device) -> dict[str, object]:
    cuda_device: dict[str, object] | None = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda_device = {
            "index": int(index),
            "name": properties.name,
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
        }
    distributions: dict[str, str | None] = {}
    for name in ("einops", "kornia", "torchvision"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    try:
        driver_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        driver_versions: list[str] | None = [
            value.strip()
            for value in driver_result.stdout.splitlines()
            if value.strip()
        ]
    except (FileNotFoundError, subprocess.CalledProcessError):
        driver_versions = None
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "torch": torch.__version__,
        "external_dependency_versions": distributions,
        "device": str(device),
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_device,
        "nvidia_driver_versions": driver_versions,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }


def _final_controller_authorization(
    *, baseline: str, dataset: str, output_dir: Path
) -> dict[str, object]:
    """Verify that a claimed one-shot controller is executing this exact run.

    The final-suite controller publishes its permanent ledger before starting a
    subprocess.  A test-partition adapter accepts authorization only when the
    ledger is a regular file, names the same canonical manifest digest, and has
    this exact step in the running state with the expected external-output
    contract.  This is an audit binding, not a secret-token mechanism.
    """

    environment = os.environ
    if environment.get("OPENPRISM_FINAL_SUITE") != "1":
        raise ValueError("test external fusion requires the one-shot final-suite controller")
    manifest_sha256 = environment.get("OPENPRISM_FINAL_SUITE_MANIFEST_SHA256", "")
    step_id = environment.get("OPENPRISM_FINAL_SUITE_STEP_ID", "")
    ledger_value = environment.get("OPENPRISM_FINAL_SUITE_LEDGER", "")
    controller_sha256 = environment.get("OPENPRISM_FINAL_SUITE_CONTROLLER_SHA256", "")
    if not _SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("final-suite manifest authorization digest is missing or malformed")
    if not _SHA256_RE.fullmatch(controller_sha256):
        raise ValueError("final-suite controller digest is missing or malformed")
    if not step_id or not ledger_value or not Path(ledger_value).is_absolute():
        raise ValueError("final-suite step/ledger authorization is incomplete")
    ledger_path = Path(ledger_value)
    try:
        metadata = ledger_path.lstat()
    except OSError as error:
        raise ValueError(f"cannot inspect final-suite ledger: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("final-suite ledger must be a non-symlink regular file")
    try:
        document = json.loads(ledger_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot decode final-suite ledger: {error}") from error
    if (
        type(document) is not dict
        or document.get("schema_version") != _FINAL_LEDGER_SCHEMA
        or document.get("manifest_canonical_sha256") != manifest_sha256
        or document.get("arming_confirmed") is not True
        or document.get("status") != "running"
    ):
        raise ValueError("final-suite ledger does not attest an armed running claim")
    controller = document.get("controller")
    if type(controller) is not dict or controller.get("sha256") != controller_sha256:
        raise ValueError("final-suite controller hash does not match its permanent ledger")
    steps = document.get("steps")
    if type(steps) is not list:
        raise ValueError("final-suite ledger has no step records")
    matching = [item for item in steps if type(item) is dict and item.get("id") == step_id]
    if len(matching) != 1 or matching[0].get("status") != "running":
        raise ValueError("the authorized external-fusion step is not uniquely running")
    expected_outputs = matching[0].get("expected_outputs")
    if type(expected_outputs) is not list:
        raise ValueError("the authorized step has no expected-output contract")
    resolved_output = output_dir.resolve()
    contracts = [
        item
        for item in expected_outputs
        if type(item) is dict
        and item.get("kind") == "external_fusion_directory"
        and item.get("required_schema_version") == "openprism.external-fusion-run/1.3"
        and item.get("required_baseline") == baseline
        and item.get("required_dataset") == dataset
        and type(item.get("path")) is str
        and Path(item["path"]).resolve() == resolved_output
    ]
    if len(contracts) != 1:
        raise ValueError("the running step does not authorize this exact external output")
    return {
        "persistent_one_shot_claim_verified": True,
        "manifest_canonical_sha256": manifest_sha256,
        "ledger_path": str(ledger_path.resolve()),
        "step_id": step_id,
        "controller_sha256": controller_sha256,
        "output_contract_id": contracts[0].get("id"),
    }


def run_external_fusion(
    baseline: str,
    source_root: Path,
    weights: Path,
    visible_dir: Path,
    thermal_dir: Path,
    output_dir: Path,
    *,
    dataset: str,
    partition: str = "validation",
    unlock_final_test: bool = False,
    expected_revision: str,
    expected_weights_sha256: str | None = None,
    device_name: str = "auto",
    overwrite: bool = False,
    allowed_case_collision_paths: Sequence[str] = (),
) -> dict[str, object]:
    if baseline not in _LOADERS:
        raise ValueError(f"unsupported external baseline: {baseline}")
    if dataset not in {"llvip", "msrs", "caltech"}:
        raise ValueError("dataset must be llvip, msrs, or caltech")
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be validation or test")
    if partition == "validation" and unlock_final_test:
        raise ValueError("validation external fusion must not use the final-test unlock")
    if partition == "test" and not unlock_final_test:
        raise ValueError("test external fusion requires --unlock-final-test")
    controller_authorization = (
        _final_controller_authorization(
            baseline=baseline, dataset=dataset, output_dir=output_dir
        )
        if partition == "test"
        else None
    )
    source_root = source_root.resolve()
    source_attestation, tracked_sources = _source_worktree_attestation(
        source_root,
        allowed_case_collision_paths=allowed_case_collision_paths,
    )
    revision = str(source_attestation["revision"])
    if revision != expected_revision:
        raise ValueError(
            f"external revision mismatch: expected {expected_revision}, found {revision}"
        )
    reviewed_sources = _reviewed_source_manifest(
        baseline, source_root, tracked_sources
    )
    weights_sha256, weight_files = _weight_manifest(weights)
    weights_total_bytes, weight_file_bytes = _weight_byte_manifest(weights)
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
    with _track_upstream_source_execution(source_root) as executed_paths:
        infer, adapter = _LOADERS[baseline](source_root, weights, device)
    executed_sources = _executed_source_manifest(
        source_root, executed_paths, tracked_sources
    )
    collision_peers = {
        str(peer).casefold()
        for item in source_attestation["documented_case_collision_exceptions"]
        for peer in item["tracked_case_peers"]
    }
    if collision_peers & {path.casefold() for path in executed_sources}:
        raise ValueError(
            "a documentation case-collision exception overlaps executed upstream source"
        )
    outputs = []
    failures: list[dict[str, object]] = []
    starting_cuda_allocated_bytes: int | None = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        starting_cuda_allocated_bytes = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for sample_id in sorted(visible_paths):
            destination = output_dir / f"{sample_id}.png"
            before: float | None = None
            stage = "decode_and_geometry"
            try:
                visible = _image(visible_paths[sample_id], "RGB")
                thermal = _image(thermal_paths[sample_id], "L")
                if visible.shape[2:] != thermal.shape[2:]:
                    raise ValueError(f"unaligned geometry for {sample_id}")
                stage = "adapter_inference_and_png_save"
                before = time.perf_counter()
                fused = infer(visible, thermal)
                _save(fused, destination)
                outputs.append({
                    "sample_id": sample_id,
                    "path": destination.name,
                    "sha256": _sha256(destination),
                    "elapsed_seconds": time.perf_counter() - before,
                })
            except Exception as error:
                destination.unlink(missing_ok=True)
                failures.append(
                    {
                        "sample_id": sample_id,
                        "stage": stage,
                        "reason": f"{type(error).__name__}: {error}",
                        "elapsed_seconds": (
                            None if before is None else time.perf_counter() - before
                        ),
                    }
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    post_attestation, post_tracked_sources = _source_worktree_attestation(
        source_root,
        allowed_case_collision_paths=allowed_case_collision_paths,
    )
    if (
        post_attestation["revision"] != source_attestation["revision"]
        or post_tracked_sources != tracked_sources
        or _executed_source_manifest(source_root, executed_paths, tracked_sources)
        != executed_sources
        or _reviewed_source_manifest(baseline, source_root, tracked_sources)
        != reviewed_sources
    ):
        raise RuntimeError("external source changed during inference")
    post_weights_sha256, post_weight_files = _weight_manifest(weights)
    post_weights_total_bytes, post_weight_file_bytes = _weight_byte_manifest(weights)
    if (
        post_weights_sha256 != weights_sha256
        or post_weight_files != weight_files
        or post_weights_total_bytes != weights_total_bytes
        or post_weight_file_bytes != weight_file_bytes
    ):
        raise RuntimeError("external checkpoint changed during inference")
    peak_cuda_allocated_bytes: int | None = None
    ending_cuda_allocated_bytes: int | None = None
    incremental_peak_cuda_allocated_bytes: int | None = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        ending_cuda_allocated_bytes = int(torch.cuda.memory_allocated(device))
        incremental_peak_cuda_allocated_bytes = max(
            0,
            peak_cuda_allocated_bytes - int(starting_cuda_allocated_bytes or 0),
        )
    report: dict[str, object] = {
        "schema_version": "openprism.external-fusion-run/1.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "locked_final_test_external_outputs_require_downstream_evaluation"
            if partition == "test"
            else "validation_external_outputs_require_downstream_evaluation"
        ),
        "baseline": baseline,
        "dataset": dataset,
        "partition": partition,
        "final_test_unlocked": partition == "test" and unlock_final_test,
        "one_shot_controller_authorized": controller_authorization is not None,
        "controller_authorization": controller_authorization,
        "repository": str(source_root.resolve()),
        "revision": revision,
        "source_attestation": {
            **source_attestation,
            "verified_before_and_after_inference": True,
            "executed_source_files_sha256": executed_sources,
            "reviewed_source_files_sha256": reviewed_sources,
            "source_execution_policy": (
                "tracked .py files under the pinned worktree are compiled directly "
                "from source; untracked bytecode caches are never executed"
            ),
        },
        "weights": str(weights.resolve()),
        "weights_sha256": weights_sha256,
        "weight_files": weight_files,
        "weights_total_bytes": weights_total_bytes,
        "weight_file_bytes": weight_file_bytes,
        "parameter_inventory": adapter.get("parameter_inventory"),
        "adapter_source_sha256": _sha256(Path(__file__)),
        "adapter": adapter,
        "inputs": {
            "visible_directory": str(visible_dir.resolve()),
            "thermal_directory": str(thermal_dir.resolve()),
            "paired_ids_sorted": sorted(visible_paths),
        },
        "runtime": _runtime_environment(device),
        "runtime_resources": {
            "starting_cuda_allocated_bytes": starting_cuda_allocated_bytes,
            "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
            "incremental_peak_cuda_allocated_bytes": incremental_peak_cuda_allocated_bytes,
            "ending_cuda_allocated_bytes": ending_cuda_allocated_bytes,
            "cuda_memory_scope": (
                "process allocations after model load; incremental peak is above that "
                "starting allocation; allocator reserved memory and other processes excluded"
                if device.type == "cuda"
                else "not applicable on a non-CUDA device"
            ),
        },
        "input_count": len(visible_paths),
        "run_complete": len(outputs) + len(failures) == len(visible_paths),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": outputs,
        "failure_accounting": {
            "attempted": len(visible_paths),
            "successful": len(outputs),
            "failed": len(failures),
            "failure_rate": len(failures) / len(visible_paths),
            "failures": failures,
        },
    }
    manifest = output_dir / "run_manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=tuple(_LOADERS), required=True)
    parser.add_argument("--dataset", choices=("llvip", "msrs", "caltech"), required=True)
    parser.add_argument(
        "--partition", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--visible-dir", type=Path, required=True)
    parser.add_argument("--thermal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-weights-sha256")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-nonexecuted-case-collision",
        action="append",
        default=[],
        metavar="TRACKED_PATH",
        help=(
            "explicitly allow one documentation-only tracked path whose differently-cased "
            "peer cannot coexist on this filesystem; repeat for distinct collision groups"
        ),
    )
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
        dataset=args.dataset,
        partition=args.partition,
        unlock_final_test=args.unlock_final_test,
        expected_revision=args.expected_revision,
        expected_weights_sha256=args.expected_weights_sha256,
        device_name=args.device,
        overwrite=args.overwrite,
        allowed_case_collision_paths=args.allow_nonexecuted_case_collision,
    )
    print(json.dumps({
        "baseline": report["baseline"],
        "revision": report["revision"],
        "weights_sha256": report["weights_sha256"],
        "input_count": report["input_count"],
        "successful": report["failure_accounting"]["successful"],
        "failed": report["failure_accounting"]["failed"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    if report["failure_accounting"]["failed"] and report["partition"] != "test":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
