#!/usr/bin/env python3
"""Prepare (and only on ``--run`` execute) an OpenPRISM ODM survey project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openprism.survey import (  # noqa: E402
    RoughOverlapContract,
    SUPPORTED_RGB_SUFFIXES,
    SurveyDatumContract,
    SurveyPreparationConfig,
    SurveyPreparationError,
    load_camera_pose_records,
    prepare_odm_survey_project,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage RGB captures and Pixhawk CameraPoseRecords as a deterministic "
            "OpenDroneMap project. This does not itself reconstruct terrain."
        )
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--horizontal-crs", default="EPSG:4326")
    parser.add_argument(
        "--vertical-datum",
        required=True,
        help="Explicit datum for CameraPoseRecord.altitude_msl_m",
    )
    parser.add_argument(
        "--geoid-model",
        required=True,
        help="Exact geoid/model identity used for the MSL altitude conversion",
    )
    parser.add_argument("--geoid-model-sha256")
    parser.add_argument("--nominal-agl-m", type=float, required=True)
    parser.add_argument("--horizontal-fov-deg", type=float, required=True)
    parser.add_argument("--vertical-fov-deg", type=float, required=True)
    parser.add_argument(
        "--planned-forward-overlap",
        type=float,
        required=True,
        help="Fraction in [0,1), for example 0.80",
    )
    parser.add_argument(
        "--planned-side-overlap",
        type=float,
        required=True,
        help="Fraction in [0,1), for example 0.70",
    )
    parser.add_argument("--minimum-forward-overlap", type=float, default=0.70)
    parser.add_argument("--minimum-side-overlap", type=float, default=0.60)
    parser.add_argument("--minimum-image-count", type=int, default=8)
    parser.add_argument(
        "--allow-unknown-position-accuracy",
        action="store_true",
        help="Stage records lacking accuracy, with the limitation preserved in provenance",
    )
    parser.add_argument(
        "--require-camera-optical-center",
        action="store_true",
        help="Reject vehicle/GNSS-origin positions whose lever arm was not applied",
    )
    parser.add_argument("--odm-image", default="opendronemap/odm:latest")
    parser.add_argument(
        "--odm-option",
        action="append",
        help="One ODM argv token; repeat it (use --odm-option=--flag for flag tokens)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Explicitly launch the generated Docker/ODM command after staging",
    )
    return parser


def _find_images(directory: Path) -> tuple[Path, ...]:
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise SurveyPreparationError(f"image directory does not exist: {root}")
    return tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() in SUPPORTED_RGB_SUFFIXES
            ),
            key=lambda path: (path.name.casefold(), path.as_posix()),
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        records = load_camera_pose_records(args.poses)
        images = _find_images(args.image_dir)
        config = SurveyPreparationConfig(
            project_name=args.project_name,
            datum=SurveyDatumContract(
                horizontal_crs_id=args.horizontal_crs,
                vertical_datum_id=args.vertical_datum,
                geoid_model_id=args.geoid_model,
                geoid_model_sha256=args.geoid_model_sha256,
            ),
            overlap=RoughOverlapContract(
                nominal_agl_m=args.nominal_agl_m,
                horizontal_fov_deg=args.horizontal_fov_deg,
                vertical_fov_deg=args.vertical_fov_deg,
                planned_forward_overlap_fraction=args.planned_forward_overlap,
                planned_side_overlap_fraction=args.planned_side_overlap,
                minimum_forward_overlap_fraction=args.minimum_forward_overlap,
                minimum_side_overlap_fraction=args.minimum_side_overlap,
            ),
            minimum_image_count=args.minimum_image_count,
            require_position_accuracy=not args.allow_unknown_position_accuracy,
            require_camera_optical_center=args.require_camera_optical_center,
            odm_image=args.odm_image,
            odm_options=tuple(args.odm_option or ("--dsm",)),
        )
        result = prepare_odm_survey_project(
            images,
            records,
            args.output_root,
            config,
            run_odm=args.run,
        )
    except (OSError, SurveyPreparationError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "project_id": result.project_id,
                "project_directory": str(result.project_directory),
                "manifest": str(result.manifest_json),
                "plan": str(result.plan_json),
                "reused_existing": result.reused_existing,
                "odm_executed": result.odm_executed,
                "docker_argv": list(result.docker_argv),
                "docker_command_powershell": result.docker_command_powershell,
                "warning": (
                    "Prepared inputs are not reconstructed terrain and are not a "
                    "survey-grade claim. Read survey_plan.json before execution."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
