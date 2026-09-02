"""Run the OpenPRISM reference operator console."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import webbrowser

from .datasets import DEFAULT_DATA_ROOT
from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--learned-checkpoint",
        type=Path,
        help="Enable PRISM-EGT automatic fusion with a provenance-bearing checkpoint.",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the local console automatically."
    )
    args = parser.parse_args()
    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve(
        host=args.host,
        port=args.port,
        data_root=args.data_root,
        learned_checkpoint=args.learned_checkpoint,
    )


if __name__ == "__main__":
    main()
