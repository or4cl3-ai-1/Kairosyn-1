#!/usr/bin/env python3
"""
KAIROSYN-1 Production Server Entrypoint
=========================================
Starts the FastAPI inference server with uvicorn.
Configurable via environment variables or CLI flags.

Usage:
    # Development (auto-reload)
    python scripts/serve.py --reload

    # Production
    python scripts/serve.py --host 0.0.0.0 --port 8080 --workers 1

    # Via Docker entrypoint (default)
    python -m uvicorn kairosyn.api.server:app --host 0.0.0.0 --port 8080
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="KAIROSYN-1 Inference Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",    default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port",    type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")),
                        help="Number of uvicorn worker processes. Keep at 1 for GPU.")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"),
                        choices=["debug", "info", "warning", "error", "critical"])
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload (development only)")
    parser.add_argument("--config",
                        default=os.environ.get("KAIROSYN_CONFIG", "configs/model/kairosyn_e4b.yaml"),
                        help="Path to model config YAML")
    parser.add_argument("--checkpoint",
                        default=os.environ.get("MODEL_CHECKPOINT", ""),
                        help="Path to fine-tuned checkpoint (optional)")
    return parser.parse_args()


def print_banner(args):
    print()
    print("━" * 55)
    print("  KAIROSYN-1 Inference Server")
    print("  Or4cl3 AI Solutions — research@or4cl3.ai")
    print("━" * 55)
    print(f"  Host      : {args.host}")
    print(f"  Port      : {args.port}")
    print(f"  Workers   : {args.workers}")
    print(f"  Log level : {args.log_level}")
    print(f"  Config    : {args.config}")
    print(f"  Reload    : {args.reload}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print("━" * 55)
    print(f"  API Docs  : http://{args.host}:{args.port}/docs")
    print(f"  Health    : http://{args.host}:{args.port}/health")
    print(f"  Metrics   : http://{args.host}:{args.port}/metrics")
    print("━" * 55)
    print()


def main():
    args = parse_args()

    # Set environment variables for the server module
    os.environ["KAIROSYN_CONFIG"]   = args.config
    os.environ["MODEL_CHECKPOINT"]  = args.checkpoint
    os.environ["PORT"]              = str(args.port)

    print_banner(args)

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn_kwargs = dict(
        app="kairosyn.api.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
        reload=args.reload,
        loop="uvloop",
        http="httptools",
    )

    # Workers only apply when not in reload mode
    if not args.reload and args.workers > 1:
        uvicorn_kwargs["workers"] = args.workers

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
