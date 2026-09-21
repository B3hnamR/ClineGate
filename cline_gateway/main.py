"""Entrypoint: `python -m cline_gateway.main` or `python main.py`."""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from cline_gateway.app import create_app
from cline_gateway.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cline-gateway",
                                     description="OpenAI/Anthropic-compatible proxy "
                                                 "over Cline's LLM API")
    parser.add_argument("--config", "-c", default=None, help="path to config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port

    if args.reload:
        # the reloader forks a child that must re-import the app, so it needs
        # an import string, not an already-constructed app object; the config
        # path travels via the environment
        if args.config:
            os.environ.setdefault("CLINE_GATEWAY_CONFIG", str(args.config))
        uvicorn.run("cline_gateway.app:create_app", factory=True, host=host,
                    port=port, reload=True, log_level="info")
        return 0

    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
