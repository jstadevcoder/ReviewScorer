from __future__ import annotations
from pathlib import Path
from typing import Any

import json
import logging
import structlog
import sys

class LoggingConfig:
    DEFAULT_CONFIG: dict[str, Any] = {
        "level": "INFO",
        "format": "console",
        "output": "console",
    }

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config()

    def configure(self) -> None:
        """Configure standard logging and structlog."""
        level_name = str(self.config.get("level", "INFO")).upper()
        level = getattr(logging, level_name, None)

        if not isinstance(level, int):
            raise ValueError(f"Invalid logging level: {level_name}")

        handler = self._create_handler(level)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(level)
        root_logger.addHandler(handler)

        renderer = self._create_renderer()

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                renderer,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _load_config(self) -> dict[str, Any]:
        """
        Load JSON or YAML configuration.

        If no file is configured or the file does not exist,
        default console settings are used.
        """
        if self.config_path is None or not self.config_path.is_file():
            return self.DEFAULT_CONFIG.copy()

        suffix = self.config_path.suffix.lower()

        if suffix == ".json":
            with self.config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)

        elif suffix in {".yaml", ".yml"}:
            import yaml

            with self.config_path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}

        else:
            raise ValueError(
                f"Unsupported configuration format: {self.config_path.suffix}"
            )

        if not isinstance(config, dict):
            raise ValueError("Logging configuration must be an object or mapping")

        result = self.DEFAULT_CONFIG.copy()
        result.update(config)
        return result

    def _create_handler(self, level: int) -> logging.Handler:
        output = str(self.config.get("output", "console")).lower()

        if output == "console":
            handler = logging.StreamHandler(sys.stdout)

        elif output == "stderr":
            handler = logging.StreamHandler(sys.stderr)

        elif output == "file":
            file_path = self.config.get("file")

            if not file_path:
                raise ValueError(
                    "'file' must be specified when output is set to 'file'"
                )

            handler = logging.FileHandler(
                file_path,
                encoding="utf-8",
            )

        else:
            raise ValueError(
                f"Unsupported output: {output}. "
                "Use 'console', 'stderr', or 'file'."
            )

        handler.setLevel(level)
        return handler

    def _create_renderer(self) -> Any:
        log_format = str(self.config.get("format", "console")).lower()

        if log_format == "console":
            return structlog.dev.ConsoleRenderer()

        if log_format == "json":
            return structlog.processors.JSONRenderer()

        raise ValueError(
            f"Unsupported format: {log_format}. "
            "Use 'console' or 'json'."
        )