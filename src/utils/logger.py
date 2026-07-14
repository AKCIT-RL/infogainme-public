"""Custom logging utilities for the info-gainme_dev project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


class ClaryLogger:
    """Custom logger class with consistent formatting and configuration."""
    
    _loggers: dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def configure(
        cls,
        level: int = logging.INFO,
        format_str: Optional[str] = None,
        log_file: Optional[Path] = None,
        include_console: bool = True
    ) -> None:
        """Configure the logging system globally.
        
        Args:
            level: Logging level (default: INFO).
            format_str: Custom format string for log messages.
            log_file: Optional file to write logs to.
            include_console: Whether to include console output.
        """
        if cls._configured:
            return
            
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create formatter
        formatter = logging.Formatter(format_str)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Reduce verbosity of third-party libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        
        # Add console handler
        if include_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance with the specified name.
        
        Args:
            name: Logger name (typically __name__).
            
        Returns:
            Configured logger instance.
        """
        if name not in cls._loggers:
            # Auto-configure on first use if not already configured
            if not cls._configured:
                cls.configure()
            
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def set_level(cls, level: int) -> None:
        """Set the logging level for all loggers.
        
        Args:
            level: Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        logging.getLogger().setLevel(level)
        for handler in logging.getLogger().handlers:
            handler.setLevel(level)
