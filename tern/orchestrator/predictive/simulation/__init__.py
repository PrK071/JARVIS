"""Secure execution foundation for future candidate simulation."""

from .sandbox import (
    DockerSandboxProvider,
    SandboxCapabilities,
    SandboxProvider,
    SandboxRunResult,
    SandboxRunSpec,
    SandboxSelfTestResult,
    UnavailableSandboxProvider,
    detect_sandbox_provider,
)

__all__ = [
    "DockerSandboxProvider",
    "SandboxCapabilities",
    "SandboxProvider",
    "SandboxRunResult",
    "SandboxRunSpec",
    "SandboxSelfTestResult",
    "UnavailableSandboxProvider",
    "detect_sandbox_provider",
]
