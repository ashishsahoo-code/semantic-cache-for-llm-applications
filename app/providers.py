import hashlib
from typing import Protocol


class LLMProvider(Protocol):
    async def generate(self, prompt: str, model: str) -> str: ...


class DemoLLMProvider:
    """Local deterministic provider used to exercise cache behavior without API credentials."""

    async def generate(self, prompt: str, model: str) -> str:
        fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        return (
            f"[{model}] Demo response for: {prompt.strip()} "
            f"(request fingerprint: {fingerprint})"
        )
