"""Guardrail scanner for detecting malicious or harmful content.

This module is attached as-is and must not be modified.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?previous\s+(instructions|prompts|rules|guidelines)",
    r"you\s+are\s+now\s+.{0,40}without\s+(any\s+)?(restrictions|guidelines|rules|ethics)",
    r"\bjailbreak\b",
    r"\bdo\s+anything\s+now\b",
    r"\bdan\s+mode\b",
    r"pretend\s+you\s+(have\s+no|are\s+without)\s+(restrictions|guidelines|rules|ethics)",
    r"act\s+as\s+(if\s+you\s+(are\s+)?)?an?\s+(unrestricted|uncensored|unfiltered|evil)",
    r"(forget|ignore)\s+(your\s+)?(safety|ethical|content)\s+(guidelines|rules|policy|policies)",
    r"override\s+(your\s+)?(safety|ethical|content)\s+(guidelines|rules|policy|policies)",
]

_HARMFUL_PATTERNS: list[str] = [
    r"\b(how\s+to\s+(make|build|create|synthesize)\s+(a\s+)?(bomb|explosive|bioweapon|chemical\s+weapon|nerve\s+agent))\b",
    r"\b(bomb|explosive)\s+(making|creation|synthesis|instructions|tutorial)\b",
    r"\b(malware|ransomware|virus|trojan|rootkit)\s+(code|script|creation|tutorial)\b",
    r"\b(child\s+(sexual|abuse|exploitation|pornography))\b",
    r"\b(how\s+to\s+(hack|crack|brute.?force)\s+.{0,30}(without\s+(permission|authoriz)|(illegally)))\b",
    r"\b(instructions?\s+for\s+(mass\s+)?(shooting|killing|murder))\b",
]

_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]
_COMPILED_HARMFUL = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _HARMFUL_PATTERNS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """Result returned by scan_text_content().

    Attributes:
        flagged: True if the content was deemed malicious or harmful.
        reason: Human-readable explanation of why the content was flagged.
        category: Machine-readable category tag (e.g. 'prompt_injection').
    """

    flagged: bool
    reason: str
    category: str


def scan_text_content(text: str) -> ScanResult:
    """Scan a text string for prompt-injection and harmful-content patterns.

    Args:
        text: The raw text to inspect (user input or model output).

    Returns:
        A ScanResult whose ``flagged`` field is True when a violation is found.
        When not flagged, ``reason`` and ``category`` are empty strings.
    """
    if not text or not text.strip():
        return ScanResult(flagged=False, reason="", category="")

    # 1. Prompt-injection checks
    for pattern in _COMPILED_INJECTION:
        match = pattern.search(text)
        if match:
            snippet = match.group(0)[:120]
            logger.warning("Guardrail — prompt injection detected: %r", snippet)
            return ScanResult(
                flagged=True,
                reason=f"Prompt-injection attempt detected: '{snippet}'",
                category="prompt_injection",
            )

    # 2. Harmful-content checks
    for pattern in _COMPILED_HARMFUL:
        match = pattern.search(text)
        if match:
            snippet = match.group(0)[:120]
            logger.warning("Guardrail — harmful content detected: %r", snippet)
            return ScanResult(
                flagged=True,
                reason=f"Harmful content detected: '{snippet}'",
                category="harmful_content",
            )

    return ScanResult(flagged=False, reason="", category="")
