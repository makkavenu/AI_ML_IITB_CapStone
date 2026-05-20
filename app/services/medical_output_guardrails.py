"""Medical-domain output guardrails for user-facing responses.

These checks run after the general text guardrail scanner. They are intentionally
conservative and deterministic: the goal is not to diagnose correctness, but to
reduce high-risk phrasing before any medical/health model response is shown to a
user in the demo app.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

MEDICAL_TOOL_NAMES: set[str] = {
    "medical_qa",
    "retfound_analyze",
    "endofm_analyze",
    "sam_med2d_segment",
    "totalsegmentator_segment",
}

_SPECIALIST_TOOL_NAMES: set[str] = {
    "retfound_analyze",
    "endofm_analyze",
    "sam_med2d_segment",
    "totalsegmentator_segment",
}

_EMERGENCY_TERMS = re.compile(
    r"\b(chest\s+pain|chest\s+discomfort|chest\s+tightness|heart\s+attack|"
    r"difficulty\s+breathing|trouble\s+breathing|trouble\s+in\s+breathing|"
    r"breathing\s+trouble|hard\s+to\s+breathe|can't\s+breathe|cannot\s+breathe|"
    r"shortness\s+of\s+breath|short\s+of\s+breath|breathless|breathlessness|"
    r"stroke|seizure|loss\s+of\s+consciousness|unconscious|fainting|"
    r"severe\s+bleeding|suicidal|anaphylaxis|blue\s+lips|worst\s+headache|"
    r"new\s+weakness|facial\s+droop|jaw\s+pain|neck\s+pain|arm\s+pain)\b",
    re.IGNORECASE,
)

_DIAGNOSIS_CERTAINTY_PATTERNS = [
    re.compile(r"\b(you\s+(definitely|certainly)\s+have)\b", re.IGNORECASE),
    re.compile(r"\b(this\s+is\s+(definitely|certainly)\s+)\b", re.IGNORECASE),
    re.compile(r"\b(the\s+diagnosis\s+is\s+)\b", re.IGNORECASE),
    re.compile(r"\b(100\s*%\s+(sure|certain|diagnostic))\b", re.IGNORECASE),
]

_UNSAFE_TREATMENT_PATTERNS = [
    re.compile(r"\b(stop|start|increase|decrease|double)\s+(taking\s+)?(your\s+)?[a-zA-Z][a-zA-Z0-9_-]{2,}\b", re.IGNORECASE),
    re.compile(r"\btake\s+\d+(\.\d+)?\s*(mg|mcg|g|ml|units?)\b", re.IGNORECASE),
    re.compile(r"\b(no\s+need\s+to\s+(see|consult|visit)\s+(a\s+)?(doctor|physician|clinician))\b", re.IGNORECASE),
]

_REQUIRED_DISCLAIMER_PHRASES = (
    "not a diagnosis",
    "research",
    "clinician",
)

_NON_MEDICAL_TOOL_NAMES: set[str] = {
    "legal_qa",
    "vision_llm",
    "object_detection",
}

_MEDICAL_QUERY_TERMS = re.compile(
    r"\b(medical|health|clinical|patient|doctor|hospital|symptom|symptoms|diagnosis|"
    r"disease|treatment|medicine|medication|dose|dosage|pain|fever|"
    r"injury|breathing|breath|breathless|breathlessness|difficulty\s+breathing|"
    r"trouble\s+breathing|trouble\s+in\s+breathing|breathing\s+trouble|"
    r"hard\s+to\s+breathe|can't\s+breathe|cannot\s+breathe|shortness\s+of\s+breath|"
    r"short\s+of\s+breath|chest\s+pain|chest\s+discomfort|chest\s+tightness|"
    r"heart\s+attack|jaw\s+pain|neck\s+pain|arm\s+pain|stroke|seizure|"
    r"scan|xray|x-ray|mri|ct|ultrasound|dicom|nifti|"
    r"retina|retinal|fundus|oct|glaucoma|diabetes|diabetic|"
    r"blood\s+pressure|hypertension|infection|cancer|asthma|"
    r"pregnancy|endoscopy|colonoscopy|polyp|lesion|tumor|tumour|"
    r"organ|vessel|segmentation|radiology|pathology|medgemma|"
    r"retfound|endo-fm|sam-med2d|totalsegmentator)\b",
    re.IGNORECASE,
)

_GENERAL_AI_SAFETY_LIMITATION = (
    "General AI safety limitation: This answer is AI-assisted and may be incomplete "
    "or incorrect. Please verify important information with reliable sources or a "
    "qualified professional before making decisions."
)

_MEDICAL_SAFETY_HEADER = (
    "Medical safety note: This is an AI-assisted research/demo explanation, "
    "not a diagnosis or treatment plan. Please confirm findings with a qualified clinician."
)

_EMERGENCY_NOTICE = (
    "Emergency warning: If symptoms are severe, rapidly worsening, or include chest pain, "
    "breathing difficulty, stroke-like symptoms, severe bleeding, seizure, loss of consciousness, "
    "or self-harm risk, seek urgent medical care immediately."
)

@dataclass
class MedicalGuardrailResult:
    """Result of medical output guardrail processing."""

    is_medical_response: bool
    text: str
    warnings: list[str] = field(default_factory=list)
    blocked: bool = False
    risk_categories: list[str] = field(default_factory=list)


def is_medical_tool_chain(tools_chain: Iterable[str] | None) -> bool:
    """Return True when the agent used any medical model/tool."""
    return bool(MEDICAL_TOOL_NAMES.intersection(set(tools_chain or [])))


def specialist_used_without_medgemma(tools_chain: Iterable[str] | None) -> bool:
    """Return True when a specialist image model was used without final MedGemma explanation."""
    chain = list(tools_chain or [])
    return any(t in _SPECIALIST_TOOL_NAMES for t in chain) and "medical_qa" not in chain


def _looks_medical_from_user_message(user_message: str) -> bool:
    """Return True when the original user request itself is medical/health related."""
    return bool(_MEDICAL_QUERY_TERMS.search(user_message or ""))


def _should_apply_medical_guardrails(
    tools_chain: Iterable[str] | None,
    *,
    user_message: str = "",
) -> bool:
    """Apply medical wording only for genuinely medical requests/responses.

    This prevents legal, object-detection, and generic vision answers from getting
    a medical disclaimer merely because a prior chat turn or an accidental extra
    model call introduced medical text.
    """
    tools = set(tools_chain or [])
    if not MEDICAL_TOOL_NAMES.intersection(tools):
        return False

    if _SPECIALIST_TOOL_NAMES.intersection(tools):
        return True

    if _looks_medical_from_user_message(user_message):
        return True

    return not _NON_MEDICAL_TOOL_NAMES.intersection(tools)


def _append_general_ai_safety_limitation(text: str) -> str:
    """Append a non-medical AI limitation note exactly once."""
    safe_text = (text or "").strip()
    if not safe_text:
        return _GENERAL_AI_SAFETY_LIMITATION
    if _GENERAL_AI_SAFETY_LIMITATION.lower() in safe_text.lower():
        return safe_text
    return f"{safe_text}\n\n{_GENERAL_AI_SAFETY_LIMITATION}"


def _normalise_notice_line(line: str) -> str:
    """Normalise a line for duplicate detection without changing displayed text."""
    cleaned = line.strip().lower()
    cleaned = re.sub(r"^[\s>*_`#•-]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _medical_notice_category_key(normalised_line: str) -> str | None:
    """Return a stable category key for repeated medical safety/disclaimer lines."""
    if normalised_line.startswith("medical safety note:"):
        return "medical_safety_note"
    if normalised_line.startswith("emergency warning:"):
        return "emergency_warning"
    if normalised_line.startswith("safety limitations:"):
        return "safety_limitations"
    if normalised_line.startswith("disclaimer: this information is for educational purposes"):
        return "educational_disclaimer"
    if normalised_line.startswith("important note: this analysis is based"):
        return "important_analysis_note"
    if normalised_line.startswith("if you are experiencing any vision changes"):
        return "vision_changes_warning"
    if normalised_line.startswith("if you are experiencing any emergency red flags"):
        return "emergency_red_flags_warning"
    if normalised_line.startswith("research warning:"):
        return "research_warning"
    if normalised_line.startswith("final explanation:"):
        return "final_explanation_heading"
    return None


def _strip_instruction_tokens(text: str) -> str:
    """Remove chat-template tokens leaked by instruction-tuned models."""
    cleaned = text or ""
    cleaned = re.sub(r"\[/?INST\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<</?SYS>>|<s>|</s>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _remove_raw_json_blocks(text: str) -> str:
    """Remove raw JSON object blocks from user-facing medical answers.

    Specialist tools return structured JSON for the backend. The final medical
    answer should summarize those values, not display repeated JSON payloads.
    """
    lines = (text or "").splitlines()
    cleaned: list[str] = []
    in_json = False
    depth = 0

    for line in lines:
        stripped = line.strip()

        if not in_json and stripped.startswith("{"):
            in_json = True
            depth = stripped.count("{") - stripped.count("}")
            if depth <= 0:
                in_json = False
            continue

        if in_json:
            depth += stripped.count("{") - stripped.count("}")
            if depth <= 0:
                in_json = False
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _is_heading_only(line: str) -> bool:
    normalised = _normalise_notice_line(line)
    return normalised in {
        "findings:",
        "explanation:",
        "final explanation:",
        "limitations:",
        "disclaimer:",
        "summary:",
        "metrics:",
    }


def _remove_empty_repeated_headings(text: str) -> str:
    """Drop headings that have no content before the next heading/notice."""
    lines = (text or "").splitlines()
    result: list[str] = []
    n = len(lines)

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not _is_heading_only(stripped):
            result.append(line)
            continue

        next_nonblank = ""
        for j in range(idx + 1, n):
            candidate = lines[j].strip()
            if candidate:
                next_nonblank = candidate
                break

        if not next_nonblank or _is_heading_only(next_nonblank) or _medical_notice_category_key(_normalise_notice_line(next_nonblank)):
            continue

        result.append(line)

    return "\n".join(result).strip()


def _dedupe_repeated_medical_notices(text: str) -> str:
    """Remove repeated model-generated medical disclaimer/warning lines.

    This keeps the first occurrence of each safety/disclaimer category and drops
    later duplicates. It does not remove per-image findings such as Image 1,
    Image 2, etc.
    """
    text = _strip_instruction_tokens(text or "")
    text = _remove_empty_repeated_headings(_remove_raw_json_blocks(text))
    lines = text.splitlines()
    cleaned_lines: list[str] = []
    seen_notice_categories: set[str] = set()
    seen_long_lines: set[str] = set()
    seen_research_lines: set[str] = set()

    previous_blank = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        previous_blank = False
        normalised = _normalise_notice_line(stripped)
        notice_key = _medical_notice_category_key(normalised)

        if notice_key:
            if notice_key in seen_notice_categories:
                continue
            seen_notice_categories.add(notice_key)
        elif normalised.startswith(("research warning:", "final explanation:", "disclaimer:")):
            if normalised in seen_research_lines:
                continue
            seen_research_lines.add(normalised)
        elif len(normalised) > 80:
            # Remove exact repeated long paragraphs while preserving distinct
            # image-specific findings and short bullet points.
            if normalised in seen_long_lines:
                continue
            seen_long_lines.add(normalised)

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _has_required_disclaimer(text: str) -> bool:
    lower = text.lower()
    return all(phrase in lower for phrase in _REQUIRED_DISCLAIMER_PHRASES)


def _replace_high_certainty_diagnosis(text: str, warnings: list[str], risk_categories: list[str]) -> str:
    updated = text
    for pattern in _DIAGNOSIS_CERTAINTY_PATTERNS:
        if pattern.search(updated):
            updated = pattern.sub("a possible finding is ", updated)
            if "overconfident_diagnosis" not in risk_categories:
                risk_categories.append("overconfident_diagnosis")
                warnings.append("Reworded over-confident diagnostic language into possibility/uncertainty language.")
    return updated


def _flag_treatment_directives(text: str, warnings: list[str], risk_categories: list[str]) -> None:
    for pattern in _UNSAFE_TREATMENT_PATTERNS:
        if pattern.search(text):
            if "unsafe_treatment_directive" not in risk_categories:
                risk_categories.append("unsafe_treatment_directive")
                warnings.append(
                    "Detected medication/treatment directive language; added instruction to verify with a clinician."
                )
            return


def apply_medical_output_guardrails(
    text: str,
    tools_chain: Iterable[str] | None,
    *,
    user_message: str = "",
) -> MedicalGuardrailResult:
    """Apply deterministic medical-domain safety checks to final responses.

    Args:
        text: Candidate user-facing answer.
        tools_chain: Ordered tools used by the agent.
        user_message: Original user question, used only for emergency keyword detection.

    Returns:
        MedicalGuardrailResult containing the possibly rewritten safe text and
        audit metadata. This function does not call an LLM and does not use PHI.
    """
    if not _should_apply_medical_guardrails(tools_chain, user_message=user_message):
        return MedicalGuardrailResult(
            is_medical_response=False,
            text=_append_general_ai_safety_limitation(text),
        )

    warnings: list[str] = []
    risk_categories: list[str] = []
    safe_text = _dedupe_repeated_medical_notices((text or "").strip())

    if not safe_text:
        return MedicalGuardrailResult(
            is_medical_response=True,
            text=(
                _MEDICAL_SAFETY_HEADER
                + "\n\nThe medical model returned an empty response. Please retry or consult a clinician."
            ),
            warnings=["Empty medical response replaced with a safe fallback."],
            risk_categories=["empty_response"],
        )

    if specialist_used_without_medgemma(tools_chain):
        warnings.append(
            "A specialist image model was used without a final MedGemma explanation; added extra caution."
        )
        risk_categories.append("specialist_without_final_explanation")

    safe_text = _replace_high_certainty_diagnosis(safe_text, warnings, risk_categories)
    _flag_treatment_directives(safe_text, warnings, risk_categories)

    combined_for_emergency = f"{user_message}\n{safe_text}"
    if _EMERGENCY_TERMS.search(combined_for_emergency):
        risk_categories.append("possible_emergency_context")
        warnings.append("Added emergency-care warning because emergency-related terms were present.")
        if _EMERGENCY_NOTICE.lower() not in safe_text.lower():
            safe_text = f"{_EMERGENCY_NOTICE}\n\n{safe_text}"

    if not _has_required_disclaimer(safe_text):
        warnings.append("Added required research-use / non-diagnostic / clinician confirmation disclaimer.")
        risk_categories.append("missing_medical_disclaimer")
        safe_text = f"{_MEDICAL_SAFETY_HEADER}\n\n{safe_text}"

    if risk_categories and "safety limitations:" not in safe_text.lower():
        safe_text += (
            "\n\nSafety limitations: AI outputs can be incomplete or incorrect, especially for medical images, "
            "rare conditions, poor-quality scans, and missing clinical history. Do not use this response as "
            "the sole basis for diagnosis, treatment, medication changes, or emergency decisions."
        )

    safe_text = _dedupe_repeated_medical_notices(safe_text)

    return MedicalGuardrailResult(
        is_medical_response=True,
        text=safe_text,
        warnings=warnings,
        blocked=False,
        risk_categories=sorted(set(risk_categories)),
    )
