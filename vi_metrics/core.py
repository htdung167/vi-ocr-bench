"""Vietnamese handwritten OCR evaluation metrics."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass

from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from rapidfuzz.distance import Levenshtein

# Vietnamese vowel characters mapped to (base without tone, tone id).
# Tone ids: 0=ngang, 1=sắc, 2=huyền, 3=hỏi, 4=ngã, 5=nặng.
_VN_CHAR_TONE: dict[str, tuple[str, int]] = {}
for _base, _forms in {
    "a": ("a", "á", "à", "ả", "ã", "ạ"),
    "ă": ("ă", "ắ", "ằ", "ẳ", "ẵ", "ặ"),
    "â": ("â", "ấ", "ầ", "ẩ", "ẫ", "ậ"),
    "e": ("e", "é", "è", "ẻ", "ẽ", "ẹ"),
    "ê": ("ê", "ế", "ề", "ể", "ễ", "ệ"),
    "i": ("i", "í", "ì", "ỉ", "ĩ", "ị"),
    "o": ("o", "ó", "ò", "ỏ", "õ", "ọ"),
    "ô": ("ô", "ố", "ồ", "ổ", "ỗ", "ộ"),
    "ơ": ("ơ", "ớ", "ờ", "ở", "ỡ", "ợ"),
    "u": ("u", "ú", "ù", "ủ", "ũ", "ụ"),
    "ư": ("ư", "ứ", "ừ", "ử", "ữ", "ự"),
    "y": ("y", "ý", "ỳ", "ỷ", "ỹ", "ỵ"),
}.items():
    for _tone_id, _char in enumerate(_forms):
        _VN_CHAR_TONE[_char] = (_base, _tone_id)
        _VN_CHAR_TONE[_char.upper()] = (_base.upper(), _tone_id)


@dataclass(frozen=True)
class MetricScores:
    cer: float
    wer: float
    diacritic_insensitive_cer: float
    tone_error_rate: float
    num_samples: int
    num_ref_chars: int
    num_ref_words: int
    num_tone_positions: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def normalize_text(text: str) -> str:
    return text.strip()


def remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics, including vowel modifiers and tone marks."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def strip_tones(text: str) -> str:
    return "".join(_strip_tone_char(char) for char in text)


def compute_cer(references: list[str], predictions: list[str]) -> float:
    refs = [normalize_text(ref) for ref in references]
    hyps = [normalize_text(pred) for pred in predictions]
    return float(jiwer_cer(refs, hyps))


def compute_wer(references: list[str], predictions: list[str]) -> float:
    refs = [normalize_text(ref) for ref in references]
    hyps = [normalize_text(pred) for pred in predictions]
    return float(jiwer_wer(refs, hyps))


def compute_diacritic_insensitive_cer(
    references: list[str], predictions: list[str]
) -> float:
    refs = [remove_diacritics(normalize_text(ref)) for ref in references]
    hyps = [remove_diacritics(normalize_text(pred)) for pred in predictions]
    return float(jiwer_cer(refs, hyps))


def compute_tone_error_rate(references: list[str], predictions: list[str]) -> float:
    total_errors = 0
    total_positions = 0
    for reference, prediction in zip(references, predictions, strict=True):
        errors, positions = _tone_errors_for_pair(reference, prediction)
        total_errors += errors
        total_positions += positions
    if total_positions == 0:
        return 0.0
    return total_errors / total_positions


def evaluate_all(references: list[str], predictions: list[str]) -> MetricScores:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")

    refs = [normalize_text(ref) for ref in references]
    hyps = [normalize_text(pred) for pred in predictions]

    tone_errors_total = 0
    tone_positions_total = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        errors, positions = _tone_errors_for_pair(ref, hyp)
        tone_errors_total += errors
        tone_positions_total += positions

    tone_error_rate = (
        tone_errors_total / tone_positions_total if tone_positions_total > 0 else 0.0
    )

    refs_no_diacritics = [remove_diacritics(ref) for ref in refs]
    hyps_no_diacritics = [remove_diacritics(hyp) for hyp in hyps]

    return MetricScores(
        cer=float(jiwer_cer(refs, hyps)),
        wer=float(jiwer_wer(refs, hyps)),
        diacritic_insensitive_cer=float(jiwer_cer(refs_no_diacritics, hyps_no_diacritics)),
        tone_error_rate=tone_error_rate,
        num_samples=len(refs),
        num_ref_chars=sum(len(ref) for ref in refs),
        num_ref_words=sum(len(ref.split()) for ref in refs),
        num_tone_positions=tone_positions_total,
    )


def _strip_tone_char(char: str) -> str:
    info = _VN_CHAR_TONE.get(char)
    if info is None:
        return char
    base, _tone = info
    return base


def _tone_id(char: str) -> int | None:
    info = _VN_CHAR_TONE.get(char)
    if info is None:
        return None
    return info[1]


def _tone_errors_for_pair(reference: str, prediction: str) -> tuple[int, int]:
    ref = normalize_text(reference)
    hyp = normalize_text(prediction)
    ref_base = strip_tones(ref)
    hyp_base = strip_tones(hyp)

    tone_errors = 0
    tone_positions = 0

    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref_base, hyp_base):
        if tag == "equal":
            for ref_idx, hyp_idx in zip(range(i1, i2), range(j1, j2), strict=False):
                ref_tone = _tone_id(ref[ref_idx])
                if ref_tone is None:
                    continue
                tone_positions += 1
                hyp_tone = _tone_id(hyp[hyp_idx])
                if hyp_tone is None or hyp_tone != ref_tone:
                    tone_errors += 1
        elif tag in {"replace", "delete"}:
            for ref_idx in range(i1, i2):
                if _tone_id(ref[ref_idx]) is not None:
                    tone_positions += 1
                    tone_errors += 1

    return tone_errors, tone_positions
