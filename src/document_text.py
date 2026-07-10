from typing import List, Optional, Protocol

try:
    from .native_runtime import apply_native_runtime_defaults
except ImportError:
    from native_runtime import apply_native_runtime_defaults


apply_native_runtime_defaults()

from charset_normalizer import from_bytes

try:
    from .document_config import (
        CONFLICTING_LEGACY_ENCODINGS,
        DETECTED_ENCODING_CHAOS_GAP,
        DETECTED_ENCODING_MAX_CHAOS,
        DETECTED_ENCODING_MIN_COHERENCE,
        MAX_SHORT_WESTERN_FALLBACK_BYTES,
        TEXT_ENCODING_ALIASES,
        TEXT_ENCODING_BOMS,
        TEXT_ENCODING_FALLBACKS,
        UTF_NUL_FAMILY_ENCODINGS,
        UTF_NUL_PATTERN_THRESHOLD,
        has_binary_control_characters,
        normalize_encoding_name,
        nul_ratio,
        western_text_penalty,
    )
except ImportError:
    from document_config import (
        CONFLICTING_LEGACY_ENCODINGS,
        DETECTED_ENCODING_CHAOS_GAP,
        DETECTED_ENCODING_MAX_CHAOS,
        DETECTED_ENCODING_MIN_COHERENCE,
        MAX_SHORT_WESTERN_FALLBACK_BYTES,
        TEXT_ENCODING_ALIASES,
        TEXT_ENCODING_BOMS,
        TEXT_ENCODING_FALLBACKS,
        UTF_NUL_FAMILY_ENCODINGS,
        UTF_NUL_PATTERN_THRESHOLD,
        has_binary_control_characters,
        normalize_encoding_name,
        nul_ratio,
        western_text_penalty,
    )


class TextDecoder(Protocol):
    def decode_text_file(
        self, document_path: str, text_encoding: Optional[str] = None
    ) -> str:
        ...

    def decode_supported_text(self, raw_content: bytes, encoding: str) -> str:
        ...

    def decode_utf8_or_western_text(self, raw_content: bytes) -> str:
        ...

    def decode_confident_detected_text(self, raw_content: bytes, matches) -> Optional[str]:
        ...

    def fallback_encoding_is_plausible(
        self, raw_content: bytes, matches, encoding: str, fallback_text: str
    ) -> bool:
        ...

    def detected_encoding_is_confident(self, matches, index: int) -> bool:
        ...

    def decode_nul_pattern_text(self, raw_content: bytes) -> Optional[str]:
        ...

    def nul_pattern_encodings(self, raw_content: bytes) -> List[str]:
        ...


class DefaultTextDecoder:
    def decode_text_file(
        self, document_path: str, text_encoding: Optional[str] = None
    ) -> str:
        with open(document_path, "rb") as file:
            raw_content = file.read()

        if text_encoding:
            normalized_text_encoding = text_encoding.strip().lower()
            if normalized_text_encoding == "utf-8-or-western":
                return self.decode_utf8_or_western_text(raw_content)
            if normalized_text_encoding != "auto":
                return self.decode_supported_text(raw_content, text_encoding.strip())

        for bom, encoding in TEXT_ENCODING_BOMS:
            if raw_content.startswith(bom):
                return self.decode_supported_text(raw_content, encoding)

        nul_pattern_text = self.decode_nul_pattern_text(raw_content)
        if nul_pattern_text is not None:
            return nul_pattern_text

        try:
            return self.decode_supported_text(raw_content, "utf-8")
        except UnicodeDecodeError:
            pass

        detected_matches = list(from_bytes(raw_content))
        detected_text = self.decode_confident_detected_text(
            raw_content, detected_matches
        )
        if detected_text is not None:
            return detected_text

        for encoding in TEXT_ENCODING_FALLBACKS:
            try:
                fallback_text = self.decode_supported_text(raw_content, encoding)
            except (LookupError, UnicodeDecodeError):
                continue
            if not self.fallback_encoding_is_plausible(
                raw_content, detected_matches, encoding, fallback_text
            ):
                continue
            return fallback_text

        raise UnicodeDecodeError(
            "text-file",
            raw_content,
            0,
            len(raw_content),
            "unable to detect a supported text encoding",
        )

    def decode_supported_text(self, raw_content: bytes, encoding: str) -> str:
        decoded_text = raw_content.decode(encoding)
        if has_binary_control_characters(decoded_text):
            raise UnicodeDecodeError(
                encoding,
                raw_content,
                0,
                len(raw_content),
                "decoded text contains binary control characters",
            )
        return decoded_text

    def decode_utf8_or_western_text(self, raw_content: bytes) -> str:
        for bom, encoding in TEXT_ENCODING_BOMS:
            if raw_content.startswith(bom):
                return self.decode_supported_text(raw_content, encoding)

        try:
            return self.decode_supported_text(raw_content, "utf-8")
        except UnicodeDecodeError:
            pass

        for encoding in TEXT_ENCODING_FALLBACKS:
            try:
                return self.decode_supported_text(raw_content, encoding)
            except (LookupError, UnicodeDecodeError):
                continue

        raise UnicodeDecodeError(
            "text-file",
            raw_content,
            0,
            len(raw_content),
            "unable to decode as UTF-8 or Western text",
        )

    def decode_confident_detected_text(
        self, raw_content: bytes, matches
    ) -> Optional[str]:
        for index, match in enumerate(matches):
            if not self.detected_encoding_is_confident(matches, index):
                continue
            candidates = [match.encoding, *(match.could_be_from_charset or [])]
            for candidate in candidates:
                if not candidate:
                    continue
                normalized = normalize_encoding_name(candidate)
                if normalized in UTF_NUL_FAMILY_ENCODINGS and b"\x00" not in raw_content:
                    continue
                try:
                    return self.decode_supported_text(raw_content, candidate)
                except (LookupError, UnicodeDecodeError):
                    continue
        return None

    def fallback_encoding_is_plausible(
        self, raw_content: bytes, matches, encoding: str, fallback_text: str
    ) -> bool:
        fallback_penalty = western_text_penalty(fallback_text)
        if fallback_penalty is None:
            return False

        normalized_encoding = normalize_encoding_name(encoding)
        aliases = TEXT_ENCODING_ALIASES.get(
            normalized_encoding or encoding, {normalized_encoding or encoding}
        )
        fallback_listed_as_low_chaos = False

        for match in matches:
            chaos = float(getattr(match, "percent_chaos", 100) or 0)
            if chaos > DETECTED_ENCODING_MAX_CHAOS:
                continue
            candidates = [match.encoding, *(match.could_be_from_charset or [])]
            for candidate in candidates:
                normalized_candidate = normalize_encoding_name(candidate)
                if normalized_candidate in aliases:
                    fallback_listed_as_low_chaos = True
                    continue
                if normalized_candidate not in CONFLICTING_LEGACY_ENCODINGS:
                    continue
                try:
                    candidate_text = self.decode_supported_text(raw_content, candidate)
                except (LookupError, UnicodeDecodeError):
                    continue
                if candidate_text != fallback_text:
                    return False

        # Auto mode must fail closed on ambiguous legacy candidates. The UI
        # defaults to Auto; Western and other legacy codecs are explicit opt-ins.
        if len(raw_content) <= MAX_SHORT_WESTERN_FALLBACK_BYTES:
            if fallback_penalty != 0:
                return False
            # charset_normalizer is unreliable for very short content. Explicitly
            # verify that no conflicting legacy encoding produces a plausible but
            # different decode — if one does, the content is ambiguous.
            for conflict_enc in CONFLICTING_LEGACY_ENCODINGS:
                try:
                    conflict_text = self.decode_supported_text(raw_content, conflict_enc)
                except (LookupError, UnicodeDecodeError):
                    continue
                if conflict_text == fallback_text:
                    continue
                if western_text_penalty(conflict_text) is not None:
                    return False
            return True

        return fallback_listed_as_low_chaos and fallback_penalty == 0

    def detected_encoding_is_confident(self, matches, index: int) -> bool:
        if index != 0:
            return False

        match = matches[index]
        coherence = float(getattr(match, "coherence", 0) or 0)
        if coherence >= DETECTED_ENCODING_MIN_COHERENCE:
            return True

        chaos = float(getattr(match, "percent_chaos", 100) or 0)
        if chaos > DETECTED_ENCODING_MAX_CHAOS:
            return False
        if index + 1 >= len(matches):
            return False

        next_chaos = float(getattr(matches[index + 1], "percent_chaos", 100) or 0)
        return next_chaos - chaos >= DETECTED_ENCODING_CHAOS_GAP

    def decode_nul_pattern_text(self, raw_content: bytes) -> Optional[str]:
        if b"\x00" not in raw_content:
            return None

        for encoding in self.nul_pattern_encodings(raw_content):
            try:
                return self.decode_supported_text(raw_content, encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return None

    def nul_pattern_encodings(self, raw_content: bytes) -> List[str]:
        encodings: List[str] = []

        if len(raw_content) % 4 == 0:
            utf32le_nul_ratio = (
                nul_ratio(raw_content, 1, 4)
                + nul_ratio(raw_content, 2, 4)
                + nul_ratio(raw_content, 3, 4)
            ) / 3
            utf32be_nul_ratio = (
                nul_ratio(raw_content, 0, 4)
                + nul_ratio(raw_content, 1, 4)
                + nul_ratio(raw_content, 2, 4)
            ) / 3
            utf32le_text_ratio = 1 - nul_ratio(raw_content, 0, 4)
            utf32be_text_ratio = 1 - nul_ratio(raw_content, 3, 4)

            if (
                utf32le_nul_ratio >= UTF_NUL_PATTERN_THRESHOLD
                and utf32le_text_ratio >= UTF_NUL_PATTERN_THRESHOLD
            ):
                encodings.append("utf-32-le")
            if (
                utf32be_nul_ratio >= UTF_NUL_PATTERN_THRESHOLD
                and utf32be_text_ratio >= UTF_NUL_PATTERN_THRESHOLD
            ):
                encodings.append("utf-32-be")

        if len(raw_content) % 2 == 0:
            even_nul_ratio = nul_ratio(raw_content, 0, 2)
            odd_nul_ratio = nul_ratio(raw_content, 1, 2)
            if (
                odd_nul_ratio >= UTF_NUL_PATTERN_THRESHOLD
                and even_nul_ratio <= 1 - UTF_NUL_PATTERN_THRESHOLD
            ):
                encodings.append("utf-16-le")
            if (
                even_nul_ratio >= UTF_NUL_PATTERN_THRESHOLD
                and odd_nul_ratio <= 1 - UTF_NUL_PATTERN_THRESHOLD
            ):
                encodings.append("utf-16-be")

        return encodings


DEFAULT_TEXT_DECODER = DefaultTextDecoder()


def decode_text_file(
    document_path: str, text_encoding: Optional[str] = None
) -> str:
    return DEFAULT_TEXT_DECODER.decode_text_file(document_path, text_encoding)


def decode_supported_text(raw_content: bytes, encoding: str) -> str:
    return DEFAULT_TEXT_DECODER.decode_supported_text(raw_content, encoding)


def decode_utf8_or_western_text(raw_content: bytes) -> str:
    return DEFAULT_TEXT_DECODER.decode_utf8_or_western_text(raw_content)


def decode_confident_detected_text(raw_content: bytes, matches) -> Optional[str]:
    return DEFAULT_TEXT_DECODER.decode_confident_detected_text(raw_content, matches)


def fallback_encoding_is_plausible(
    raw_content: bytes, matches, encoding: str, fallback_text: str
) -> bool:
    return DEFAULT_TEXT_DECODER.fallback_encoding_is_plausible(
        raw_content, matches, encoding, fallback_text
    )


def detected_encoding_is_confident(matches, index: int) -> bool:
    return DEFAULT_TEXT_DECODER.detected_encoding_is_confident(matches, index)


def decode_nul_pattern_text(raw_content: bytes) -> Optional[str]:
    return DEFAULT_TEXT_DECODER.decode_nul_pattern_text(raw_content)


def nul_pattern_encodings(raw_content: bytes) -> List[str]:
    return DEFAULT_TEXT_DECODER.nul_pattern_encodings(raw_content)
