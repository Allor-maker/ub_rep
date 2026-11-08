"""Byte-level reconstruction for text returned by the read_file tool."""

import hashlib
import unittest
from codecs import BOM_UTF8
from pathlib import PurePosixPath

from nsu_budget_agent.contracts.models import ArtifactRef
from nsu_budget_agent.validation.data_plan import _matches_original_text


def reference(raw: bytes) -> ArtifactRef:
    return ArtifactRef(
        path=PurePosixPath("input.csv"), size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )


class ArtifactTextIntegrityTests(unittest.TestCase):
    def test_utf8_bom_crlf_and_leading_bom_character(self):
        text = "Müller,west\r\n"
        encoded = text.encode("utf-8")
        for raw in (encoded, BOM_UTF8 + encoded):
            with self.subTest(has_bom=raw.startswith(BOM_UTF8)):
                self.assertTrue(_matches_original_text(text, reference(raw)))
                self.assertFalse(
                    _matches_original_text(text.replace("west", "east"), reference(raw))
                )
        # Decoding with utf-8-sig removes exactly one BOM, not both.
        self.assertTrue(_matches_original_text("\ufeff" + text, reference(BOM_UTF8 * 2 + encoded)))
        self.assertFalse(_matches_original_text("\ufeff" + text, reference(BOM_UTF8 + encoded)))

    def test_wrong_size_hash_or_invalid_unicode_cannot_match(self):
        raw = b"amount\n10\n"
        self.assertFalse(_matches_original_text("amount\n99\n", reference(raw)))
        self.assertFalse(_matches_original_text("amount\n100\n", reference(raw)))
        self.assertFalse(_matches_original_text("\ud800", reference(raw)))
