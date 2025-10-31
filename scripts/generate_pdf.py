#!/usr/bin/env python3
"""Generate a single PDF that concatenates all Markdown documentation.

This script collects every Markdown file in the repository (including the
project README) and creates a single, plain-text PDF. It performs light
Markdown to text conversion so that the output is easy to read even without a
full Markdown renderer. The PDF is written to the repository root as
``ovos-technical-manual.pdf``.
"""

from __future__ import annotations

import re
import textwrap
import unicodedata
from pathlib import Path
from typing import Iterable, List

# PDF page metrics (letter size with one inch margins)
PAGE_WIDTH = 612  # 8.5 in * 72 points
PAGE_HEIGHT = 792  # 11 in * 72 points
MARGIN = 72
FONT_SIZE = 12
LEADING = 14  # line height

LINES_PER_PAGE = int((PAGE_HEIGHT - 2 * MARGIN) // LEADING)
START_X = MARGIN
START_Y = PAGE_HEIGHT - MARGIN

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "ovos-technical-manual.pdf"


def find_markdown_files() -> List[Path]:
    """Return every Markdown file that should be included in the PDF."""

    files: List[Path] = []
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        if path.name.lower() == "license.md":
            # The repository already ships a LICENSE file; skip any alias.
            continue
        files.append(path)

    def sort_key(p: Path) -> tuple[int, str]:
        relative = p.relative_to(ROOT).as_posix()
        # README first, then everything else alphabetically.
        if p.name.lower() == "readme.md":
            return (0, relative)
        return (1, relative)

    return sorted(files, key=sort_key)


def markdown_to_plain_lines(text: str) -> List[str]:
    """Convert a Markdown string into wrapped plain-text lines."""

    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            lines.append("")
            continue

        heading_match = re.match(r"^(#+)\\s*(.*)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            if heading:
                heading_text = heading.upper()
                underline_char = "=" if level == 1 else "-"
                wrapped_heading = textwrap.wrap(heading_text, width=80) or [heading_text]
                lines.extend(wrapped_heading)
                lines.append(underline_char * min(len(wrapped_heading[0]), 80))
                lines.append("")
            continue

        # Strip list markers.
        list_match = re.match(r"^([*+-]|\\d+\\.)\\s+(.*)$", line)
        if list_match:
            content = list_match.group(2)
            line = f"- {content}"

        # Convert links and images to readable text.
        line = re.sub(r"!\\[(.*?)\\]\\((.*?)\\)", r"[Image: \\1] (\\2)", line)
        line = re.sub(r"\\[(.*?)\\]\\((.*?)\\)", r"\\1 (\\2)", line)

        # Remove simple Markdown formatting characters.
        line = line.replace("**", "").replace("__", "").replace("`", "")
        line = line.lstrip("> ")

        wrapped = textwrap.wrap(line, width=90) or [line]
        lines.extend(wrapped)

    return lines


def normalise_for_pdf(text: str) -> str:
    """Normalise text so it is safe to embed in a PDF literal string."""

    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return ascii_text


def chunk_lines(all_lines: List[str]) -> List[List[str]]:
    """Split the full document into page-sized chunks."""

    if not all_lines:
        return [[]]

    pages = [
        all_lines[i : i + LINES_PER_PAGE]
        for i in range(0, len(all_lines), LINES_PER_PAGE)
    ]
    return pages


def build_page_stream(lines: Iterable[str]) -> bytes:
    parts = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LEADING} TL", f"{START_X} {START_Y} Td"]
    for line in lines:
        cleaned = normalise_for_pdf(line)
        parts.append(f"({cleaned}) Tj")
        parts.append("T*")
    parts.append("ET")
    stream_text = "\n".join(parts) + "\n"
    return stream_text.encode("ascii")


def create_pdf(pages: List[List[str]]) -> bytes:
    content_streams = [build_page_stream(page) for page in pages]
    num_pages = len(content_streams)

    objects: List[bytes] = []

    # 1. Catalog object (points to the pages tree)
    catalog_obj = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects.append(catalog_obj)

    # 2. Pages object
    kids_refs = " ".join(f"{3 + idx} 0 R" for idx in range(num_pages))
    pages_dict = f"<< /Type /Pages /Kids [ {kids_refs} ] /Count {num_pages} >>".encode("ascii")
    objects.append(pages_dict)

    font_obj_number = 3 + 2 * num_pages

    # 3..(2+num_pages) Page dictionaries
    for page_index in range(num_pages):
        content_ref = 3 + num_pages + page_index
        page_dict = (
            "<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 {0} {1}] "
            "/Resources << /Font << /F1 {2} 0 R >> >> "
            "/Contents {3} 0 R >>"
        ).format(PAGE_WIDTH, PAGE_HEIGHT, font_obj_number, content_ref)
        objects.append(page_dict.encode("ascii"))

    # Content streams
    for stream in content_streams:
        obj = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream"
        objects.append(obj)

    # Font object (Helvetica)
    font_obj = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    objects.append(font_obj)

    # Assemble the final PDF file.
    output = bytearray()
    output.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        if not obj.endswith(b"\n"):
            output.extend(b"\n")
        output.extend(b"endobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            "trailer\n"
            "<< /Size {size} /Root 1 0 R >>\n"
            "startxref\n{start}\n"
            "%%EOF"
        ).format(size=len(objects) + 1, start=xref_offset).encode("ascii")
    )
    return bytes(output)


def main() -> None:
    markdown_files = find_markdown_files()

    all_lines: List[str] = []
    for path in markdown_files:
        rel_path = path.relative_to(ROOT).as_posix()
        header = f"File: {rel_path}"
        all_lines.append(header)
        all_lines.append("=" * min(len(header), 90))
        all_lines.append("")
        content = path.read_text(encoding="utf-8")
        all_lines.extend(markdown_to_plain_lines(content))
        all_lines.append("")

    pages = chunk_lines(all_lines)
    pdf_bytes = create_pdf(pages)
    OUTPUT_FILE.write_bytes(pdf_bytes)
    print(f"Wrote {OUTPUT_FILE} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
