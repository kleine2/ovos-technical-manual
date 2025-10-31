#!/usr/bin/env python3
"""Generate a PDF that concatenates all Markdown documentation in the repository.

This script traverses the repository looking for `.md` files and emits a
single PDF where each file's contents are rendered as pre-formatted text.
The PDF is built manually to avoid external dependencies, which keeps the
script runnable in constrained environments.
"""
from __future__ import annotations

import argparse
import textwrap
import unicodedata
from pathlib import Path
from typing import Iterable, List

PAGE_WIDTH = 612  # 8.5in in points
PAGE_HEIGHT = 792  # 11in in points
LEFT_MARGIN = 72
TOP_MARGIN = 72
BOTTOM_MARGIN = 72
LINE_HEIGHT = 12
CHARS_PER_LINE = 95
LINES_PER_PAGE = int((PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) // LINE_HEIGHT)


def normalise_text(value: str) -> str:
    """Return a best-effort ASCII representation of ``value``."""
    normalised = unicodedata.normalize("NFKD", value)
    return normalised.encode("ascii", "ignore").decode("ascii")


def pdf_escape(value: str) -> str:
    """Escape characters that are special in PDF string literals."""
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PDFBuilder:
    """Minimal PDF writer that supports text-only pages."""

    def __init__(self) -> None:
        self.objects: List[bytes | None] = [None]  # 1-indexed storage

    def add_object(self, content: bytes | None) -> int:
        self.objects.append(content)
        return len(self.objects) - 1

    def set_object(self, obj_id: int, content: bytes) -> None:
        self.objects[obj_id] = content

    def build(self, output_path: Path) -> None:
        with output_path.open("wb") as handle:
            handle.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
            offsets: List[int] = []
            for obj_id, obj in enumerate(self.objects[1:], start=1):
                if obj is None:
                    raise ValueError(f"Object {obj_id} has no content")
                offsets.append(handle.tell())
                handle.write(f"{obj_id} 0 obj\n".encode("latin-1"))
                handle.write(obj)
                if not obj.endswith(b"\n"):
                    handle.write(b"\n")
                handle.write(b"endobj\n")

            xref_position = handle.tell()
            handle.write(f"xref\n0 {len(self.objects)}\n".encode("latin-1"))
            handle.write(b"0000000000 65535 f \n")
            for offset in offsets:
                handle.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
            handle.write(b"trailer\n")
            handle.write(
                f"<< /Size {len(self.objects)} /Root 1 0 R >>\n".encode("latin-1")
            )
            handle.write(b"startxref\n")
            handle.write(f"{xref_position}\n".encode("latin-1"))
            handle.write(b"%%EOF")


def chunked(iterable: Iterable[str], chunk_size: int) -> Iterable[List[str]]:
    chunk: List[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def gather_markdown_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def wrap_lines_for_pdf(lines: Iterable[str]) -> List[str]:
    wrapped: List[str] = []
    for line in lines:
        clean = normalise_text(line.rstrip("\n").replace("\t", "    "))
        if not clean:
            wrapped.append("")
            continue
        segments = textwrap.wrap(clean, width=CHARS_PER_LINE) or [""]
        wrapped.extend(segments)
    return wrapped


def build_page_stream(lines: Iterable[str]) -> bytes:
    data_lines = ["BT", "/F1 10 Tf", f"{LINE_HEIGHT} TL", f"{LEFT_MARGIN} {PAGE_HEIGHT - TOP_MARGIN} Td"]
    first = True
    for line in lines:
        escaped = pdf_escape(line)
        if first:
            data_lines.append(f"({escaped}) Tj")
            first = False
            continue
        data_lines.append("T*")
        data_lines.append(f"({escaped}) Tj")
    data_lines.append("ET")
    text_data = "\n".join(data_lines).encode("latin-1")
    return (
        f"<< /Length {len(text_data)} >>\n".encode("latin-1")
        + b"stream\n"
        + text_data
        + b"\nendstream"
    )


def create_documentation_pdf(output: Path, root: Path) -> None:
    pdf = PDFBuilder()
    catalog_id = pdf.add_object(None)
    pages_id = pdf.add_object(None)
    font_id = pdf.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    markdown_files = gather_markdown_files(root)
    all_lines: List[str] = []
    for md_path in markdown_files:
        relative = md_path.relative_to(root)
        all_lines.append(f"===== {relative.as_posix()} =====")
        all_lines.append("")
        contents = md_path.read_text(encoding="utf-8")
        all_lines.extend(wrap_lines_for_pdf(contents.splitlines()))
        all_lines.append("")

    page_ids: List[int] = []
    for lines in chunked(all_lines, LINES_PER_PAGE):
        stream_id = pdf.add_object(build_page_stream(lines))
        page_id = pdf.add_object(None)
        page_ids.append(page_id)
        page_dict = (
            "<< /Type /Page /Parent {parent} 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /Font << /F1 {font} 0 R >> >> /Contents {contents} 0 R >>"
        ).format(
            parent=pages_id,
            width=PAGE_WIDTH,
            height=PAGE_HEIGHT,
            font=font_id,
            contents=stream_id,
        )
        pdf.set_object(page_id, page_dict.encode("latin-1"))

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_dict = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>"
    catalog_dict = f"<< /Type /Catalog /Pages {pages_id} 0 R >>"

    pdf.set_object(pages_id, pages_dict.encode("latin-1"))
    pdf.set_object(catalog_id, catalog_dict.encode("latin-1"))

    pdf.build(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("documentation.pdf"),
        help="Destination PDF file (default: documentation.pdf)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root containing Markdown documentation.",
    )
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    create_documentation_pdf(options.output, options.root)


if __name__ == "__main__":
    main()
