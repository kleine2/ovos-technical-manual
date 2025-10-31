#!/usr/bin/env python3
"""Combine repository documentation into a single PDF file."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "documentation.pdf"
PAGE_WIDTH = 612  # 8.5in * 72
PAGE_HEIGHT = 792  # 11in * 72
LEFT_MARGIN = 72
TOP_MARGIN = 72
FONT_SIZE = 11
LINE_HEIGHT = 12  # points
MAX_LINE_WIDTH = 95
LINES_PER_PAGE = math.floor((PAGE_HEIGHT - 2 * TOP_MARGIN) / LINE_HEIGHT)


def escape_pdf_text(value: str) -> str:
    """Escape characters that are special in PDF text objects."""
    return (
        value.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def normalize_line(text: str) -> str:
    """Ensure line text fits PDF encoding constraints."""
    cleaned = text.replace("\t", "    ")
    return cleaned.encode("latin-1", "replace").decode("latin-1")


def wrap_line(text: str, width: int = MAX_LINE_WIDTH) -> List[str]:
    if len(text) <= width:
        return [text]
    wrapped: List[str] = []
    start = 0
    while start < len(text):
        wrapped.append(text[start : start + width])
        start += width
    return wrapped


def gather_documentation_files() -> List[Path]:
    paths: List[Path] = []
    candidates = [ROOT / "README.md", ROOT / "docs", ROOT / "it"]
    for candidate in candidates:
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            for path in sorted(candidate.rglob("*.md")):
                paths.append(path)
    return [path for path in paths if path.exists()]


def iter_document_lines(paths: Iterable[Path]) -> Iterable[str]:
    first = True
    for path in paths:
        if not first:
            yield ""
        first = False
        relative = path.relative_to(ROOT).as_posix()
        title = f"=== {relative} ==="
        yield from wrap_and_normalize(title)
        yield ""
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.rstrip("\n")
                if raw_line.strip() == "":
                    yield ""
                    continue
                for chunk in wrap_and_normalize(raw_line):
                    yield chunk
        yield ""


def wrap_and_normalize(text: str) -> Iterable[str]:
    for part in wrap_line(normalize_line(text)):
        yield part


class SimplePDFBuilder:
    def __init__(self) -> None:
        self.objects: List[bytes | None] = []
        self.contents: List[int] = []
        self.font_object = self._add_object(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"
        )

    def add_page(self, lines: Iterable[str]) -> None:
        content_stream = self._build_content_stream(lines)
        content_id = self._add_stream(content_stream)
        self.contents.append(content_id)

    def write(self, destination: Path) -> None:
        if not self.contents:
            self.add_page(["Documentation is empty."])

        pages_object = self._reserve_object()
        page_ids: List[int] = []
        for content_id in self.contents:
            page_description = (
                f"<< /Type /Page /Parent {pages_object} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 {self.font_object} 0 R >> >> >>"
            )
            page_ids.append(self._add_object(page_description.encode("latin-1")))

        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        pages_description = (
            f"<< /Type /Pages /Kids [ {kids} ] /Count {len(page_ids)} >>"
        )
        self._set_object(pages_object, pages_description.encode("latin-1"))

        catalog_id = self._add_object(
            f"<< /Type /Catalog /Pages {pages_object} 0 R >>".encode("latin-1")
        )

        with destination.open("wb") as handle:
            handle.write(b"%PDF-1.4\n")
            offsets: List[int] = []
            for index, obj in enumerate(self.objects, start=1):
                if obj is None:
                    raise ValueError(f"Object {index} is undefined")
                offsets.append(handle.tell())
                handle.write(f"{index} 0 obj\n".encode("latin-1"))
                handle.write(obj + b"\n")
                handle.write(b"endobj\n")
            xref_position = handle.tell()
            handle.write(b"xref\n")
            total_objects = len(self.objects) + 1
            handle.write(f"0 {total_objects}\n".encode("latin-1"))
            handle.write(b"0000000000 65535 f \n")
            for offset in offsets:
                handle.write(f"{offset:010} 00000 n \n".encode("latin-1"))
            handle.write(b"trailer\n")
            trailer = f"<< /Size {total_objects} /Root {catalog_id} 0 R >>\n"
            handle.write(trailer.encode("latin-1"))
            handle.write(b"startxref\n")
            handle.write(f"{xref_position}\n".encode("latin-1"))
            handle.write(b"%%EOF")

    def _build_content_stream(self, lines: Iterable[str]) -> bytes:
        parts = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LINE_HEIGHT} TL", f"{LEFT_MARGIN} {PAGE_HEIGHT - TOP_MARGIN} Td"]
        for line in lines:
            escaped = escape_pdf_text(line)
            parts.append(f"({escaped}) Tj")
            parts.append("T*")
        parts.append("ET")
        text = "\n".join(parts)
        return text.encode("latin-1")

    def _add_stream(self, data: bytes) -> int:
        stream = (
            f"<< /Length {len(data)} >>\n".encode("latin-1") + b"stream\n" + data + b"\nendstream"
        )
        return self._add_object(stream)

    def _add_object(self, data: bytes) -> int:
        self.objects.append(data)
        return len(self.objects)

    def _reserve_object(self) -> int:
        self.objects.append(None)
        return len(self.objects)

    def _set_object(self, obj_id: int, data: bytes) -> None:
        self.objects[obj_id - 1] = data


def chunk_lines(lines: Iterable[str], chunk_size: int) -> Iterable[List[str]]:
    chunk: List[str] = []
    for line in lines:
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
        chunk.append(line)
    if chunk:
        yield chunk


def main() -> None:
    paths = gather_documentation_files()
    line_iter = iter_document_lines(paths)
    pdf = SimplePDFBuilder()
    for chunk in chunk_lines(line_iter, LINES_PER_PAGE):
        pdf.add_page(chunk)
    pdf.write(OUTPUT_PATH)
    print(f"PDF created at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
