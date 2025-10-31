from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Sequence, Tuple

PAGE_WIDTH = 612  # 8.5in * 72pt
PAGE_HEIGHT = 792  # 11in * 72pt
MARGIN = 50
LINE_HEIGHT = 14
FONT_SIZE = 11
FONT_KEY = "F1"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "ovos-technical-manual-documentation.pdf"


def iter_markdown_files(base: Path) -> List[Path]:
    markdown_files = sorted(
        [
            path
            for path in base.rglob("*.md")
            if ".git" not in path.parts and not any(part.startswith(".") for part in path.parts)
        ]
    )
    return markdown_files


def gather_lines(files: Sequence[Path]) -> List[str]:
    lines: List[str] = []
    for path in files:
        relative = path.relative_to(REPO_ROOT)
        lines.append(f"# {relative.as_posix()}")
        lines.append("")
        text = path.read_text(encoding="utf-8")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines.extend(text.split("\n"))
        lines.append("")
    return lines


def wrap_lines(lines: Sequence[str], width: int = 90) -> List[str]:
    wrapped: List[str] = []
    for line in lines:
        if line == "":
            wrapped.append("")
            continue
        segments = textwrap.wrap(
            line,
            width=width,
            expand_tabs=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        if not segments:
            wrapped.append("")
        else:
            wrapped.extend(segments)
    return wrapped


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def encode_win_ansi(text: str) -> str:
    return text.encode("cp1252", "replace").decode("cp1252")


def paginate_lines(lines: Sequence[str]) -> List[List[Tuple[str, int]]]:
    pages: List[List[Tuple[str, int]]] = []
    y = PAGE_HEIGHT - MARGIN
    current: List[Tuple[str, int]] = []
    for line in lines:
        if y < MARGIN:
            pages.append(current)
            current = []
            y = PAGE_HEIGHT - MARGIN
        current.append((line, y))
        y -= LINE_HEIGHT
    if current:
        pages.append(current)
    return pages


def build_content_stream(page_lines: Sequence[Tuple[str, int]]) -> bytes:
    parts = [f"BT\n/{FONT_KEY} {FONT_SIZE} Tf\n"]
    for line, y in page_lines:
        encoded_line = encode_win_ansi(escape_pdf_text(line))
        parts.append(f"1 0 0 1 {MARGIN} {y} Tm ({encoded_line}) Tj\n")
    parts.append("ET\n")
    content_bytes = "".join(parts).encode("cp1252")
    return f"<< /Length {len(content_bytes)} >>\nstream\n".encode("latin-1") + content_bytes + b"endstream\n"


def build_pdf(pages: Sequence[List[Tuple[str, int]]]) -> bytes:
    num_pages = len(pages)
    if num_pages == 0:
        raise ValueError("No content to render")

    catalog_id = 1
    pages_id = 2
    page_ids = [3 + i for i in range(num_pages)]
    content_ids = [3 + num_pages + i for i in range(num_pages)]
    font_id = 3 + 2 * num_pages

    objects: List[bytes] = []

    # 1: Catalog
    objects.append(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    # 2: Pages container (placeholder, fill later)
    objects.append(b"")

    # 3..: Page objects (placeholders)
    for _ in page_ids:
        objects.append(b"")

    # Content stream objects appended later
    for page_lines in pages:
        objects.append(build_content_stream(page_lines))

    # Font object
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Fill pages object
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Count {num_pages} /Kids [ {kids} ] >>".encode("latin-1")
    )

    # Fill each page object
    for index, page_id in enumerate(page_ids):
        content_ref = f"{content_ids[index]} 0 R"
        objects[page_id - 1] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Contents {content_ref} /Resources << /Font << /{FONT_KEY} {font_id} 0 R >> >> >>".encode(
                "latin-1"
            )
        )

    # Build PDF structure
    pdf_parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    xref_positions = []

    for obj_id, obj in enumerate(objects, start=1):
        xref_positions.append(sum(len(part) for part in pdf_parts))
        pdf_parts.append(f"{obj_id} 0 obj\n".encode("latin-1"))
        pdf_parts.append(obj + b"\n")
        pdf_parts.append(b"endobj\n")

    xref_offset = sum(len(part) for part in pdf_parts)
    pdf_parts.append(b"xref\n")
    pdf_parts.append(f"0 {len(objects) + 1}\n".encode("latin-1"))
    pdf_parts.append(b"0000000000 65535 f \n")
    for pos in xref_positions:
        pdf_parts.append(f"{pos:010} 00000 n \n".encode("latin-1"))

    pdf_parts.append(b"trailer\n")
    pdf_parts.append(
        f"<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n".encode("latin-1")
    )
    pdf_parts.append(b"startxref\n")
    pdf_parts.append(f"{xref_offset}\n".encode("latin-1"))
    pdf_parts.append(b"%%EOF\n")

    return b"".join(pdf_parts)


def main() -> None:
    markdown_files = iter_markdown_files(REPO_ROOT)
    if not markdown_files:
        raise SystemExit("No markdown files found")
    raw_lines = gather_lines(markdown_files)
    wrapped_lines = wrap_lines(raw_lines)
    pages = paginate_lines(wrapped_lines)
    pdf_bytes = build_pdf(pages)
    OUTPUT_PATH.write_bytes(pdf_bytes)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} with {len(pages)} pages")


if __name__ == "__main__":
    main()
