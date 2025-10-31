#!/usr/bin/env python3
"""Generate a styled PDF that collates the repository Markdown documentation."""
from __future__ import annotations

import argparse
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PAGE_WIDTH = 612  # 8.5in in points
PAGE_HEIGHT = 792  # 11in in points
LEFT_MARGIN = 72
RIGHT_MARGIN = 72
TOP_MARGIN = 72
BOTTOM_MARGIN = 72
TEXT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

# Layout tuning constants
PARAGRAPH_FONT_SIZE = 11
PARAGRAPH_LINE_HEIGHT = PARAGRAPH_FONT_SIZE * 1.5
SMALL_SPACING = PARAGRAPH_FONT_SIZE * 0.5


@dataclass
class LinkTarget:
    """Represents a hyperlink target."""

    url: Optional[str] = None
    anchor: Optional[str] = None

    @property
    def is_internal(self) -> bool:
        return self.anchor is not None


@dataclass
class InlineSegment:
    """Inline formatted piece of text."""

    text: str
    style: str = "regular"
    link: Optional[LinkTarget] = None


@dataclass
class Block:
    pass


@dataclass
class Heading(Block):
    level: int
    segments: List[InlineSegment]
    anchor_key: str


@dataclass
class Paragraph(Block):
    segments: List[InlineSegment]


@dataclass
class BulletList(Block):
    items: List[List[InlineSegment]]
    ordered: bool = False


@dataclass
class CodeBlock(Block):
    lines: List[str]


@dataclass
class Annotation:
    rect: Tuple[float, float, float, float]
    link_anchor: Optional[str] = None
    uri: Optional[str] = None


@dataclass
class Page:
    operations: List[str] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class Destination:
    page_index: int
    x: float
    y: float


def normalise_text(value: str) -> str:
    """Return a best-effort ASCII representation of ``value``."""

    normalised = unicodedata.normalize("NFKD", value)
    return normalised.encode("ascii", "ignore").decode("ascii")


def pdf_escape(value: str) -> str:
    """Escape characters that are special in PDF string literals."""

    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def slugify(value: str) -> str:
    text = normalise_text(value).lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "section"


def gather_markdown_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def split_tokens(text: str) -> List[str]:
    tokens = re.findall(r"\S+|\s+", text)
    return tokens or [""]


def measure_text(token: str, font_size: float, style: str) -> float:
    if not token:
        return 0.0
    base_width = font_size * 0.52
    if style == "code":
        base_width = font_size * 0.6
    elif style in {"bold", "heading"}:
        base_width = font_size * 0.56
    elif style == "italic":
        base_width = font_size * 0.52
    return base_width * len(token)


def format_float(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def wrap_segments(
    segments: List[InlineSegment],
    font_size: float,
    max_width: float,
    first_line_indent: float = 0.0,
    subsequent_indent: float = 0.0,
) -> List[List[InlineSegment]]:
    lines: List[List[InlineSegment]] = []
    current_line: List[InlineSegment] = []
    current_width = 0.0
    available = max(max_width - first_line_indent, font_size)
    subsequent_available = max(max_width - subsequent_indent, font_size)
    for segment in segments:
        pieces = split_tokens(segment.text)
        for piece in pieces:
            if piece == "":
                continue
            width = measure_text(piece, font_size, segment.style)
            if piece.strip() == "" and not current_line:
                continue
            if current_line and current_width + width > available and piece.strip():
                lines.append(current_line)
                current_line = []
                current_width = 0.0
                available = subsequent_available
            if width > available and not current_line:
                slice_width = max(1, int(len(piece) * available / width))
                slice_width = max(1, slice_width)
                for start in range(0, len(piece), slice_width):
                    part = piece[start : start + slice_width]
                    part_width = measure_text(part, font_size, segment.style)
                    current_line.append(
                        InlineSegment(part, style=segment.style, link=segment.link)
                    )
                    current_width += part_width
                    lines.append(current_line)
                    current_line = []
                    current_width = 0.0
                    available = subsequent_available
                continue
            current_line.append(InlineSegment(piece, style=segment.style, link=segment.link))
            current_width += width
    if current_line:
        lines.append(current_line)
    return lines


class LayoutState:
    def __init__(self) -> None:
        self.pages: List[Page] = [Page()]
        self.cursor_y = PAGE_HEIGHT - TOP_MARGIN
        self.anchor_positions: Dict[str, Destination] = {}

    @property
    def current_page(self) -> Page:
        return self.pages[-1]

    def new_page(self) -> None:
        self.pages.append(Page())
        self.cursor_y = PAGE_HEIGHT - TOP_MARGIN

    def ensure_space(self, required_height: float) -> None:
        if self.cursor_y - required_height < BOTTOM_MARGIN:
            self.new_page()

    def add_vertical_space(self, amount: float) -> None:
        if amount <= 0:
            return
        self.ensure_space(amount)
        self.cursor_y -= amount

    def begin_line(self, line_height: float) -> float:
        self.ensure_space(line_height)
        self.cursor_y -= line_height
        return self.cursor_y

    def record_anchor(self, anchor_key: str, x: float, y: float) -> None:
        self.anchor_positions[anchor_key] = Destination(len(self.pages) - 1, x, y)


def build_annotation(rect: Tuple[float, float, float, float], segment: InlineSegment) -> Optional[Annotation]:
    if segment.link is None:
        return None
    link = segment.link
    if link.is_internal:
        return Annotation(rect=rect, link_anchor=link.anchor)
    if link.url:
        return Annotation(rect=rect, uri=link.url)
    return None


def add_text_operations(
    page: Page,
    x: float,
    y: float,
    segments: List[InlineSegment],
    font_name: str,
    font_size: float,
) -> List[Annotation]:
    annotations: List[Annotation] = []
    operations = [
        "BT",
        f"/{font_name} {format_float(font_size)} Tf",
        f"1 0 0 1 {format_float(x)} {format_float(y)} Tm",
    ]
    active_font = font_name
    cursor_x = x
    for piece in segments:
        text = pdf_escape(normalise_text(piece.text))
        if text == "":
            continue
        if piece.style == "code":
            desired_font = "F3"
        elif piece.style in {"bold", "heading"}:
            desired_font = "F2"
        elif piece.style == "italic":
            desired_font = "F4"
        else:
            desired_font = "F1"
        if desired_font != active_font:
            operations.append(f"/{desired_font} {format_float(font_size)} Tf")
            active_font = desired_font
        if piece.link and piece.link.url:
            operations.append("0 0.2 0.6 rg")
        operations.append(f"({text}) Tj")
        if piece.link and piece.link.url:
            operations.append("0 g")
        width = measure_text(piece.text, font_size, piece.style if piece.style != "heading" else "bold")
        left = cursor_x
        right = cursor_x + width
        top = y + font_size * 0.8
        bottom = y - font_size * 0.2
        annotation = build_annotation((left, bottom, right, top), piece)
        if annotation:
            annotations.append(annotation)
        cursor_x += width
    operations.append("ET")
    page.operations.extend(operations)
    return annotations


def render_paragraph(state: LayoutState, segments: List[InlineSegment]) -> None:
    lines = wrap_segments(segments, PARAGRAPH_FONT_SIZE, TEXT_WIDTH)
    for i, line in enumerate(lines):
        y = state.begin_line(PARAGRAPH_LINE_HEIGHT)
        annotations = add_text_operations(
            state.current_page,
            LEFT_MARGIN,
            y,
            line,
            "F1",
            PARAGRAPH_FONT_SIZE,
        )
        state.current_page.annotations.extend(annotations)
    state.add_vertical_space(SMALL_SPACING)


def render_heading(state: LayoutState, block: Heading) -> None:
    font_sizes = {1: 24, 2: 20, 3: 18, 4: 16, 5: 14, 6: 13}
    font_size = font_sizes.get(block.level, 13)
    line_height = font_size * 1.4
    segments = [InlineSegment(seg.text, style="heading", link=seg.link) for seg in block.segments]
    lines = wrap_segments(segments, font_size, TEXT_WIDTH)
    state.add_vertical_space(font_size * 0.6)
    first_line_y = None
    for line in lines:
        y = state.begin_line(line_height)
        if first_line_y is None:
            first_line_y = y + font_size * 0.8
            state.record_anchor(block.anchor_key, LEFT_MARGIN, first_line_y)
        annotations = add_text_operations(state.current_page, LEFT_MARGIN, y, line, "F2", font_size)
        state.current_page.annotations.extend(annotations)
    state.add_vertical_space(font_size * 0.3)


def render_bullet_list(state: LayoutState, block: BulletList) -> None:
    font_size = PARAGRAPH_FONT_SIZE
    line_height = PARAGRAPH_FONT_SIZE * 1.5
    bullet_offset = 18
    for index, item in enumerate(block.items, start=1):
        marker = f"{index}." if block.ordered else "•"
        state.add_vertical_space(SMALL_SPACING)
        lines = wrap_segments(item, font_size, TEXT_WIDTH - bullet_offset)
        for line_index, line in enumerate(lines):
            y = state.begin_line(line_height)
            text_x = LEFT_MARGIN + bullet_offset
            annotations = add_text_operations(state.current_page, text_x, y, line, "F1", font_size)
            state.current_page.annotations.extend(annotations)
            if line_index == 0:
                add_text_operations(
                    state.current_page,
                    LEFT_MARGIN,
                    y,
                    [InlineSegment(marker, style="bold")],
                    "F2",
                    font_size,
                )
        state.add_vertical_space(SMALL_SPACING)


def render_code_block(state: LayoutState, block: CodeBlock) -> None:
    font_size = 10
    line_height = font_size * 1.6
    padding = 6
    block_height = line_height * len(block.lines) + padding * 2
    state.add_vertical_space(SMALL_SPACING)
    state.ensure_space(block_height)
    start_y = state.cursor_y
    x_left = LEFT_MARGIN - 4
    page = state.current_page
    page.operations.extend(
        [
            "0.95 g",
            f"{format_float(x_left)} {format_float(start_y - block_height)} {format_float(TEXT_WIDTH + 8)} {format_float(block_height)} re",
            "f",
            "0 g",
        ]
    )
    state.cursor_y = start_y - padding
    for line in block.lines:
        y = state.begin_line(line_height)
        segments = [InlineSegment(line.expandtabs(4), style="code")]
        annotations = add_text_operations(page, LEFT_MARGIN, y, segments, "F3", font_size)
        page.annotations.extend(annotations)
    state.cursor_y -= padding
    state.add_vertical_space(SMALL_SPACING)


def parse_inline(
    text: str,
    current_file: str,
    root: Path,
    anchor_registry: Dict[Tuple[str, str], str],
    file_top_anchors: Dict[str, str],
) -> List[InlineSegment]:
    segments: List[InlineSegment] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        if buffer:
            segments.append(InlineSegment("".join(buffer)))
            buffer.clear()

    i = 0
    length = len(text)
    while i < length:
        char = text[i]
        if char == "`":
            end = text.find("`", i + 1)
            if end == -1:
                buffer.append(char)
                i += 1
                continue
            flush_buffer()
            segments.append(InlineSegment(text[i + 1 : end], style="code"))
            i = end + 1
            continue
        if text.startswith("**", i):
            end = text.find("**", i + 2)
            if end == -1:
                buffer.append(char)
                i += 1
                continue
            flush_buffer()
            segments.append(InlineSegment(text[i + 2 : end], style="bold"))
            i = end + 2
            continue
        if char == "*":
            end = text.find("*", i + 1)
            if end == -1:
                buffer.append(char)
                i += 1
                continue
            flush_buffer()
            segments.append(InlineSegment(text[i + 1 : end], style="italic"))
            i = end + 1
            continue
        if char == "[":
            end_label = text.find("]", i + 1)
            if end_label != -1 and end_label + 1 < length and text[end_label + 1] == "(":
                end_dest = text.find(")", end_label + 2)
                if end_dest != -1:
                    flush_buffer()
                    label = text[i + 1 : end_label]
                    destination = text[end_label + 2 : end_dest]
                    link = resolve_link(destination, current_file, root, anchor_registry, file_top_anchors)
                    segments.append(InlineSegment(label, link=link))
                    i = end_dest + 1
                    continue
        buffer.append(char)
        i += 1

    flush_buffer()
    return segments or [InlineSegment("")]


def resolve_link(
    destination: str,
    current_file: str,
    root: Path,
    anchor_registry: Dict[Tuple[str, str], str],
    file_top_anchors: Dict[str, str],
) -> LinkTarget:
    destination = destination.strip()
    if destination.startswith("#"):
        target = destination[1:]
        if not target:
            top_anchor = file_top_anchors.get(current_file)
            if top_anchor:
                return LinkTarget(anchor=top_anchor)
            return LinkTarget()
        slug = slugify(target)
        anchor_key = anchor_registry.get((current_file, slug))
        if anchor_key:
            return LinkTarget(anchor=anchor_key)
        return LinkTarget()
    if destination.lower().endswith(".md") or ".md#" in destination.lower():
        path_part, _, anchor_part = destination.partition("#")
        target_path = (root / Path(current_file).parent / path_part).resolve()
        try:
            relative = target_path.relative_to(root)
        except ValueError:
            return LinkTarget(url=destination)
        file_key = relative.as_posix()
        if anchor_part:
            slug = slugify(anchor_part)
            anchor_key = anchor_registry.get((file_key, slug))
            if anchor_key:
                return LinkTarget(anchor=anchor_key)
        top_anchor = file_top_anchors.get(file_key)
        if top_anchor:
            return LinkTarget(anchor=top_anchor)
        return LinkTarget()
    return LinkTarget(url=destination)


def register_anchor(
    file_key: str,
    title: str,
    anchor_registry: Dict[Tuple[str, str], str],
) -> str:
    base_slug = slugify(title)
    slug = base_slug
    index = 1
    while (file_key, slug) in anchor_registry:
        slug = f"{base_slug}-{index}"
        index += 1
    anchor_key = f"{file_key}#{slug}"
    anchor_registry[(file_key, slug)] = anchor_key
    return anchor_key


def scan_headings(
    path: Path,
    root: Path,
    anchor_registry: Dict[Tuple[str, str], str],
) -> Tuple[List[str], Dict[int, Tuple[int, str, bool, str]]]:
    current_file = path.relative_to(root).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    heading_map: Dict[int, Tuple[int, str, bool, str]] = {}
    in_code = False
    fence_char: Optional[str] = None
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker_char = stripped[0]
            if not in_code:
                in_code = True
                fence_char = marker_char
            elif fence_char == marker_char:
                in_code = False
                fence_char = None
            continue
        if in_code:
            continue
        atx = re.match(r"^(#{1,6})\s+(.*)$", raw_line)
        if atx:
            level = len(atx.group(1))
            heading_text = atx.group(2).strip()
            anchor_key = register_anchor(current_file, heading_text, anchor_registry)
            slug = anchor_key.split("#", 1)[1]
            heading_map[idx] = (level, heading_text, False, slug)
            continue
        if idx + 1 < len(lines) and raw_line.strip():
            underline = lines[idx + 1].strip()
            if underline and set(underline) <= {"=", "-"} and len(underline) >= 3:
                level = 1 if underline[0] == "=" else 2
                heading_text = raw_line.strip()
                anchor_key = register_anchor(current_file, heading_text, anchor_registry)
                slug = anchor_key.split("#", 1)[1]
                heading_map[idx] = (level, heading_text, True, slug)
    return lines, heading_map


def parse_markdown_file(
    current_file: str,
    lines: List[str],
    root: Path,
    anchor_registry: Dict[Tuple[str, str], str],
    file_top_anchors: Dict[str, str],
    heading_map: Dict[int, Tuple[int, str, bool, str]],
) -> List[Block]:
    blocks: List[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence_char = stripped[0]
            code_lines: List[str] = []
            i += 1
            while i < len(lines):
                inner = lines[i].strip()
                if inner.startswith(fence_char * 3):
                    break
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            blocks.append(CodeBlock(code_lines))
            continue
        if i in heading_map:
            level, heading_text, consumes_next, slug = heading_map[i]
            anchor_key = anchor_registry[(current_file, slug)]
            segments = parse_inline(
                heading_text,
                current_file,
                root,
                anchor_registry,
                file_top_anchors,
            )
            blocks.append(Heading(level=level, segments=segments, anchor_key=anchor_key))
            i += 1
            if consumes_next:
                i += 1
            continue
        list_match = re.match(r"^(\s*)([-*+]\s+|\d+\.\s+)(.*)$", line)
        if list_match:
            ordered = list_match.group(2).strip().endswith(".")
            items: List[List[InlineSegment]] = []
            while i < len(lines):
                match = re.match(r"^(\s*)([-*+]\s+|\d+\.\s+)(.*)$", lines[i])
                if not match:
                    break
                content = match.group(3).rstrip()
                segments = parse_inline(
                    content,
                    current_file,
                    root,
                    anchor_registry,
                    file_top_anchors,
                )
                items.append(segments)
                i += 1
            blocks.append(BulletList(items=items, ordered=ordered))
            continue
        if line.strip() == "":
            i += 1
            continue
        paragraph_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6})\s+", lines[i]) and not re.match(r"^(\s*)([-*+]\s+|\d+\.\s+)", lines[i]):
            paragraph_lines.append(lines[i])
            i += 1
        text = " ".join(l.strip() for l in paragraph_lines)
        segments = parse_inline(text, current_file, root, anchor_registry, file_top_anchors)
        blocks.append(Paragraph(segments=segments))
    return blocks


def build_document_structure(root: Path) -> Tuple[List[Block], Dict[str, Destination]]:
    markdown_files = gather_markdown_files(root)
    anchor_registry: Dict[Tuple[str, str], str] = {}
    file_top_anchors: Dict[str, str] = {}
    file_lines: Dict[str, List[str]] = {}
    heading_maps: Dict[str, Dict[int, Tuple[int, str, bool, str]]] = {}
    for path in markdown_files:
        relative = path.relative_to(root).as_posix()
        anchor_key = register_anchor(relative, relative, anchor_registry)
        file_top_anchors[relative] = anchor_key
    for path in markdown_files:
        relative = path.relative_to(root).as_posix()
        lines, heading_map = scan_headings(path, root, anchor_registry)
        file_lines[relative] = lines
        heading_maps[relative] = heading_map
    blocks: List[Block] = []
    for path in markdown_files:
        relative = path.relative_to(root).as_posix()
        anchor_key = file_top_anchors[relative]
        heading_segments = [InlineSegment(relative, style="bold")]
        blocks.append(Heading(level=1, segments=heading_segments, anchor_key=anchor_key))
        blocks.extend(
            parse_markdown_file(
                relative,
                file_lines[relative],
                root,
                anchor_registry,
                file_top_anchors,
                heading_maps[relative],
            )
        )
    state = LayoutState()
    for block in blocks:
        if isinstance(block, Heading):
            render_heading(state, block)
        elif isinstance(block, Paragraph):
            render_paragraph(state, block.segments)
        elif isinstance(block, BulletList):
            render_bullet_list(state, block)
        elif isinstance(block, CodeBlock):
            render_code_block(state, block)
    return blocks, state


class PDFBuilder:
    def __init__(self) -> None:
        self.objects: List[bytes | None] = [None]

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
            handle.write(f"<< /Size {len(self.objects)} /Root 1 0 R >>\n".encode("latin-1"))
            handle.write(b"startxref\n")
            handle.write(f"{xref_position}\n".encode("latin-1"))
            handle.write(b"%%EOF")


def create_documentation_pdf(output: Path, root: Path) -> None:
    _, layout = build_document_structure(root)
    pdf = PDFBuilder()
    catalog_id = pdf.add_object(None)
    pages_id = pdf.add_object(None)
    font_objects = {
        "F1": pdf.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        "F2": pdf.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"),
        "F3": pdf.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"),
        "F4": pdf.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>"),
    }
    page_entries: List[Tuple[int, int, Page]] = []
    for page in layout.pages:
        stream_data = "\n".join(page.operations).encode("latin-1")
        stream_content = (
            f"<< /Length {len(stream_data)} >>\n".encode("latin-1")
            + b"stream\n"
            + stream_data
            + b"\nendstream"
        )
        stream_id = pdf.add_object(stream_content)
        page_id = pdf.add_object(None)
        page_entries.append((page_id, stream_id, page))
    for index, (page_id, stream_id, page) in enumerate(page_entries):
        annot_refs: List[str] = []
        for annotation in page.annotations:
            if annotation.link_anchor:
                dest = layout.anchor_positions.get(annotation.link_anchor)
                if not dest:
                    continue
                target_page_id = page_entries[dest.page_index][0]
                dest_array = (
                    f"[{target_page_id} 0 R /XYZ {format_float(dest.x)} {format_float(dest.y)} null]"
                )
                annot_dict = (
                    "<< /Type /Annot /Subtype /Link "
                    f"/Rect [{format_float(annotation.rect[0])} {format_float(annotation.rect[1])} {format_float(annotation.rect[2])} {format_float(annotation.rect[3])}] "
                    "/Border [0 0 0] "
                    f"/A << /S /GoTo /D {dest_array} >> >>"
                )
            elif annotation.uri:
                annot_dict = (
                    "<< /Type /Annot /Subtype /Link "
                    f"/Rect [{format_float(annotation.rect[0])} {format_float(annotation.rect[1])} {format_float(annotation.rect[2])} {format_float(annotation.rect[3])}] "
                    "/Border [0 0 0] "
                    f"/A << /S /URI /URI ({pdf_escape(annotation.uri)}) >> >>"
                )
            else:
                continue
            annot_id = pdf.add_object(annot_dict.encode("latin-1"))
            annot_refs.append(f"{annot_id} 0 R")
        fonts_dict = " ".join(f"/{name} {obj_id} 0 R" for name, obj_id in font_objects.items())
        page_dict = [
            "<< /Type /Page",
            f"/Parent {pages_id} 0 R",
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}]",
            f"/Resources << /Font << {fonts_dict} >> >>",
            f"/Contents {stream_id} 0 R",
        ]
        if annot_refs:
            page_dict.append(f"/Annots [{' '.join(annot_refs)}]")
        page_dict.append(">>")
        pdf.set_object(page_id, "\n".join(page_dict).encode("latin-1"))
    kids = " ".join(f"{page_id} 0 R" for page_id, _, _ in page_entries)
    pages_dict = f"<< /Type /Pages /Count {len(page_entries)} /Kids [{kids}] >>"
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
