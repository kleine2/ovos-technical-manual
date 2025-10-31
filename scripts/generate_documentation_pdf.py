#!/usr/bin/env python3
"""Render the repository documentation into a styled, cross-linked PDF."""

from __future__ import annotations

import argparse
import re
import textwrap
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


PAGE_WIDTH = 612  # 8.5 in
PAGE_HEIGHT = 792  # 11 in
LEFT_MARGIN = 72
RIGHT_MARGIN = 72
TOP_MARGIN = 72
BOTTOM_MARGIN = 72


# ---------------------------------------------------------------------------
# Utility helpers


def slugify(value: str) -> str:
    """Create a slug suitable for anchors from ``value``."""

    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-")
    return slug.lower()


def pdf_escape(text: str) -> str:
    """Escape characters used in PDF string literals."""

    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def estimate_text_width(text: str, font: str, size: float) -> float:
    """Approximate the rendered width of ``text`` in points."""

    if not text:
        return 0.0

    if font == "Courier":
        return len(text) * size * 0.6

    # Helvetica (and Helvetica-Bold) average width approximation.
    return len(text) * size * 0.52


SPECIAL_REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": "--",
    "\u2015": "--",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2022": "*",
    "\u2026": "...",
    "\u00a0": " ",
}


def sanitize_text(value: str) -> str:
    """Return a PDF-safe text representation using ASCII characters."""

    if not value:
        return ""

    normalized = unicodedata.normalize("NFKD", value)
    result: List[str] = []
    for char in normalized:
        if char in SPECIAL_REPLACEMENTS:
            result.append(SPECIAL_REPLACEMENTS[char])
            continue
        codepoint = ord(char)
        if 32 <= codepoint < 127:
            result.append(char)
            continue
        if char == "\t":
            result.append("    ")
            continue
        ascii_equiv = char.encode("ascii", "ignore").decode("ascii")
        if ascii_equiv:
            result.append(ascii_equiv)
    return "".join(result)


# ---------------------------------------------------------------------------
# Markdown parsing


@dataclass
class InlineSegment:
    text: str
    kind: str = "text"  # "text" | "code"
    link: Optional[str] = None  # internal:anchor or external:url


@dataclass
class Block:
    type: str
    text: str = ""
    level: int = 0
    items: List[List[InlineSegment]] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)
    inline: List[InlineSegment] = field(default_factory=list)
    anchor: Optional[str] = None


def parse_inline(text: str, resolve_link) -> List[InlineSegment]:
    segments: List[InlineSegment] = []
    i = 0
    while i < len(text):
        if text[i] == "`":
            closing = text.find("`", i + 1)
            if closing == -1:
                segments.append(InlineSegment(text=text[i:], kind="text"))
                break
            code_text = text[i + 1 : closing]
            segments.append(InlineSegment(text=code_text, kind="code"))
            i = closing + 1
            continue
        if text[i] == "[":
            closing = text.find("]", i + 1)
            if closing != -1 and closing + 1 < len(text) and text[closing + 1] == "(":
                end = text.find(")", closing + 2)
                if end != -1:
                    label = text[i + 1 : closing]
                    target = text[closing + 2 : end]
                    link_target = resolve_link(target)
                    segments.append(
                        InlineSegment(text=label, kind="text", link=link_target)
                    )
                    i = end + 1
                    continue
        segments.append(InlineSegment(text=text[i], kind="text"))
        i += 1

    # Merge adjacent segments of the same type/link to simplify wrapping.
    merged: List[InlineSegment] = []
    for seg in segments:
        if (
            merged
            and merged[-1].kind == seg.kind
            and merged[-1].link == seg.link
        ):
            merged[-1].text += seg.text
        else:
            merged.append(seg)
    return merged


def parse_markdown(text: str, anchor_prefix: str, resolve_link) -> List[Block]:
    blocks: List[Block] = []
    lines = text.splitlines()
    i = 0
    in_code = False
    code_fence = ""
    code_lines: List[str] = []
    current_list: Optional[Block] = None
    paragraph_lines: List[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            paragraph = " ".join(line.strip() for line in paragraph_lines)
            inline = parse_inline(paragraph, resolve_link)
            blocks.append(Block(type="paragraph", inline=inline))
            paragraph_lines = []

    def flush_list() -> None:
        nonlocal current_list
        if current_list is not None:
            blocks.append(current_list)
            current_list = None

    while i < len(lines):
        line = lines[i]

        if not in_code and line.strip().startswith("```"):
            flush_paragraph()
            flush_list()
            in_code = True
            code_fence = line.strip()
            code_lines = []
            i += 1
            continue

        if in_code:
            if line.strip() == code_fence:
                blocks.append(Block(type="code", lines=code_lines.copy()))
                in_code = False
            else:
                code_lines.append(line)
            i += 1
            continue

        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            flush_list()
            level = len(heading_match.group(1))
            text_content = heading_match.group(2).strip()
            anchor = f"{anchor_prefix}#{slugify(text_content)}"
            inline = parse_inline(text_content, resolve_link)
            blocks.append(
                Block(
                    type="heading",
                    level=level,
                    text=text_content,
                    inline=inline,
                    anchor=anchor,
                )
            )
            i += 1
            continue

        list_match = re.match(r"^(\s*)([-*+]\s+|\d+\.\s+)(.*)$", line)
        if list_match and list_match.group(1) == "":
            flush_paragraph()
            marker = list_match.group(2)
            text_after = list_match.group(3)
            ordered = marker[0].isdigit()
            if current_list is None or (current_list.type == "olist") != ordered:
                flush_list()
                current_list = Block(type="olist" if ordered else "ulist", items=[])
            inline = parse_inline(text_after.strip(), resolve_link)
            current_list.items.append(inline)
            i += 1
            continue

        # Continuation of list item (indented)
        if current_list is not None and line.startswith("    "):
            inline = parse_inline(line.strip(), resolve_link)
            current_list.items[-1].extend([InlineSegment(text=" ")] + inline)
            i += 1
            continue

        paragraph_lines.append(line)
        i += 1

    flush_paragraph()
    flush_list()
    if in_code:
        blocks.append(Block(type="code", lines=code_lines.copy()))

    return blocks


# ---------------------------------------------------------------------------
# Layout


@dataclass
class PositionedSegment:
    text: str
    font: str
    size: float
    color: Tuple[float, float, float]
    x: float
    width: float
    link: Optional[str] = None


@dataclass
class Line:
    y: float
    leading: float
    segments: List[PositionedSegment]
    anchor: Optional[str] = None


@dataclass
class Annotation:
    rect: Tuple[float, float, float, float]
    target: str


@dataclass
class Page:
    lines: List[Line] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)


def wrap_inline_segments(
    segments: Sequence[InlineSegment],
    max_width: float,
    font: str,
    size: float,
) -> List[List[InlineSegment]]:
    lines: List[List[InlineSegment]] = []
    current: List[InlineSegment] = []
    width = 0.0

    def flush_line() -> None:
        nonlocal current, width
        if current:
            lines.append(current)
            current = []
            width = 0.0

    for segment in segments:
        tokens = re.findall(r"\S+|\s+", segment.text)
        for token in tokens:
            clean_token = token
            if token.strip():
                clean_token = sanitize_text(token)
            if token.strip() and not clean_token:
                continue
            token_seg = InlineSegment(text=clean_token, kind=segment.kind, link=segment.link)
            token_font = font
            token_size = size
            if token_seg.kind == "code":
                token_font = STYLE_DEFS["inline_code"]["font"]
                token_size = STYLE_DEFS["inline_code"]["size"]
            token_width = estimate_text_width(token_seg.text, token_font, token_size)

            if token_seg.text.strip() == "" and not current:
                # Discard leading whitespace on a new line.
                continue

            if width + token_width > max_width and current:
                flush_line()
                if token_seg.text.strip() == "":
                    continue

            if token_width > max_width:
                # Break overly long tokens.
                remaining = token_seg.text
                while remaining:
                    slice_width = 0.0
                    slice_text = ""
                    for char in remaining:
                        char_width = estimate_text_width(char, token_font, token_size)
                        if slice_width + char_width > max_width and slice_text:
                            break
                        slice_text += char
                        slice_width += char_width
                    current.append(
                        InlineSegment(
                            text=slice_text, kind=token_seg.kind, link=token_seg.link
                        )
                    )
                    flush_line()
                    remaining = remaining[len(slice_text) :]
                width = 0.0
                continue

            current.append(token_seg)
            width += token_width

    if current:
        lines.append(current)

    return lines


def inline_to_positioned(
    inline_segments: Sequence[InlineSegment],
    font: str,
    size: float,
    color: Tuple[float, float, float],
    indent: float,
) -> Tuple[List[PositionedSegment], float]:
    positioned: List[PositionedSegment] = []
    x_offset = LEFT_MARGIN + indent
    for segment in inline_segments:
        clean_text = segment.text
        if clean_text.strip():
            clean_text = sanitize_text(clean_text)
        if not clean_text:
            continue
        width = estimate_text_width(clean_text, font, size)
        positioned.append(
            PositionedSegment(
                text=clean_text,
                font=font,
                size=size,
                color=color,
                x=x_offset,
                width=width,
                link=segment.link,
            )
        )
        x_offset += width
    total_width = x_offset - LEFT_MARGIN - indent
    return positioned, total_width


STYLE_DEFS = {
    "body": {"font": "Helvetica", "size": 11.0, "leading": 15.0, "color": (0, 0, 0)},
    "heading1": {
        "font": "Helvetica-Bold",
        "size": 24.0,
        "leading": 32.0,
        "color": (0.1, 0.1, 0.1),
    },
    "heading2": {
        "font": "Helvetica-Bold",
        "size": 20.0,
        "leading": 26.0,
        "color": (0.12, 0.12, 0.12),
    },
    "heading3": {
        "font": "Helvetica-Bold",
        "size": 18.0,
        "leading": 22.0,
        "color": (0.15, 0.15, 0.15),
    },
    "heading4": {
        "font": "Helvetica-Bold",
        "size": 16.0,
        "leading": 20.0,
        "color": (0.2, 0.2, 0.2),
    },
    "heading5": {
        "font": "Helvetica-Bold",
        "size": 14.0,
        "leading": 18.0,
        "color": (0.2, 0.2, 0.2),
    },
    "heading6": {
        "font": "Helvetica-Bold",
        "size": 13.0,
        "leading": 16.0,
        "color": (0.2, 0.2, 0.2),
    },
    "code": {
        "font": "Courier",
        "size": 10.0,
        "leading": 14.0,
        "color": (0.2, 0.2, 0.2),
    },
    "inline_code": {
        "font": "Courier",
        "size": 10.0,
        "leading": 15.0,
        "color": (0.3, 0.1, 0.1),
    },
    "list": {
        "font": "Helvetica",
        "size": 11.0,
        "leading": 15.0,
        "color": (0, 0, 0),
    },
}


def layout_blocks(blocks: List[Block]) -> Tuple[List[Page], Dict[str, Tuple[int, float]]]:
    pages: List[Page] = [Page()]
    current_y = PAGE_HEIGHT - TOP_MARGIN
    anchor_positions: Dict[str, Tuple[int, float]] = {}

    def new_page() -> None:
        nonlocal current_y
        pages.append(Page())
        current_y = PAGE_HEIGHT - TOP_MARGIN

    def ensure_space(amount: float) -> None:
        if current_y - amount < BOTTOM_MARGIN:
            new_page()

    def collect_annotations(page: Page, line: Line) -> None:
        current_link: Optional[str] = None
        start_x = end_x = 0.0
        max_size = 0.0
        for segment in line.segments:
            if segment.link != current_link:
                if current_link is not None and end_x > start_x:
                    ascent = max_size * 0.8
                    descent = max_size * 0.2
                    page.annotations.append(
                        Annotation(
                            rect=(start_x, line.y - descent, end_x, line.y + ascent),
                            target=current_link,
                        )
                    )
                if segment.link is not None:
                    current_link = segment.link
                    start_x = segment.x
                    end_x = segment.x + segment.width
                    max_size = segment.size
                else:
                    current_link = None
                    start_x = end_x = 0.0
                    max_size = 0.0
            else:
                end_x = max(end_x, segment.x + segment.width)
                max_size = max(max_size, segment.size)
        if current_link is not None and end_x > start_x:
            ascent = max_size * 0.8
            descent = max_size * 0.2
            page.annotations.append(
                Annotation(
                    rect=(start_x, line.y - descent, end_x, line.y + ascent),
                    target=current_link,
                )
            )

    def add_line(line: Line) -> None:
        page = pages[-1]
        page.lines.append(line)
        collect_annotations(page, line)

    for block in blocks:
        if block.type == "heading":
            style_key = f"heading{min(block.level, 6)}"
            style = STYLE_DEFS[style_key]
            ensure_space(style["leading"] + style["leading"] / 2)
            current_y -= style["leading"] / 2
            wrapped = wrap_inline_segments(
                block.inline,
                PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN,
                style["font"],
                style["size"],
            )
            for line_segments in wrapped:
                ensure_space(style["leading"])
                current_y -= style["leading"]
                positioned, _ = inline_to_positioned(
                    line_segments,
                    style["font"],
                    style["size"],
                    style["color"],
                    indent=0,
                )
                line = Line(y=current_y, leading=style["leading"], segments=positioned)
                if block.anchor and block.anchor not in anchor_positions:
                    anchor_positions[block.anchor] = (len(pages) - 1, current_y)
                    line.anchor = block.anchor
                add_line(line)
            current_y -= style["leading"] / 2
            continue

        if block.type == "paragraph":
            style = STYLE_DEFS["body"]
            available = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
            wrapped = wrap_inline_segments(
                block.inline,
                available,
                style["font"],
                style["size"],
            )
            for line_segments in wrapped:
                ensure_space(style["leading"])
                current_y -= style["leading"]
                positioned_segments: List[PositionedSegment] = []
                line_width = 0.0
                for segment in line_segments:
                    seg_style = STYLE_DEFS["inline_code"] if segment.kind == "code" else style
                    pos, width = inline_to_positioned(
                        [segment],
                        seg_style["font"],
                        seg_style["size"],
                        seg_style["color"],
                        indent=line_width,
                    )
                    positioned_segments.extend(pos)
                    line_width += width
                add_line(Line(y=current_y, leading=style["leading"], segments=positioned_segments))
            current_y -= style["leading"] / 2
            continue

        if block.type in {"ulist", "olist"}:
            style = STYLE_DEFS["list"]
            bullet_indent = 18
            text_indent = 32
            counter = 1
            for item in block.items:
                raw_marker = "•" if block.type == "ulist" else f"{counter}."
                marker_text = sanitize_text(raw_marker) or "-"
                counter += 1
                ensure_space(style["leading"])
                current_y -= style["leading"]
                bullet_seg = PositionedSegment(
                    text=marker_text,
                    font="Helvetica-Bold",
                    size=style["size"],
                    color=style["color"],
                    x=LEFT_MARGIN,
                    width=estimate_text_width(marker_text, "Helvetica-Bold", style["size"]),
                )
                wrapped = wrap_inline_segments(
                    item,
                    PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN - (text_indent - bullet_indent),
                    style["font"],
                    style["size"],
                )
                first_line = True
                for line_segments in wrapped:
                    positioned_segments: List[PositionedSegment] = []
                    if first_line:
                        positioned_segments.append(bullet_seg)
                    line_width = 0.0
                    for segment in line_segments:
                        seg_style = STYLE_DEFS["inline_code"] if segment.kind == "code" else style
                        indent = text_indent if first_line else text_indent
                        pos, width = inline_to_positioned(
                            [segment],
                            seg_style["font"],
                            seg_style["size"],
                            seg_style["color"],
                            indent=indent + line_width,
                        )
                        positioned_segments.extend(pos)
                        line_width += width
                    add_line(Line(y=current_y, leading=style["leading"], segments=positioned_segments))
                    first_line = False
                    ensure_space(style["leading"])
                    current_y -= style["leading"]
                current_y += style["leading"]
            current_y -= style["leading"] / 2
            continue

        if block.type == "code":
            style = STYLE_DEFS["code"]
            available = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
            for raw_line in block.lines or [""]:
                if raw_line:
                    wrapped_lines = textwrap.wrap(raw_line, width=80) or [raw_line]
                else:
                    wrapped_lines = [""]
                for code_line in wrapped_lines:
                    ensure_space(style["leading"])
                    current_y -= style["leading"]
                    segment = InlineSegment(text=code_line, kind="text")
                    pos, _ = inline_to_positioned(
                        [segment],
                        style["font"],
                        style["size"],
                        style["color"],
                        indent=0,
                    )
                    add_line(Line(y=current_y, leading=style["leading"], segments=pos))
            current_y -= style["leading"] / 2
            continue

    return pages, anchor_positions


# ---------------------------------------------------------------------------
# PDF assembly


class PDFBuilder:
    def __init__(self) -> None:
        self.objects: List[Optional[bytes]] = [None]

    def add_object(self, content: Optional[bytes]) -> int:
        self.objects.append(content)
        return len(self.objects) - 1

    def set_object(self, obj_id: int, content: bytes) -> None:
        self.objects[obj_id] = content

    def build(self, output_path: Path) -> None:
        with output_path.open("wb") as handle:
            handle.write(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
            offsets: List[int] = []
            for index, obj in enumerate(self.objects[1:], start=1):
                if obj is None:
                    raise ValueError(f"Object {index} is undefined")
                offsets.append(handle.tell())
                handle.write(f"{index} 0 obj\n".encode("latin-1"))
                handle.write(obj)
                if not obj.endswith(b"\n"):
                    handle.write(b"\n")
                handle.write(b"endobj\n")
            xref_pos = handle.tell()
            handle.write(f"xref\n0 {len(self.objects)}\n".encode("latin-1"))
            handle.write(b"0000000000 65535 f \n")
            for offset in offsets:
                handle.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
            handle.write(b"trailer\n")
            handle.write(
                f"<< /Size {len(self.objects)} /Root 1 0 R >>\n".encode("latin-1")
            )
            handle.write(b"startxref\n")
            handle.write(f"{xref_pos}\n".encode("latin-1"))
            handle.write(b"%%EOF")


def build_page_stream(lines: List[Line], font_names: Dict[str, str]) -> bytes:
    commands: List[str] = ["BT"]
    current_font = None
    current_size = None
    current_color = None
    for line in lines:
        for segment in line.segments:
            font = font_names.get(segment.font, "F1")
            if current_font != font or current_size != segment.size:
                commands.append(f"/{font} {segment.size:.2f} Tf")
                current_font = font
                current_size = segment.size
            if current_color != segment.color:
                r, g, b = segment.color
                commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
                current_color = segment.color
            commands.append(f"1 0 0 1 {segment.x:.2f} {line.y:.2f} Tm")
            commands.append(f"({pdf_escape(segment.text)}) Tj")
    commands.append("ET")
    stream_data = "\n".join(commands).encode("latin-1")
    return (
        f"<< /Length {len(stream_data)} >>\n".encode("latin-1")
        + b"stream\n"
        + stream_data
        + b"\nendstream"
    )


def assemble_pdf(
    pages: List[Page],
    anchor_positions: Dict[str, Tuple[int, float]],
    output: Path,
) -> None:
    pdf = PDFBuilder()
    catalog_id = pdf.add_object(None)
    pages_id = pdf.add_object(None)

    font_map = {
        "Helvetica": pdf.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        "Helvetica-Bold": pdf.add_object(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
        ),
        "Courier": pdf.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"),
    }

    font_names = {"Helvetica": "F1", "Helvetica-Bold": "F2", "Courier": "F3"}

    page_ids: List[int] = []
    stream_ids: List[int] = []
    for page in pages:
        stream_id = pdf.add_object(build_page_stream(page.lines, font_names))
        stream_ids.append(stream_id)
        page_id = pdf.add_object(None)
        page_ids.append(page_id)

    for idx, page in enumerate(pages):
        resources = (
            "<< /Font << /F1 {f1} 0 R /F2 {f2} 0 R /F3 {f3} 0 R >> >>".format(
                f1=font_map["Helvetica"], f2=font_map["Helvetica-Bold"], f3=font_map["Courier"]
            )
        )

        annot_ids: List[int] = []
        for annotation in page.annotations:
            x1, y1, x2, y2 = annotation.rect
            if annotation.target.startswith("internal:"):
                anchor = annotation.target.split(":", 1)[1]
                if anchor not in anchor_positions:
                    continue
                page_index, dest_y = anchor_positions[anchor]
                dest_page_id = page_ids[page_index]
                dest_array = (
                    f"[{dest_page_id} 0 R /XYZ {LEFT_MARGIN:.2f} {dest_y:.2f} null]"
                )
                annot_dict = (
                    "<< /Type /Annot /Subtype /Link /Rect [{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}] "
                    "/Border [0 0 0] /A << /S /GoTo /D {dest} >> >>"
                ).format(dest=dest_array, x1=x1, y1=y1, x2=x2, y2=y2)
                annot_ids.append(pdf.add_object(annot_dict.encode("latin-1")))
            elif annotation.target.startswith("external:"):
                uri = pdf_escape(annotation.target.split(":", 1)[1])
                annot_dict = (
                    "<< /Type /Annot /Subtype /Link /Rect [{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}] "
                    "/Border [0 0 0] /A << /S /URI /URI ({uri}) >> >>"
                ).format(uri=uri, x1=x1, y1=y1, x2=x2, y2=y2)
                annot_ids.append(pdf.add_object(annot_dict.encode("latin-1")))

        annots_str = (
            "/Annots [" + " ".join(f"{annot_id} 0 R" for annot_id in annot_ids) + "]"
            if annot_ids
            else ""
        )

        page_dict = (
            "<< /Type /Page /Parent {parent} 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources {resources} /Contents {contents} 0 R {annots} >>"
        ).format(
            parent=pages_id,
            width=PAGE_WIDTH,
            height=PAGE_HEIGHT,
            resources=resources,
            contents=stream_ids[idx],
            annots=annots_str,
        )
        pdf.set_object(page_ids[idx], page_dict.encode("latin-1"))

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_dict = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>"
    pdf.set_object(pages_id, pages_dict.encode("latin-1"))
    catalog_dict = f"<< /Type /Catalog /Pages {pages_id} 0 R >>"
    pdf.set_object(catalog_id, catalog_dict.encode("latin-1"))

    pdf.build(output)


# ---------------------------------------------------------------------------
# High level orchestration


@dataclass
class MarkdownDocument:
    path: Path
    blocks: List[Block]


def resolve_link_factory(current_path: Path, anchors: Dict[str, str], root: Path):
    root_abs = root.resolve()
    current_abs = (root_abs / current_path).resolve(strict=False)

    def resolver(target: str) -> Optional[str]:
        if not target:
            return None
        parsed = urlparse(target)
        if parsed.scheme or parsed.netloc:
            return f"external:{target}"

        if target.startswith("#"):
            slug = slugify(target[1:]) if target[1:] else "top"
            anchor_key = f"{current_path.as_posix()}#{slug}"
            anchor_value = anchors.get(anchor_key, anchor_key)
            return f"internal:{anchor_value}"

        file_part, anchor_part = (target, "")
        if "#" in target:
            file_part, anchor_part = target.split("#", 1)

        if not file_part:
            relative_path = current_path
        else:
            resolved = (current_abs.parent / file_part).resolve(strict=False)
            try:
                relative_path = resolved.relative_to(root_abs)
            except ValueError:
                relative_path = Path(file_part)

        slug = slugify(anchor_part) if anchor_part else "top"
        anchor_key = f"{relative_path.as_posix()}#{slug}"
        anchor_value = anchors.get(anchor_key, anchor_key)
        return f"internal:{anchor_value}"

    return resolver


def gather_markdown_documents(root: Path) -> List[MarkdownDocument]:
    md_files = sorted(path for path in root.rglob("*.md") if path.is_file())

    anchors: Dict[str, str] = {}
    file_texts: List[Tuple[Path, str]] = []
    for md_path in md_files:
        relative = md_path.relative_to(root)
        anchor_prefix = relative.as_posix()
        anchors[f"{anchor_prefix}#top"] = f"{anchor_prefix}#top"

        text = md_path.read_text(encoding="utf-8")
        file_texts.append((relative, text))
        blocks = parse_markdown(text, anchor_prefix=anchor_prefix, resolve_link=lambda _: None)
        for block in blocks:
            if block.type == "heading" and block.anchor:
                anchors[f"{anchor_prefix}#{slugify(block.text)}"] = block.anchor

    documents: List[MarkdownDocument] = []
    root_abs = root.resolve()
    for relative, text in file_texts:
        resolver = resolve_link_factory(relative, anchors, root_abs)
        blocks = parse_markdown(text, anchor_prefix=relative.as_posix(), resolve_link=resolver)
        file_heading = Block(
            type="heading",
            level=1,
            text=relative.as_posix(),
            inline=parse_inline(relative.as_posix(), resolver),
            anchor=f"{relative.as_posix()}#top",
        )
        documents.append(MarkdownDocument(path=relative, blocks=[file_heading] + blocks))

    return documents


def create_documentation_pdf(output: Path, root: Path) -> None:
    documents = gather_markdown_documents(root)
    all_blocks: List[Block] = []
    for doc in documents:
        all_blocks.extend(doc.blocks)

    pages, anchor_positions = layout_blocks(all_blocks)
    assemble_pdf(pages, anchor_positions, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ovos-documentation.pdf"),
        help="Destination PDF file (default: ovos-documentation.pdf)",
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
