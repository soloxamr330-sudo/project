"""Helpers to build extractable PDFs for tests."""

from __future__ import annotations

from pathlib import Path


def write_pdf(path: Path, pages: list[str]) -> Path:
    objects = ["placeholder"]  # 1-based PDF object numbering
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")

    font_id = 3 + len(pages) * 2
    for i, text in enumerate(pages):
        page_id = 3 + i * 2
        content_id = page_id + 1
        escaped = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .replace("\n", " ")[:180]
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects[1:], start=1):
        offsets.append(len(body))
        body.extend(f"{index} 0 obj\n{obj}\nendobj\n".encode("latin-1"))

    xref_pos = len(body)
    body.extend(f"xref\n0 {len(objects)}\n".encode("latin-1"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    body.extend(
        (
            f"trailer\n<< /Size {len(objects)} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("latin-1")
    )
    path.write_bytes(bytes(body))
    return path
