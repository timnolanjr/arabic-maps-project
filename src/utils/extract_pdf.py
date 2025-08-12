# src/utils/extract_pdf.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Union
import fitz  # PyMuPDF


def extract_images_from_pdf(
    pdf_path: Union[str, Path],
    out_dir: Union[str, Path, None] = None,
    basename: Optional[str] = None,
    *,
    dedupe: bool = True,
    include_smask: bool = False,
    fallback_to_pixmap: bool = True,
) -> List[Path]:
    """
    Extract embedded images from a single PDF, preserving the original encoding
    and file extension whenever possible.

    Strategy:
      1) Use doc.extract_image(xref) to get raw bytes + original "ext".
      2) If unavailable (rare or inline/unsupported), optionally fall back to
         rendering that XObject as a Pixmap and saving PNG.

    Args:
        pdf_path: Path to the .pdf file.
        out_dir:  Output directory. Defaults to the PDF's parent directory.
        basename: Base name for outputs. Defaults to the PDF stem.
        dedupe:   If True, skip duplicate xrefs (same image reused across pages).
        include_smask: If True, also extract soft-mask (alpha) images if present.
        fallback_to_pixmap: If True, save a PNG via Pixmap when raw bytes are not available.

    Returns:
        A list of Path objects pointing to the saved image files, in encounter order.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir) if out_dir is not None else pdf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = basename or pdf_path.stem

    doc = fitz.open(pdf_path)
    written: List[Path] = []
    seen_xrefs: set[int] = set()
    counter = 1

    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)

            # full=True includes metadata like smask xref
            for img in page.get_images(full=True):
                xref = img[0]
                smask_xref = img[1] if len(img) > 1 else 0

                if xref == 0:
                    # Inline images generally cannot be extracted via extract_image.
                    # No reliable path to "original" bytes; skip or rely on Pixmap (but needs geometry).
                    continue

                if dedupe and xref in seen_xrefs:
                    continue

                # Try to extract original bytes + extension.
                info: Optional[Dict] = None
                try:
                    info = doc.extract_image(xref)
                except Exception:
                    info = None

                if info and "image" in info:
                    ext = (info.get("ext") or "").lower()
                    if not ext:
                        # Unknown; default to binary blob with .bin
                        ext = "bin"
                    data: bytes = info["image"]
                    out_path = out_dir / f"{stem}_{counter}.{ext}"
                    with open(out_path, "wb") as f:
                        f.write(data)
                    written.append(out_path)
                    seen_xrefs.add(xref)
                    counter += 1

                    # Optionally write the soft mask as a separate image
                    if include_smask and smask_xref and smask_xref not in seen_xrefs:
                        try:
                            s_info = doc.extract_image(smask_xref)
                            if s_info and "image" in s_info:
                                s_ext = (s_info.get("ext") or "").lower() or "bin"
                                s_out = out_dir / f"{stem}_{counter}.{s_ext}"
                                with open(s_out, "wb") as f:
                                    f.write(s_info["image"])
                                written.append(s_out)
                                seen_xrefs.add(smask_xref)
                                counter += 1
                        except Exception:
                            # Soft mask extraction failed; skip silently
                            pass

                    continue

                # Fallback: render to PNG (colorspace-corrected) if requested
                if fallback_to_pixmap:
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        try:
                            # Convert CMYK/alpha to RGB for PNG compatibility
                            if pix.n >= 5:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            out_path = out_dir / f"{stem}_{counter}.png"
                            pix.save(out_path)
                            written.append(out_path)
                            seen_xrefs.add(xref)
                            counter += 1
                        finally:
                            pix = None
                    except Exception:
                        # Could not render; skip this image
                        pass

    finally:
        doc.close()

    return written


def extract_images_from_folder(
    folder: Union[str, Path],
    out_base: Union[str, Path, None] = None,
    *,
    dedupe: bool = True,
    include_smask: bool = False,
    fallback_to_pixmap: bool = True,
) -> dict[Path, List[Path]]:
    """
    Extract images (original encoding if possible) from all PDFs in a folder.

    If out_base is provided, each PDF's images go to <out_base>/<pdf_stem>/.
    Otherwise they are written next to the source PDF.

    Returns:
        Dict mapping PDF path -> list of extracted image paths.
    """
    folder = Path(folder)
    results: dict[Path, List[Path]] = {}

    for pdf in sorted(folder.glob("*.pdf")):
        target_dir = (Path(out_base) / pdf.stem) if out_base else pdf.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        results[pdf] = extract_images_from_pdf(
            pdf,
            out_dir=target_dir,
            basename=pdf.stem,
            dedupe=dedupe,
            include_smask=include_smask,
            fallback_to_pixmap=fallback_to_pixmap,
        )
    return results