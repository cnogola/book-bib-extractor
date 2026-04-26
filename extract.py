"""
Extract bibliographic data from book images using local OCR (Tesseract) and
local LLM (Ollama). Results are appended to a CSV file.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import pytesseract
from PIL import Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False

# If Tesseract is not on PATH (common on Windows), set its path here or via env TESSERACT_CMD
if os.name == "nt":
    _tesseract_cmd = os.environ.get("TESSERACT_CMD", r"C:\Users\goncalo.lourenco\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
    if os.path.isfile(_tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


# CSV columns (Portuguese bibliographic fields)
BIB_FIELDS = [
    "Título",
    "Autor",
    "Autor secundário",
    "Edição",
    "Nome da editora",
    "Ano de publicação",
    "Nome da coleção",
    "ISBN",
    "N.º depósito legal",
    "source_image",
]

# LLM JSON keys -> CSV column names
_LLM_TO_CSV = {
    "titulo": "Título",
    "autor": "Autor",
    "autor_secundario": "Autor secundário",
    "edicao": "Edição",
    "nome_editora": "Nome da editora",
    "ano_publicacao": "Ano de publicação",
    "nome_colecao": "Nome da coleção",
    "isbn": "ISBN",
    "numero_deposito_legal": "N.º depósito legal",
}


def preprocess_image(image_path: Path) -> Image.Image:
    """Load and optionally preprocess image to improve OCR."""
    img = Image.open(image_path).convert("RGB")
    if not HAS_OPENCV:
        return img

    import numpy as np
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Light denoising and threshold to get cleaner text
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def ocr_image(image_path: Path, preprocess: bool = True) -> str:
    """Run Tesseract OCR on the image and return extracted text."""
    if preprocess and HAS_OPENCV:
        img = preprocess_image(image_path)
    else:
        img = Image.open(image_path)
    return pytesseract.image_to_string(img).strip()


EXTRACTION_PROMPT = """You are a librarian. Extract bibliographic fields from the following text, which was read from a book cover or title page via OCR. The text may be in Portuguese and may have recognition errors.

Return ONLY a single JSON object with exactly these keys (use empty string "" if not found):
- titulo
- autor
- autor_secundario
- edicao
- nome_editora
- ano_publicacao
- nome_colecao
- isbn
- numero_deposito_legal

Do not include any other text, explanation, or markdown—only the raw JSON object."""


def extract_bib_with_ollama(ocr_text: str, model: str = "llama3.2") -> dict:
    """Use local Ollama model to structure OCR text into bibliographic fields."""
    if not HAS_OLLAMA:
        raise RuntimeError("ollama package not installed. Install with: pip install ollama")

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": EXTRACTION_PROMPT + "\n\n---\n\n" + ocr_text},
        ],
    )
    content = (response.message.content or "").strip()
    # Strip markdown code block if present
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)
    # Try parsing; if it fails, find first { and balance braces to extract JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        if start == -1:
            raise
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(content[start : i + 1])
        raise


def ensure_csv_header(csv_path: Path, fieldnames: list[str]) -> None:
    """Create CSV with header if the file does not exist or is empty."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_bib_to_csv(csv_path: Path, row: dict, fieldnames: list[str]) -> None:
    """Append one bibliographic record to the CSV."""
    ensure_csv_header(csv_path, fieldnames)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def incomplete_csv_path(main_csv: Path) -> Path:
    """Same columns as main output; used when Título or Autor is missing."""
    return main_csv.parent / f"{main_csv.stem}_incomplete{main_csv.suffix}"


def row_has_titulo_and_autor(row: dict) -> bool:
    t = (row.get("Título") or "").strip()
    a = (row.get("Autor") or "").strip()
    return bool(t and a)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".heic",
    ".heif",
}


def collect_image_paths(paths: list[Path], recursive: bool = False) -> list[Path]:
    """Expand paths: files are kept if they look like images; directories yield all images inside."""
    out: list[Path] = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                out.append(p.resolve())
            continue
        if p.is_dir():
            if recursive:
                for f in sorted(p.rglob("*")):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        out.append(f.resolve())
            else:
                for f in sorted(p.iterdir()):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        out.append(f.resolve())
    # Stable unique order (same path from two args)
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in out:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def process_one(
    image_path: Path,
    csv_path: Path,
    incomplete_path: Path,
    preprocess: bool,
    model: str,
) -> str | None:
    """Process a single image. Returns 'main', 'incomplete', or None."""
    print(f"\n--- {image_path} ---")
    ocr_text = ocr_image(image_path, preprocess=preprocess)
    if not ocr_text:
        print("  No text detected.", file=sys.stderr)
        return None
    print(f"  OCR: {len(ocr_text)} chars")
    try:
        bib = extract_bib_with_ollama(ocr_text, model=model)
    except Exception as e:
        print(f"  LLM failed: {e}", file=sys.stderr)
        return None
    row = {}
    for llm_key, csv_col in _LLM_TO_CSV.items():
        row[csv_col] = bib.get(llm_key, "") or ""
    row["source_image"] = str(image_path)
    if row_has_titulo_and_autor(row):
        append_bib_to_csv(csv_path, row, BIB_FIELDS)
        print(f"  Appended to {csv_path}")
        return "main"
    append_bib_to_csv(incomplete_path, row, BIB_FIELDS)
    print(f"  Missing Título or Autor — appended to {incomplete_path}", file=sys.stderr)
    return "incomplete"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract bibliographic data from book images (local OCR + local LLM) and save to CSV."
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        metavar="PATH",
        help="Image file(s) and/or folder(s) containing images (e.g. images/)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="When PATH is a folder, include images in subfolders",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("bibliography.csv"),
        help="Output CSV path (default: bibliography.csv)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip image preprocessing (use if image is already clean)",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model name (default: llama3.2)",
    )
    args = parser.parse_args()

    missing = [p for p in args.paths if not p.exists()]
    for p in missing:
        print(f"Skip (not found): {p}", file=sys.stderr)

    image_paths = collect_image_paths([p for p in args.paths if p.exists()], recursive=args.recursive)
    if not image_paths:
        print("No image files found. Pass image files or a folder of images.", file=sys.stderr)
        sys.exit(1)

    heif_ext = {".heic", ".heif"}
    if not HAS_HEIF and any(p.suffix.lower() in heif_ext for p in image_paths):
        print(
            "HEIC/HEIF images require pillow-heif. Install with: pip install pillow-heif",
            file=sys.stderr,
        )
        sys.exit(1)

    preprocess = not args.no_preprocess
    inc_path = incomplete_csv_path(args.output)
    main_n = 0
    incomplete_n = 0
    for image_path in image_paths:
        r = process_one(image_path, args.output, inc_path, preprocess, args.model)
        if r == "main":
            main_n += 1
        elif r == "incomplete":
            incomplete_n += 1
    print(
        f"\nDone: {main_n} → {args.output}, {incomplete_n} → {inc_path} "
        f"({main_n + incomplete_n}/{len(image_paths)} with LLM output)"
    )
    if main_n == 0 and incomplete_n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
