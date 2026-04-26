# Book bibliographic extractor (fully local)

Extract basic bibliographic data (title, author, publisher, year, ISBN, etc.) from images of book covers or title pages using **local** OCR and a **local** LLM. Results are appended to a CSV file.

The purpose of this project was to automate the process of cataloging my kids school book library collection.

## Requirements

- **Python 3.8+**
- **Tesseract OCR** installed on your system
- **Ollama** installed and running, with at least one model (e.g. `llama3.2`)

## Setup

### 1. Install Tesseract

- **Windows**: Download installer from [GitHub UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and install. Add the install directory (e.g. `C:\Program Files\Tesseract-OCR`) to your PATH, or set `TESSDATA_PREFIX` if needed.
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr` (or your distro’s package).

### 2. Install Ollama and pull a model

- Download from [ollama.com](https://ollama.com) and install.
- Start Ollama (it runs in the background).
- Pull a model:  
  `ollama pull llama3.2`  
  (Smaller/faster: `ollama pull phi3` or `ollama pull tinyllama`.)

### 3. Python environment

```bash
cd book-bib-extractor
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

If Tesseract is not on your PATH, set it before running:

```bash
# Windows (PowerShell); adjust path if needed
$env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR"
# Or set pytesseract.pytesseract.tesseract_cmd in the script to the full path of tesseract.exe
```

## Usage

**Single image:**

```bash
python extract.py path/to/book_cover.jpg -o bibliography.csv
```

**Folder of images (typical for many covers):**

```bash
python extract.py images/ -o bibliography.csv
```

**Mix files and folders:**

```bash
python extract.py images/ cover_extra.jpg -o bibliography.csv
```

**Subfolders too:**

```bash
python extract.py images/ -r -o bibliography.csv
```

**Options:**

- `-o, --output` – Output CSV path (default: `bibliography.csv`).
- `-r, --recursive` – When a path is a folder, include images in subfolders.
- `--no-preprocess` – Skip image preprocessing (use for already clean scans).
- `--model` – Ollama model name (default: `llama3.2`).

## CSV output

The script creates or appends to a CSV with columns:

`Título`, `Autor`, `Autor secundário`, `Edição`, `Nome da editora`, `Ano de publicação`, `Nome da coleção`, `ISBN`, `N.º depósito legal`, `source_image`

Empty fields are left blank. `source_image` stores the path of the image used.

## Tips

- Use clear, well-lit photos or scans; preprocessing helps with photos.
- For best OCR, crop to the cover or title page only.
- If the LLM returns invalid JSON, try a different model (`--model phi3`) or a slightly larger one.
