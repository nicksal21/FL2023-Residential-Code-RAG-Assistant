"""
Optional local utility: convert your own PDFs to markdown for ingestion.

This script is not used by the Flask app at runtime. The public demo ships
with original sample markdown in sample_docs/ to avoid redistributing
copyrighted building code PDFs.

Usage:
  1. Place PDFs in ./local_pdfs/ (gitignored)
  2. Run: python doc_parser.py
  3. Review output in ./local_mds/ before adding to your private vector store
"""

from docling.document_converter import DocumentConverter
from pathlib import Path
from os import listdir
from os.path import isfile, join

pdf_directory = Path("./local_pdfs")
md_directory = Path("./local_mds")

md_directory.mkdir(exist_ok=True)

if not pdf_directory.is_dir():
    raise FileNotFoundError(
        "Create a local_pdfs/ directory and add your own licensed PDFs first."
    )

files = sorted(f for f in listdir(pdf_directory) if isfile(join(pdf_directory, f)))

for file in files:
    if not file.lower().endswith(".pdf"):
        continue
    pdf_file_path = pdf_directory / file
    md_file_path = md_directory / file.replace(".pdf", ".md")
    converter = DocumentConverter()
    result = converter.convert(str(pdf_file_path))
    md_result = result.document.export_to_markdown()
    md_file_path.write_text(md_result, encoding="utf-8")
    print(f"Wrote {md_file_path}")
