from docling.document_converter import DocumentConverter
from pathlib import Path
from os import listdir
from os.path import isfile, join

pdf_directory = Path("./florida_residential_code_pdfs")
md_directory = Path("./florida_residential_code_mds")
# Source - https://stackoverflow.com/a/3207973
# Posted by pycruft, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-11, License - CC BY-SA 4.0
files = [f for f in listdir(pdf_directory) if isfile(join(pdf_directory, f))]
files.sort()

for file in files:
    pdf_file_path = str(pdf_directory) + "/" + file
    md_file_path = str(md_directory) + "/" + file.replace(".pdf", ".md")
    converter = DocumentConverter()
    result = converter.convert(pdf_file_path)
    md_result = result.document.export_to_markdown()
    with open(md_file_path, "w") as f:
        f.write(md_result)


