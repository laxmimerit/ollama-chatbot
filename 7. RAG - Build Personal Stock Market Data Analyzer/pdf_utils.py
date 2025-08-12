import pymupdf
from io import BytesIO

def get_pdf_text(file_path):
      doc = pymupdf.open(file_path)
      texts = []
      for page in doc:
          temp = page.get_text()
          texts.append(temp.strip())
      doc.close()
      return texts