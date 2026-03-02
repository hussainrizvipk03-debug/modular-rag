import PyPDF2

def load_pdf(path="Natural Language Processing.pdf"):
    reader = PyPDF2.PdfReader(path)
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())


print("book loaded")
    
