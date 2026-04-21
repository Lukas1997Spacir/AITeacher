import fitz
import docx


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]

        chunks.append({
            "text": chunk,
            "chunk_id": chunk_id
        })

        chunk_id += 1
        start += chunk_size - overlap

    return chunks


# PDF
def extract_text_from_pdf(uploaded_file):
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


# TXT
def extract_text_from_txt(uploaded_file):
    uploaded_file.seek(0)
    return uploaded_file.read().decode("utf-8")


# DOCX
def extract_text_from_docx(uploaded_file):
    uploaded_file.seek(0)
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text(uploaded_file):
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)

    elif file_type == "text/plain":
        return extract_text_from_txt(uploaded_file)

    elif file_type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        return extract_text_from_docx(uploaded_file)

    else:
        raise ValueError(f"Nepodporovaný formát: {file_type}")
