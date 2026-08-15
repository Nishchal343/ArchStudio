import io
import textwrap

from ...schemas import EditableDocument


def export_document(document: EditableDocument, output_format: str) -> tuple[bytes, str, str]:
    if output_format == "docx":
        from docx import Document
        stream = io.BytesIO()
        doc = Document()
        doc.add_heading(document.title, level=0)
        for section in document.sections:
            doc.add_heading(section.heading, level=1)
            for paragraph in section.content.split("\n"):
                doc.add_paragraph(paragraph)
        doc.save(stream)
        return stream.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"
    if output_format == "pdf":
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        stream = io.BytesIO()
        pdf = canvas.Canvas(stream, pagesize=letter)
        width, height = letter
        y = height - 50

        def write(text: str, size: int = 10, leading: int = 14) -> None:
            nonlocal y
            pdf.setFont("Helvetica", size)
            for line in textwrap.wrap(text, width=100) or [""]:
                if y < 45:
                    pdf.showPage()
                    y = height - 50
                    pdf.setFont("Helvetica", size)
                pdf.drawString(45, y, line)
                y -= leading

        write(document.title, 16, 22)
        for section in document.sections:
            y -= 8
            write(section.heading, 12, 16)
            for paragraph in section.content.split("\n"):
                write(paragraph)
        pdf.save()
        return stream.getvalue(), "application/pdf", "pdf"
    raise ValueError("Unsupported document format")
