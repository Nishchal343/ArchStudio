import io
import re
import textwrap


def _minimum_pages(layout: dict) -> int:
    configured = layout.get("minimum_pages")
    if configured:
        return max(1, int(configured))
    target = str(layout.get("page_target", ""))
    match = re.search(r"\d+", target)
    return max(1, int(match.group())) if match else 1


def _section_break_indices(section_count: int, minimum_pages: int) -> set[int]:
    """Return no artificial breaks; document flow determines pagination."""
    # A minimum page target must not be achieved by inserting blank space.
    # ReportLab and Word will naturally continue content onto the next page
    # when the current page is full.
    return set()

from ...schemas import EditableDocument


def export_document(document: EditableDocument, output_format: str) -> tuple[bytes, str, str]:
    if output_format == "docx":
        from docx import Document
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        stream = io.BytesIO()
        doc = Document()
        columns = int(document.layout.get("columns", 1) or 1)
        if columns > 1:
            section = doc.sections[0]
            cols = section._sectPr.xpath("./w:cols")
            element = cols[0] if cols else OxmlElement("w:cols")
            if not cols:
                section._sectPr.append(element)
            element.set(qn("w:num"), str(columns))
        title = doc.add_heading(document.title, level=0)
        title.alignment = 1
        for index in range(0, len(document.authors), 3):
            row = document.authors[index:index + 3]
            paragraph = doc.add_paragraph()
            paragraph.alignment = 1
            paragraph.add_run("    ".join(f"{author.get('name', '[AUTHOR]')}\n{author.get('institution', '[INSTITUTION / AFFILIATION]')}" for author in row))
        date = doc.add_paragraph(document.date)
        date.alignment = 1
        breaks = _section_break_indices(len(document.sections), _minimum_pages(document.layout))
        for index, section in enumerate(document.sections):
            if index in breaks:
                doc.add_page_break()
            doc.add_heading(section.heading, level=1)
            for paragraph in section.content.split("\n"):
                doc.add_paragraph(paragraph)
        doc.save(stream)
        return stream.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"
    if output_format == "pdf":
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        stream = io.BytesIO()
        width, height = letter
        columns = max(1, int(document.layout.get("columns", 1) or 1))
        gutter = 0.22 * inch
        frame_width = (width - 1.0 * inch - gutter * (columns - 1)) / columns
        frames = [Frame(0.5 * inch + i * (frame_width + gutter), 0.5 * inch,
                        frame_width, height - inch, id=f"col{i}", leftPadding=5, rightPadding=5,
                        topPadding=5, bottomPadding=5) for i in range(columns)]
        pdf = BaseDocTemplate(stream, pagesize=letter,
                              leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                              topMargin=0.5 * inch, bottomMargin=0.5 * inch)
        pdf.addPageTemplates([PageTemplate(id="report", frames=frames)])
        styles = getSampleStyleSheet()
        story = [Paragraph(document.title, styles["Title"])]
        for index in range(0, len(document.authors), 3):
            row = document.authors[index:index + 3]
            author_text = " &nbsp;&nbsp;&nbsp; ".join(f"{a.get('name', '[AUTHOR]')}<br/>{a.get('institution', '[INSTITUTION / AFFILIATION]')}" for a in row)
            story.append(Paragraph(author_text, styles["BodyText"]))
        story.append(Paragraph(document.date, styles["BodyText"]))
        breaks = _section_break_indices(len(document.sections), _minimum_pages(document.layout))
        for index, section in enumerate(document.sections):
            if index in breaks:
                story.append(PageBreak())
            story.extend([Spacer(1, 8), Paragraph(section.heading, styles["Heading2"])])
            story.extend(Paragraph(paragraph, styles["BodyText"]) for paragraph in section.content.split("\n") if paragraph.strip())
        pdf.build(story)
        return stream.getvalue(), "application/pdf", "pdf"
    raise ValueError("Unsupported document format")
