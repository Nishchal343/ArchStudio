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
    """Legacy compatibility helper retained for existing callers/tests."""
    return set()


def _target_pages(layout: dict) -> int:
    configured = layout.get("page_count")
    if configured:
        return max(1, int(configured))
    return _minimum_pages(layout)


def _page_groups(sections: list, target_pages: int) -> list[list]:
    """Distribute report sections across explicit output pages.

    The old exporter left pagination entirely to the flowable engine, so a
    short model response could collapse a ten-page report into two pages. A
    section-balanced page plan keeps every target page populated while still
    allowing normal paragraph flow inside each page/column.
    """
    groups = [[] for _ in range(target_pages)]
    if not sections:
        return groups
    total_words = sum(len(str(section.content or "").split()) for section in sections)
    page_budget = max(1, (total_words + target_pages - 1) // target_pages)
    page = 0
    words_on_page = 0
    for section in sections:
        words = str(section.content or "").split()
        if not words:
            groups[page].append(section)
            continue
        first_chunk = True
        while words:
            available = page_budget - words_on_page if page < target_pages - 1 else len(words)
            chunk_size = max(1, min(len(words), available))
            chunk = " ".join(words[:chunk_size])
            words = words[chunk_size:]
            heading = section.heading if first_chunk else f"{section.heading} (continued)"
            groups[page].append(section.model_copy(update={"heading": heading, "content": chunk}))
            first_chunk = False
            words_on_page += chunk_size
            if words and page < target_pages - 1:
                page += 1
                words_on_page = 0
    # A short fallback document still needs every requested page populated by
    # a section. Move the last available sections into empty trailing pages.
    for page_index in range(target_pages):
        if groups[page_index]:
            continue
        donor = next((index for index in range(target_pages - 1, page_index, -1) if len(groups[index]) > 1), None)
        if donor is not None:
            groups[page_index].append(groups[donor].pop())
    return groups

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
        groups = _page_groups(document.sections, _target_pages(document.layout))
        for page_index, group in enumerate(groups):
            if page_index:
                doc.add_page_break()
            for section in group:
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
        groups = _page_groups(document.sections, _target_pages(document.layout))
        for page_index, group in enumerate(groups):
            if page_index:
                story.append(PageBreak())
            for section in group:
                story.extend([Spacer(1, 8), Paragraph(section.heading, styles["Heading2"])])
                story.extend(Paragraph(paragraph, styles["BodyText"]) for paragraph in section.content.split("\n") if paragraph.strip())
        pdf.build(story)
        return stream.getvalue(), "application/pdf", "pdf"
    raise ValueError("Unsupported document format")
