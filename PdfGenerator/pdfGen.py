from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from io import BytesIO

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.enums import TA_CENTER


def create_notes_pdf(notes):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading1"]
    )

    subheading_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"]
    )

    body_style = styles["BodyText"]

    elements = []

    elements.append(
        Paragraph(
            "YouTube AI Generated Notes",
            title_style
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    bullet_buffer = []

    lines = notes.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith("# "):

            if bullet_buffer:

                elements.append(
                    ListFlowable(
                        bullet_buffer,
                        bulletType="bullet"
                    )
                )

                bullet_buffer = []

            elements.append(
                Paragraph(
                    line[2:],
                    heading_style
                )
            )

            elements.append(
                Spacer(1, 10)
            )

        elif line.startswith("## "):

            if bullet_buffer:

                elements.append(
                    ListFlowable(
                        bullet_buffer,
                        bulletType="bullet"
                    )
                )

                bullet_buffer = []

            elements.append(
                Paragraph(
                    line[3:],
                    subheading_style
                )
            )

            elements.append(
                Spacer(1, 6)
            )

        elif line.startswith("- "):

            bullet_buffer.append(
                ListItem(
                    Paragraph(
                        line[2:],
                        body_style
                    )
                )
            )

        else:

            if bullet_buffer:

                elements.append(
                    ListFlowable(
                        bullet_buffer,
                        bulletType="bullet"
                    )
                )

                bullet_buffer = []

            elements.append(
                Paragraph(
                    line,
                    body_style
                )
            )

            elements.append(
                Spacer(1, 5)
            )

    if bullet_buffer:

        elements.append(
            ListFlowable(
                bullet_buffer,
                bulletType="bullet"
            )
        )

    doc.build(elements)

    buffer.seek(0)

    return buffer