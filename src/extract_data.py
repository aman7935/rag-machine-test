import pdfplumber


def extract_pdf(path):
    text_chunks = []
    table_rows = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = table[0]
                for row in table[1:]:
                    table_rows.append({"page": page_num, "header": header, "row": row})

            page_text = page.extract_text() or ""
            text_chunks.append({"page": page_num, "text": page_text})

    return text_chunks, table_rows
