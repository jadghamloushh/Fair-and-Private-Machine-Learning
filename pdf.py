from PyPDF2 import PdfReader, PdfWriter

def extract_pages(input_pdf, output_pdf, start_page, end_page):
    """
    Extract pages from a PDF file and save them to a new PDF.
    
    Args:
        input_pdf (str): Path to the input PDF file
        output_pdf (str): Path to save the output PDF file
        start_page (int): Starting page number (1-based indexing as seen in PDF viewers)
        end_page (int): Ending page number (inclusive)
    """
    # Adjust for 0-based indexing in PyPDF2
    start_idx = start_page - 1
    end_idx = end_page - 1
    
    # Create reader and writer objects
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    
    # Check if page range is valid
    if start_idx < 0 or end_idx >= len(reader.pages):
        raise ValueError(f"Invalid page range. PDF has {len(reader.pages)} pages.")
    
    # Add selected pages to the writer
    for page_num in range(start_idx, end_idx + 1):
        writer.add_page(reader.pages[page_num])
    
    # Write the extracted pages to the output file
    with open(output_pdf, "wb") as output_file:
        writer.write(output_file)
    
    print(f"Successfully extracted pages {start_page}-{end_page} to {output_pdf}")

# Example usage
if __name__ == "__main__":
    extract_pages("Joel+Feinberg++Russ+Shafer-Landau+-+Reason+and+Responsibility_+Readings+in+Some+Basic+Problems+of+Philosophy+(0).pdf", "extracted_pages2.pdf", 169, 178)