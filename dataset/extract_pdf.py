import os
import pdfplumber

folder_input = "raw/pdf"
folder_outpout = "processed/extracted_texts"

os.makedirs(folder_outpout, exist_ok=True)

for filename in os.listdir(folder_input):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_input, filename)
        text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()

                    if extracted:
                        text += extracted + "\n"
            
            output_name = filename.replace(".pdf", ".txt")
            output_path = os.path.join(folder_outpout, output_name)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"[OK] {filename}")

        except Exception as e:
            print(f"[ERROR] {filename} -> {e}")