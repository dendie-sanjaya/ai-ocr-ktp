from flask import Flask, request, jsonify
import os
import tempfile
import cv2 # OpenCV for image preprocessing
import pytesseract
from PIL import Image # Pillow for image handling
import re
import json

# --- Tesseract Configuration (IMPORTANT!) ---
# If Tesseract is not in your system's PATH, you need to specify the location of the tesseract.exe executable.
# For WSL/Linux, it's usually '/usr/bin/tesseract'.
# For Windows, it could be something like r'C:\Program Files\Tesseract-OCR\tesseract.exe'.
# Also ensure TESSDATA_PREFIX is set if language files are not found automatically.

# Example for WSL/Linux:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
# os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/' # Adjust to your tessdata location

app = Flask(__name__)

# Dictionary for normalizing commonly misdetected OCR values
NORMALIZATION_MAPS = {
    "agama": {
        "BA": "ISLAM",
        "BUDHA": "BUDHA",
        "HINDU": "HINDU",
        "ISLAM": "ISLAM",
        "KATOLIK": "KATOLIK",
        "KONGHUCU": "KONGHUCU",
        "KRISTEN": "KRISTEN"
    },
    "jenis_kelamin": {
        "LAKI-LAKI": "LAKI-LAKI",
        "LAKH MU": "LAKI-LAKI", # Typo example
        "LAKI": "LAKI-LAKI",
        "PEREMPUAN": "PEREMPUAN",
        "PEREM PUAN": "PEREMPUAN"
    },
    "status_perkawinan": {
        "KAWIN": "KAWIN",
        "YAWN A": "KAWIN", # Typo example
        "BELUM KAWIN": "BELUM KAWIN",
        "CERAI HIDUP": "CERAI HIDUP",
        "CERAI MATI": "CERAI MATI"
    },
    "kewarganegaraan": {
        "WNI": "WNI",
        "WN": "WNI", # Abbreviation or typo
        "WNA": "WNA"
    },
    "pekerjaan": {
        "MENGURUS RUMAH TANGGA": "MENGURUS RUMAH TANGGA",
        "MENGUMUS RUMAH TANGGA": "MENGURUS RUMAH TANGGA", # Typo example
        "KARYAWAN SWASTA": "KARYAWAN SWASTA",
        "KARYAWAN": "KARYAWAN SWASTA",
        "PELAJAR/MAHASISWA": "PELAJAR/MAHASISWA",
        "TNI": "TNI",
        "POLRI": "POLRI",
        "PNS": "PNS",
        "WIRASWASTA": "WIRASWASTA",
        # Add more job variations if needed
    },
    "berlaku_hingga": {
        "SEUMUR HIDUP": "SEUMUR HIDUP",
        "UMUR HIDUP": "SEUMUR HIDUP", # Typo example
        "SF UMUR HIDUP": "SEUMUR HIDUP", # Typo example
        "BARTAU HINGGA": "SEUMUR HIDUP" # Typo example
    }
}

def normalize_value(field, value):
    """Normalizes extracted values based on a dictionary."""
    if value is None:
        return None
    
    # Pre-clean value before normalization lookup
    value = value.upper().strip()
    value = re.sub(r'[^A-Z0-9\s/.,-]', '', value) # Remove common non-alphanumeric characters

    if field in NORMALIZATION_MAPS:
        for key, norm_val in NORMALIZATION_MAPS[field].items():
            if key in value: # Check substring
                return norm_val
    
    return value.strip() # Return stripped and uppercased value if not in map

def extract_ktp_data(image_path, lang='ind'):
    """
    Performs OCR on a KTP image, extracts key data using improved regex,
    and returns it as a dictionary.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found at '{image_path}'"}

    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not load image from '{image_path}'. Ensure it's a valid image format."}

    # --- Image Preprocessing for Better OCR Results ---
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Binarization using Otsu's method
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert back to PIL Image format for Tesseract
    pil_img = Image.fromarray(thresh)

    # Perform OCR
    try:
        # Use PSM 6 (Assume a single uniform block of text) or try other PSMs (e.g., 11 for sparse text)
        raw_text = pytesseract.image_to_string(pil_img, lang=lang, config='--psm 6 --oem 3')
    except pytesseract.TesseractNotFoundError:
        return {"error": "Tesseract OCR engine not found. Ensure it's installed and its path is correct."}
    except Exception as e:
        return {"error": f"An error occurred during OCR: {e}"}

    # --- Data Extraction from Raw Text (Improved Logic) ---
    # Initialize with None for all fields
    extracted_data = {
        "NIK": None,
        "Nama": None,
        "Tempat_Lahir": None,
        "Tanggal_Lahir": None,
        "Jenis_Kelamin": None,
        "Alamat": None,
        "RT_RW": None,
        "Kel_Desa": None,
        "Kecamatan": None,
        "Agama": None,
        "Status_Perkawinan": None,
        "Pekerjaan": None,
        "Kewarganegaraan": None,
        "Berlaku_Hingga": None,
        "raw_ocr_text": raw_text
    }

    # Split text into lines and normalize each line
    lines = [line.strip().upper() for line in raw_text.split('\n') if line.strip()]
    full_text_normalized = " ".join(lines) # Normalized full text for global regex

    # --- NIK ---
    # Search for NIK with more flexible patterns, including common typos
    # Allow non-digit characters between digits, then clean them
    nik_match = re.search(r'NIK\s*[: ]*\s*([0-9OIEZSLGAQ\s]{16,})', full_text_normalized)
    if nik_match:
        nik_raw_candidate = nik_match.group(1)
        # Cleaning NIK: replace common OCR errors
        cleaned_nik = (
            nik_raw_candidate.replace('O', '0').replace('I', '1').replace('L', '1')
            .replace('Z', '2').replace('S', '5').replace('G', '6').replace('Q', '9')
            .replace('A', '4').replace(' ', '') # Remove spaces
        )
        # Take only the first 16 digits
        cleaned_nik = ''.join(filter(str.isdigit, cleaned_nik))[:16]
        if len(cleaned_nik) == 16:
            extracted_data['NIK'] = cleaned_nik

    # --- Iterate per line for extraction ---
    for i, line in enumerate(lines):
        # Nama
        # Search for "NAMA", "NAMA LENGKAP", "TAMA" (common typo) followed by optional colon/space and then capture the value.
        name_label_match = re.search(r'(NAMA(?:\s*LENGKAP)?|TAMA)\s*[: ]*\s*(.*)', line)
        if name_label_match and extracted_data['Nama'] is None:
            name_value = name_label_match.group(2).strip() # Capture everything after the label and optional colon/space
            if name_value:
                # Clean the extracted name value
                extracted_data['Nama'] = re.sub(r'[^A-Z\s\.]', '', name_value).strip()
            continue
        # Fallback: If the label is found but the value is on the next line (e.g., "NAMA:\nJOHN DOE")
        elif (re.search(r'NAMA(?:\s*LENGKAP)?\s*[: ]*$', line) or ("TAMA" in line.strip() and line.strip().endswith(':'))) and extracted_data['Nama'] is None:
            if i + 1 < len(lines):
                name_value = lines[i+1].strip()
                if name_value:
                    extracted_data['Nama'] = re.sub(r'[^A-Z\s\.]', '', name_value).strip()
            continue

        # Place/Date of Birth
        # More flexible for labels and date formats
        ttl_match = re.search(r'(?:TEMPAT/?TGL LAHIR|TEMPAT DAN TGL LAHIR|TEMPAT *TGL LAHIR|TEMPAT/IY AR)\s*[: ]*([A-Z\s]+)[, ]*(\d{2}[-/]\d{2}[-/]\d{4})', line)
        if ttl_match:
            extracted_data['Tempat_Lahir'] = ttl_match.group(1).strip()
            extracted_data['Tanggal_Lahir'] = ttl_match.group(2).replace('/', '-').strip()
            continue
        elif extracted_data['Tempat_Lahir'] is None: # If not detected yet, try separate patterns
             tempat_match = re.search(r'(?:TEMPAT/?TGL LAHIR|TEMPAT LAHIR|TEMPAT/IY AR)\s*[: ]*([A-Z\s]+?)(?:,|$)', line)
             if tempat_match:
                 extracted_data['Tempat_Lahir'] = tempat_match.group(1).strip()
             tanggal_match = re.search(r'(\d{2}[-/]\d{2}[-/]\d{4})', line)
             if tanggal_match:
                 extracted_data['Tanggal_Lahir'] = tanggal_match.group(1).replace('/', '-').strip()


        # Gender
        jk_match = re.search(r'(?:JENIS KELAMIN|JARAN KETAUAN)\s*[: ]*(LAKI-LAKI|PEREMPUAN|LAKI|PEREMPUAN)', line) # 'LAKI'/'PEREMPUAN' without strip
        if jk_match:
            extracted_data['Jenis_Kelamin'] = normalize_value("jenis_kelamin", jk_match.group(1))
            continue

        # --- Address, RT/RW, Kel/Desa, Kecamatan (Multi-line) ---
        if "ALAMAT" in line:
            alamat_lines_buffer = []
            start_collecting = False
            # Collect lines starting from the "ALAMAT" line
            for j in range(i, len(lines)):
                current_sub_line = lines[j]
                if "ALAMAT" in current_sub_line and not start_collecting:
                    start_collecting = True
                    val = current_sub_line.split('ALAMAT', 1)[-1].strip()
                    if val.startswith(':'): val = val[1:].strip()
                    if val: alamat_lines_buffer.append(val)
                elif start_collecting:
                    # Stop if next major KTP field is detected
                    if any(keyword in current_sub_line for keyword in ["AGAMA", "STATUS PERKAWINAN", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU HINGGA", "JENIS KELAMIN", "TEMPAT/TGL LAHIR", "NIK", "NAMA"]):
                        break
                    alamat_lines_buffer.append(current_sub_line)
            
            full_address_block = " ".join(alamat_lines_buffer).strip()
            
            # --- Extract RT/RW from the full address block first ---
            rt_rw_match = re.search(r'(?:RT|R\.T|AT)\s*[: ]*(\d{2,3})\s*(?:RW|R\.W|AW)\s*[: ]*(\d{2,3})', full_address_block, re.IGNORECASE)
            if rt_rw_match:
                extracted_data['RT_RW'] = f"{rt_rw_match.group(1)}/{rt_rw_match.group(2)}"
                # Remove RT/RW part from the address for cleaner alamat field
                full_address_block = re.sub(re.escape(rt_rw_match.group(0)), '', full_address_block, flags=re.IGNORECASE).strip()
            else: # Try simple XX/YYY format anywhere
                simple_rt_rw_match = re.search(r'(\d{2,3}/\d{2,3})', full_address_block)
                if simple_rt_rw_match:
                    extracted_data['RT_RW'] = simple_rt_rw_match.group(1)
                    full_address_block = re.sub(re.escape(simple_rt_rw_match.group(0)), '', full_address_block, flags=re.IGNORECASE).strip()

            # --- Extract Kel/Desa ---
            kel_desa_match = re.search(r'(?:KEL/DESA|KELDASA|KAUS)\s*[: ]*([A-Z\s\.]+)', full_address_block)
            if kel_desa_match:
                extracted_data['Kel_Desa'] = kel_desa_match.group(1).strip()
                full_address_block = re.sub(re.escape(kel_desa_match.group(0)), '', full_address_block, flags=re.IGNORECASE).strip()

            # --- Extract Kecamatan ---
            kecamatan_match = re.search(r'(?:KECAMATAN|KEAMATAN)\s*[: ]*([A-Z\s\.]+)', full_address_block)
            if kecamatan_match:
                extracted_data['Kecamatan'] = kecamatan_match.group(1).strip()
                full_address_block = re.sub(re.escape(kecamatan_match.group(0)), '', full_address_block, flags=re.IGNORECASE).strip()

            # The remaining text in full_address_block should be the main street address
            extracted_data['Alamat'] = re.sub(r'^\s*[:\s]*', '', full_address_block).strip() # Remove colon or spaces at the beginning

            continue # Important to avoid double-processing the same line

        # Religion
        if "AGAMA" in line:
            agama_match = re.search(r'AGAMA\s*[: ]*([A-Z\s]+)', line)
            if agama_match:
                extracted_data['Agama'] = normalize_value("agama", agama_match.group(1))
            continue

        # Marital Status
        if "STATUS PERKAWINAN" in line or "SINTA PERKAMNAN" in line:
            status_match = re.search(r'(?:STATUS PERKAWINAN|SINTA PERKAMNAN)\s*[: ]*([A-Z\s]+)', line)
            if status_match:
                extracted_data['Status_Perkawinan'] = normalize_value("status_perkawinan", status_match.group(1))
            continue

        # Occupation
        if "PEKERJAAN" in line or "REHENAAAN" in line:
            # Search for "Pekerjaan" keyword or its typo, then get the rest of the line
            # More flexible to include "MENGURUS RUMAH TANGGA" or "KARYAWAN SWASTA"
            pekerjaan_match = re.search(r'(?:PEKERJAAN|REHENAAAN)\s*[: ]*([A-Z\s\.]+)', line)
            if pekerjaan_match:
                extracted_data['Pekerjaan'] = normalize_value("pekerjaan", pekerjaan_match.group(1))
            continue

        # Nationality
        if "KEWARGANEGARAAN" in line or "#EERGANEYER" in line:
            kewarganegaraan_match = re.search(r'(?:KEWARGANEGARAAN|#EERGANEYER)\s*[: ]*([A-Z]+)', line)
            if kewarganegaraan_match:
                extracted_data['Kewarganegaraan'] = normalize_value("kewarganegaraan", kewarganegaraan_match.group(1))
            continue

        # Valid Until
        if "BERLAKU HINGGA" in line or "BARTAU HINGGA" in line:
            berlaku_match = re.search(r'(?:BERLAKU HINGGA|BARTAU HINGGA)\s*[: ]*(SEUMUR HIDUP|\d{2}[-/]\d{2}[-/]\d{4})', line, re.IGNORECASE)
            if berlaku_match:
                extracted_data['Berlaku_Hingga'] = normalize_value("berlaku_hingga", berlaku_match.group(1))
                if extracted_data['Berlaku_Hingga'] is not None and re.match(r'\d{2}[-/]\d{2}[-/]\d{4}', extracted_data['Berlaku_Hingga']):
                    extracted_data['Berlaku_Hingga'] = extracted_data['Berlaku_Hingga'].replace('/', '-')
            continue
            
    # --- Return the extracted_data dictionary as populated, without explicit reordering ---
    return extracted_data

@app.route('/ocr/ktp', methods=['POST'])
def ocr_ktp():
    """
    API endpoint to upload a KTP image and get extracted data.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No 'file' part in the request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file:
        temp_path = None # Initialize temp_path outside try block
        try:
            # Save the file temporarily for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                file.save(tmp_file.name)
                temp_path = tmp_file.name
            
            # Call the KTP data extraction function
            extracted_data = extract_ktp_data(temp_path, lang='ind')
            
            # Check for errors from the data extraction function
            if "error" in extracted_data and extracted_data["error"]:
                return jsonify(extracted_data), 500
            
            return jsonify(extracted_data), 200
        except Exception as e:
            # Handle unexpected errors during the process
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500
        finally:
            # Ensure the temporary file is deleted
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    return jsonify({"error": "Unknown error."}), 500

@app.route('/')
def home():
    """
    Simple home page to check if the API is running.
    """
    return "KTP OCR API is running. Send a POST request to /ocr/ktp with a 'file' parameter containing the KTP image."

if __name__ == '__main__':
    # Run the Flask application.
    app.run(debug=True, host='0.0.0.0', port=5000)
