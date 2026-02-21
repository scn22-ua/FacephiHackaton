import re
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)

def check_digit(data):
    weights, total = [7, 3, 1], 0
    for i, c in enumerate(data):
        val = ord(c)-ord('0') if '0'<=c<='9' else ord(c)-ord('A')+10 if 'A'<=c<='Z' else 0
        total += val * weights[i % 3]
    return str(total % 10)

def extract_text(ocr_model, img_path):
    return ocr_model(DocumentFile.from_images(img_path))

def show_ocr_result(result):
    result.show()
    for p in result.pages:
        for b in p.blocks:
            for l in b.lines:
                print(" ".join([w.value for w in l.words if w.confidence > 0.5]))

def get_data_from_res(res, is_back=False):
    lines = ["".join([w.value for w in l.words if w.confidence > 0.4]).upper() for p in res.pages for b in p.blocks for l in b.lines]
    if is_back:
        for s in lines:
            if len(s) >= 30 and s[0] in "IAC" and ("<" in s or re.search(r'[A-Z]{3}\d{6}', s)):
                return {"tipo": s[:2], "pais": s[2:5], "num": s[5:14].replace("<",""), "dc": s[14], "opc": s[15:30], "ok": check_digit(s[5:14]) == s[14]}
        return None
    full_text = " ".join(lines)
    idesps = re.findall(r'([A-Z]{3}\d{6}\d?)', full_text)
    idesp = next((i for i in idesps if not i.startswith("DNI")), None)
    return {"dni": (re.search(r'(\d{8}[A-Z])', full_text) or [None])[0], "idesp": idesp}

# --- PROCESO ---
print("Procesando Front...")
text_front = extract_text(model, 'dni_front_especimen.jpg')
show_ocr_result(text_front)

print("\nProcesando Back...")
text_back = extract_text(model, 'dni_back_especimen.jpg')
show_ocr_result(text_back)

f = get_data_from_res(text_front)
b = get_data_from_res(text_back, True)

print("\n" + "="*50 + "\n  RESULTADOS COMPARATIVA\n" + "="*50)
if f and b:
    print(f"MRZ L1: {b['tipo']} | {b['pais']} | {b['num']} | DC:{b['dc']} ({'OK' if b['ok'] else 'ERR'}) | OBC:{b['opc']}")
    print(f"FRONT:  DNI:{f['dni']} | IDESP:{f['idesp']}")
    print("-" * 50)
    match_id = f['idesp'].startswith(b['num']) if f['idesp'] else False
    print(f"ESTADO: {'COHERENTE' if match_id and b['ok'] else 'DISCREPANCIA'}")
else:
    print("Error en extracción")
print("="*50)
