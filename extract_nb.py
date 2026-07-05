import json

with open('credit_scoring.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('credit_scoring.py', 'w', encoding='utf-8') as f:
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            f.write("".join(cell.get('source', [])))
            f.write("\n\n")
