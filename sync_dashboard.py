import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index_html_path = os.path.join(BASE_DIR, "index.html")

with open(index_html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

def sync_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    marker = 'HTML_DASHBOARD = r"""'
    idx = content.find(marker)
    if idx == -1:
        marker = 'HTML_DASHBOARD = """'
        idx = content.find(marker)

    if idx == -1:
        print(f"[ERROR] Could not find HTML_DASHBOARD marker in {file_path}")
        return False

    prefix = content[:idx + len(marker)]
    new_content = prefix + html_content + '"""\n'

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[SUCCESS] Synchronized index.html into {file_path}")
    return True

sync_file(os.path.join(BASE_DIR, "main.py"))
sync_file(os.path.join(BASE_DIR, "api", "index.py"))
