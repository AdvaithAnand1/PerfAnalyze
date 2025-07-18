import re
from rapidfuzz import process, fuzz

def normalize_cpu_name(name):
    # 1) Lowercase
    name = name.lower()
    # 2) Remove any “w/ …” and everything after it
    name = re.sub(r"w/.*$", "", name)
    # 3) Drop trademark symbols, parentheses and their contents, “cpu”/“processor”
    name = re.sub(r"®|™", "", name)
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\bcpu\b|\bprocessor\b", "", name)
    # 4) Remove frequencies (“@ 2.60ghz”, etc.)
    name = re.sub(r"@\s*\d+(\.\d+)?\s*ghz", "", name)
    # 5) Collapse non-alphanumerics to spaces, strip
    name = re.sub(r"[^a-z0-9]+", " ", name).strip()
    return name
