import re, wmi
from rapidfuzz import process, fuzz

def normalizeCPUName(name):
    name = name.lower()
    name = re.sub(r"w/.*$", "", name)
    name = re.sub(r"®|™", "", name)
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\bcpu\b|\bprocessor\b", "", name)
    name = re.sub(r"@\s*\d+(\.\d+)?\s*ghz", "", name)
    name = re.sub(r"[^a-z0-9]+", " ", name).strip()
    return name

def normalizeGPUName(name):
    name = name.lower()
    name = re.sub(r"®|™", "", name)
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\b(?:nvidia|amd|ati|intel)\b", "", name)
    name = re.sub(r"\bgpu\b|\bgraphics\b|\bvideo\b|\bcontroller\b", "", name)
    name = re.sub(r"[^a-z0-9]+", " ", name).strip()
    return name

def getDetails(df, name):
    return df[df["name"] == name]

def getCPU():
    c = wmi.WMI()
    for cpu in c.Win32_Processor():
        return normalizeCPUName(cpu.Name)
    
def getGPU():
    c = wmi.WMI()
    for gpu in c.Win32_VideoController():
        return normalizeGPUName(gpu.Name)