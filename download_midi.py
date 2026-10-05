import re, os, requests, urllib3
from urllib.parse import urljoin
urllib3.disable_warnings()

os.makedirs("data/midi", exist_ok=True)
headers = {"User-Agent": "Mozilla/5.0"}
pages = {"chopin": "chopin.htm", "bach": "bach.htm", "beethoven": "beeth.htm", "mozart": "mozart.htm"}

for composer, name in pages.items():
    page = "https://www.piano-midi.de/" + name
    html = requests.get(page, headers=headers, timeout=30, verify=False).text
    links = sorted(set(re.findall(r'href=["\']([^"\']+\.mid)["\']', html, flags=re.I)))[:15]
    print(composer, len(links), "files found")
    for l in links:
        url = urljoin(page, l)
        out = os.path.join("data/midi", composer + "_" + os.path.basename(l))
        r = requests.get(url, headers=headers, timeout=30, verify=False)
        if r.ok:
            open(out, "wb").write(r.content)