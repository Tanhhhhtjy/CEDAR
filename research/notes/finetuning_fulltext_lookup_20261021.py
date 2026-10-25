"""Read selected publisher-hosted arXiv HTML for targeted source checks."""
import concurrent.futures
import datetime
import json
import pathlib
import re
import sys
import urllib.request
from html.parser import HTMLParser

sys.stdout.reconfigure(encoding='utf-8')

class PlainTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

ROOT = pathlib.Path('D:/CEDAR/research_notes')
PAPERS = {
    '2409.01658': ['limitation', 'correct', 'pinpoint', 'LoRA'],
    '2412.00967': ['limitation', 'correct', 'penalt', 'probe'],
}

def read_paper(item):
    pid, terms = item
    url = 'https://arxiv.org/html/' + pid + 'v1'
    record = {'id': pid, 'url': url,
              'accessed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'CEDAR-literature-review/1.0'})
        with urllib.request.urlopen(request, timeout=18) as response:
            raw = response.read()
        page = raw.decode('utf-8', errors='replace')
        page = re.sub(r'<(script|style|nav|footer)\b[^>]*>.*?</\1>', ' ', page, flags=re.S | re.I)
        parser = PlainTextParser()
        parser.feed(page)
        text = ' '.join(parser.parts)
        text = re.sub(r'\s+', ' ', text)
        record.update(status='ok', characters=len(text))
        text_path = ROOT / ('FINETUNING_SOURCE_' + pid.replace('.', '_') + '_20261021.txt')
        text_path.write_text(text, encoding='utf-8')
        record['local_text'] = str(text_path)
        record['excerpts'] = {}
        for term in terms:
            matches = list(re.finditer(re.escape(term), text, re.I))
            record['excerpts'][term] = [text[max(0, m.start()-250):m.end()+750] for m in matches[1:7]]
    except Exception as exc:
        record.update(status='error', error=str(exc))
    return record

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read_paper, PAPERS.items()))
    path = ROOT / 'FINETUNING_FULLTEXT_CHECKS_20261021.json'
    previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    combined = {row['id']: row for row in previous}
    combined.update({row['id']: row for row in results})
    path.write_text(json.dumps(list(combined.values()), ensure_ascii=False, indent=2), encoding='utf-8')
    for result in results:
        print(json.dumps({k:v for k,v in result.items() if k != 'excerpts'}, ensure_ascii=False))
