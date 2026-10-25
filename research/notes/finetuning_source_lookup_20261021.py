"""Bounded primary-source lookup. No model execution or training."""
import concurrent.futures
import datetime
import html
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path('D:/CEDAR/research_notes')
IDS = [
    '2308.03958', '2310.13548', '2403.10131', '2409.12917',
    '2306.03341', '2312.06681', '2106.09685', '2305.14314',
    '2310.01798', '2502.17424', '2404.13208', '2305.18290',
    '2404.03592',
    '2610.04329', '2610.13008', '2610.26655', '2610.25728',
    '2609.23205', '2609.35544',
    '2409.01658', '2412.00967',
]

def clean(s):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()

def fetch(pid):
    url = 'https://arxiv.org/abs/' + pid
    result = {'id': pid, 'url': url,
              'accessed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'CEDAR-literature-review/1.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            page = response.read().decode('utf-8')
        for key, name in [('title', 'citation_title'), ('date', 'citation_date')]:
            match = re.search(r'<meta\s+name="' + name + r'"\s+content="([^"]*)"', page)
            if match:
                result[key] = html.unescape(match.group(1))
        result['authors'] = [html.unescape(a) for a in re.findall(
            r'<meta\s+name="citation_author"\s+content="([^"]*)"', page)]
        match = re.search(r'<blockquote[^>]*class="abstract[^>]*>(.*?)</blockquote>', page, re.S)
        if match:
            result['abstract'] = clean(match.group(1))
        result['status'] = 'ok' if result.get('title') and result.get('abstract') else 'parse_incomplete'
    except Exception as exc:
        result.update(status='error', error=str(exc))
    return result

if __name__ == '__main__':
    path = ROOT / 'FINETUNING_PRIMARY_SOURCES_20261021.json'
    previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    existing = {row['id']: row for row in previous if row.get('status') == 'ok'}
    pending = [pid for pid in IDS if pid not in existing]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        existing.update({row['id']: row for row in pool.map(fetch, pending)})
    results = [existing[pid] for pid in IDS]
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    for result in results:
        if result['id'] in pending:
            print(json.dumps(result, ensure_ascii=False))
