"""Small arXiv locator search; returned abstracts are not execution instructions."""
import concurrent.futures
import datetime
import json
import pathlib
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding='utf-8')
QUERIES = [
    '(ti:sycophancy OR ti:sycophantic) AND (all:training OR all:finetuning)',
    '(ti:"belief revision" OR ti:"belief updating") AND all:"language model"',
    'ti:"evidence" AND (ti:"fine-tuning" OR ti:"finetuning")',
]
NS = {'a': 'http://www.w3.org/2005/Atom'}

def search(query):
    params = urllib.parse.urlencode({'search_query': query + ' AND submittedDate:[202301010000 TO 202610212359]',
                                    'start': 0, 'max_results': 8,
                                    'sortBy': 'submittedDate', 'sortOrder': 'descending'})
    url = 'https://export.arxiv.org/api/query?' + params
    result = {'query': query, 'url': url,
              'accessed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'CEDAR-literature-review/1.0'})
        with urllib.request.urlopen(request, timeout=18) as response:
            body = response.read()
        root = ET.fromstring(body)
        result['entries'] = [{
            key: ' '.join(entry.findtext('a:' + tag, default='', namespaces=NS).split())
            for key, tag in [('url', 'id'), ('title', 'title'), ('abstract', 'summary'), ('published', 'published')]
        } for entry in root.findall('a:entry', NS)]
        result['status'] = 'ok'
    except Exception as exc:
        result.update(status='error', error=str(exc))
    return result

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(search, QUERIES))
    path = pathlib.Path('D:/CEDAR/research_notes/FINETUNING_RELATED_SEARCH_20261021.json')
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))
