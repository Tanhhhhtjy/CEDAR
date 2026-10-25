"""Restore pinned public assets into the isolated ARR workspace; no credentials."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'cache' / 'hub'
OUT = ROOT / 'explore_outputs'
OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT / 'assets_status.json'
EVENTS = OUT / 'download_retry_events.jsonl'


def write(state, **kwargs):
    payload = dict(status=state, updated=time.time(), **kwargs)
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(STATUS)
    print(json.dumps(payload), flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download_with_retry(label, operation, *, max_attempts=16, **metadata):
    from requests.exceptions import ConnectionError, Timeout, ChunkedEncodingError, HTTPError
    for attempt in range(1, max_attempts + 1):
        write(label, attempt=attempt, max_attempts=max_attempts, **metadata)
        try:
            return operation()
        except (ConnectionError, Timeout, ChunkedEncodingError, HTTPError) as error:
            if isinstance(error, HTTPError):
                code = getattr(error.response, 'status_code', None)
                if code not in (408, 429, 500, 502, 503, 504):
                    raise
            delay = min(15 * 2**(attempt - 1), 60)
            event = dict(operation=label, attempt=attempt, time=time.time(),
                error_class=type(error).__name__, error=str(error).splitlines()[0][:350],
                retry_in_seconds=delay if attempt < max_attempts else None)
            with EVENTS.open('a') as f:
                f.write(json.dumps(event) + '\n')
            if attempt == max_attempts:
                raise
            write('retrying_download', **event, **metadata)
            # Hugging Face retains .incomplete blobs and resumes their byte ranges.
            time.sleep(delay)


def main():
    from huggingface_hub import snapshot_download, hf_hub_download
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    kwargs = dict(cache_dir=str(CACHE), token=False)
    repo = 'Qwen/Qwen3-32B'
    revision = '9216db5781bf21249d130ec9da846c4624c16137'
    model = download_with_retry('downloading_tokenizer_and_cap_assets', lambda: snapshot_download(repo, revision=revision,
        allow_patterns=['*.json', '*.jinja', 'merges.txt', 'vocab.json'],
        max_workers=2, **kwargs))
    assets = {}
    for name, expected in [
        ('assistant_axis.pt', 'a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211'),
        ('capping_config.pt', '6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24')]:
        p = download_with_retry('downloading_cap_assets', lambda: hf_hub_download('lu-christina/assistant-axis-vectors',
            filename='qwen-3-32b/' + name, repo_type='dataset',
            revision='3b3b788432ad33e3a28d9ff08e88a530c0740814', **kwargs), asset=name)
        actual = sha(p)
        assert actual == expected, (name, actual)
        assets[name] = dict(path=p, sha256=actual)
    (OUT / 'asset_paths.json').write_text(json.dumps(dict(model=model, assets=assets), indent=2))
    free = shutil.disk_usage(ROOT).free
    index = json.loads((Path(model) / 'model.safetensors.index.json').read_text())
    blobs = CACHE / 'models--Qwen--Qwen3-32B' / 'blobs'
    downloaded = sum(p.stat().st_size for p in blobs.iterdir()
        if p.is_file() and re.fullmatch(r'[0-9a-f]{64}(\.incomplete)?', p.name))
    remaining = max(0, index['metadata']['total_size'] + 32 * 1024**2 - downloaded)
    if free < remaining + 10 * 1024**3:
        raise RuntimeError(f'Insufficient free bytes: {free}; estimated remaining download: {remaining}')
    download_with_retry('downloading_pinned_weights', lambda: snapshot_download(
        repo, revision=revision, allow_patterns=['*.safetensors'], max_workers=2, **kwargs),
        model=model, free_bytes=free, assets=assets, estimated_remaining_bytes=remaining)
    write('verifying_weights', model=model, assets=assets)
    files = {}
    for p in sorted(Path(model).iterdir()):
        if p.is_file():
            digest = sha(p)
            # HF LFS blobs are named by their SHA-256; verify all model shards.
            if p.suffix == '.safetensors':
                assert digest == p.resolve().name, f'Weight hash mismatch: {p}'
            files[p.name] = dict(size=p.stat().st_size, sha256=digest)
    index = json.loads((Path(model) / 'model.safetensors.index.json').read_text())
    assert set(index['weight_map'].values()) <= set(files)
    write('ready', model=model, assets=assets, files=files, revision=revision)


if __name__ == '__main__':
    import fcntl
    lock = (OUT / 'assets.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except Exception as e:
        write('failed', error=type(e).__name__ + ': ' + str(e))
        traceback.print_exc()
        raise
