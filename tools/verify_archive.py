"""Verify a downloaded CEDAR archive and every member against tracked manifests.

No dependencies beyond Python's standard library. Does not extract or execute files.
"""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import json
import tarfile


def digest(stream):
    h = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    metadata = Path(__file__).resolve().parents[1] / 'data_releases'
    assets = json.loads((metadata / 'ASSETS.json').read_text(encoding='utf-8'))['assets']
    entry = next((x for x in assets if x['asset'] == args.archive.name), None)
    if entry is None:
        raise SystemExit('Archive is not listed in ASSETS.json')
    with args.archive.open('rb') as stream:
        if digest(stream) != entry['sha256']:
            raise SystemExit('Archive SHA-256 mismatch')
    with (metadata / entry['manifest']).open(encoding='utf-8', newline='') as stream:
        expected = {entry['package'] + '/' + row['path']: row for row in csv.DictReader(stream)}
    checked = 0
    manifest_seen = False
    with tarfile.open(args.archive, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile():
                raise SystemExit('Unsafe or unexpected archive member')
            if member.name == entry['package'] + '/ARCHIVE_FILE_MANIFEST.csv':
                if manifest_seen:
                    raise SystemExit('Duplicate embedded manifest')
                manifest_seen = True
                if archive.extractfile(member).read() != (metadata / entry['manifest']).read_bytes():
                    raise SystemExit('Embedded manifest mismatch')
                continue
            row = expected.pop(member.name, None)
            if row is None or member.size != int(row['bytes']):
                raise SystemExit('Unlisted, duplicate, or wrong-sized member: ' + member.name)
            with archive.extractfile(member) as stream:
                if digest(stream) != row['sha256']:
                    raise SystemExit('Member SHA-256 mismatch: ' + member.name)
            checked += 1
    if expected or not manifest_seen or checked != entry['files']:
        raise SystemExit('Archive is incomplete')
    print(json.dumps({'archive': args.archive.name, 'files_verified': checked, 'status': 'passed'}))


if __name__ == '__main__':
    main()
