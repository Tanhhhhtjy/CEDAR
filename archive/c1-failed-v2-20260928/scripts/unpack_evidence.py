"""Verify and safely mirror the server evidence on Windows without losing names."""
import hashlib,json,tarfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'evidence_metadata_20260928T054910Z.tgz'
EXPECTED='9e0868b46fc7dbdbbed8b463446d66728962a961e7acd67b4586a58bf47014e8'
def sha(data):return hashlib.sha256(data).hexdigest()
assert sha(ARCHIVE.read_bytes())==EXPECTED
dest=ROOT/'server_evidence_20260928T054910Z'
seen=set();manifest=[]
with tarfile.open(ARCHIVE) as archive:
    for member in archive.getmembers():
        if member.isdir():continue
        assert member.isfile(),member.name
        source=PurePosixPath(member.name)
        assert not source.is_absolute() and '..' not in source.parts
        parts=[p.replace(':','_COLON_') for p in source.parts]
        target=dest.joinpath(*parts).resolve()
        assert target.is_relative_to(dest.resolve()) and target not in seen
        seen.add(target);data=archive.extractfile(member).read()
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
        manifest.append(dict(server_name=member.name,local_name=str(target.relative_to(ROOT)),sha256=sha(data),bytes=len(data)))
out=ROOT/'explore_outputs/evidence_download_manifest.json'
out.write_text(json.dumps(dict(archive=ARCHIVE.name,archive_sha256=EXPECTED,complete=True,
    excludes='Direction tensors and original 384 NPZ remain on server; direction metadata and hashes included.',
    entries=manifest),indent=2)+'\n')
print(json.dumps(dict(status='verified_and_extracted',files=len(manifest),destination=str(dest))))
