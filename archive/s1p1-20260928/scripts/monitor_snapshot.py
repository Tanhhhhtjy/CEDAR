"""Read-only process/resource inspection; write timestamped monitoring records."""
from pathlib import Path
import base64
import datetime
import json
import os
import shutil
from stat import S_ISREG
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'explore_outputs'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    old = read(OUT / 'monitor_state.json')
    campaign = read(OUT / 'campaign_status.json')
    assets = read(OUT / 'assets_status.json')
    processes = []
    # Restrict command inspection to processes with this exact working directory.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / 'cwd').resolve() != ROOT:
                continue
            cmd = (proc / 'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            roles = [name for name in ('fetch_assets.py', 'run_campaign.py', 'run_arr_acquisition.py', 'analyze_s1.py', 'analyze_p1.py')
                     if any(arg.endswith('/' + name) or arg == name for arg in cmd)]
            if not roles:
                continue
            processes.append(dict(pid=int(proc.name), cwd=str(ROOT), command=cmd, roles=roles,
                                  state=(proc / 'stat').read_text().split(') ', 1)[1].split()[0]))
        except (OSError, PermissionError):
            continue
    blobs = ROOT / 'cache/hub/models--Qwen--Qwen3-32B/blobs'
    blob_stats = {}
    for path in blobs.iterdir() if blobs.exists() else ():
        try:
            blob_stat = path.stat()
        except FileNotFoundError:
            # Hugging Face renames completed partial files during inspection.
            if path.suffix != '.incomplete':
                continue
            try:
                blob_stat = path.with_suffix('').stat()
            except FileNotFoundError:
                continue
        if S_ISREG(blob_stat.st_mode):
            # The directory iterator may see both names for the same inode.
            blob_stats[(blob_stat.st_dev, blob_stat.st_ino)] = blob_stat
    size = sum(s.st_size for s in blob_stats.values())
    run_states = {}
    record_counts = {}
    for scope in ('smoke', 'formal'):
        folder = ROOT / 'runs' / scope
        run_states[scope] = read(folder / 'status.json')
        record_counts[scope] = sum(1 for condition in ('uncapped', 'full_p25', 'uncapped_repeat')
            for p in folder.glob('????_' + condition + '.json') if p.with_suffix('.npz').exists())
    logs = {}
    for name in ('fetch_assets', 'campaign', 'smoke', 'formal', 's1_analysis', 'p1_analysis'):
        path = ROOT / 'logs' / (name + '.log')
        if path.exists():
            stat = path.stat()
            logs[name] = dict(bytes=stat.st_size, modified_epoch=stat.st_mtime,
                growth_bytes=stat.st_size-old.get('logs', {}).get(name, {}).get('bytes', stat.st_size))
    gpu_raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    gpu = [dict(zip(('index', 'total_mib', 'free_mib', 'utilization_percent'), [int(x.strip()) for x in line.split(',')]))
           for line in gpu_raw.strip().splitlines()]
    state = dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        automation_id='cedar', interval_minutes=30, campaign_stage=campaign.get('stage'),
        campaign_pid=campaign.get('pid'), assets_status=assets.get('status'),
        asset_attempt=assets.get('attempt'), downloaded_blob_bytes=size,
        download_growth_bytes=size-old.get('downloaded_blob_bytes', size),
        latest_blob_modification_epoch=max((s.st_mtime for s in blob_stats.values()), default=None),
        processes=processes, run_states=run_states, completed_records=record_counts,
        completed_new_model_forwards=sum(record_counts.values()), logs=logs, gpus=gpu,
        disk_free_bytes=shutil.disk_usage(ROOT).free,
        campaign_error=campaign.get('error'), assets_error=assets.get('error'),
        retry_events=sum(1 for _ in (OUT/'download_retry_events.jsonl').open()) if (OUT/'download_retry_events.jsonl').exists() else 0,
        inspection_epoch=time.time())
    data = json.dumps(state, indent=2, sort_keys=True).encode() + b'\n'
    temporary = OUT / 'monitor_state.tmp'
    temporary.write_bytes(data)
    temporary.replace(OUT / 'monitor_state.json')
    with (OUT / 'monitor_history.jsonl').open('a') as f:
        f.write(json.dumps(state, sort_keys=True) + '\n')
    encoded = base64.b64encode(data).decode()
    print('\nMONITOR_SNAPSHOT_BEGIN')
    for i in range(0, len(encoded), 60):
        print('MONDATA:' + encoded[i:i+60])
    print('MONITOR_SNAPSHOT_END', flush=True)


if __name__ == '__main__':
    main()
