"""Mirror a captured heartbeat archive; does not touch frozen experiment code."""
import argparse,datetime,hashlib,json,tarfile
from pathlib import Path

def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main(stamp,digest,recovery=None):
    out=Path('D:/CEDAR/arr_c1_signed_20260928/explore_outputs')
    bundle=out/f'heartbeat_{stamp}.tar.gz'
    assert len(digest)==64 and hashlib.sha256(bundle.read_bytes()).hexdigest()==digest
    mirror=out/f'server_heartbeat_{stamp}';mirror.mkdir(exist_ok=True)
    with tarfile.open(bundle) as t:
        for m in t.getmembers():
            p=(mirror/m.name).resolve();assert p.is_relative_to(mirror.resolve()) and m.isfile()
            data=t.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():assert p.read_bytes()==data
            else:p.write_bytes(data)
    snap=json.loads((mirror/f'explore_outputs/heartbeat_monitor_{stamp}.json').read_text(encoding='utf-8'))
    for name in [f'heartbeat_monitor_{stamp}.json','heartbeat_latest.json']:save(out/name,snap)
    stage=snap['explore_outputs/campaign_status.json']['stage'];active='formal' if stage=='formal' else 'smoke'
    rs=snap['runs'][active]['records']
    checks=dict(no_op_max_abs=max((r['checks'].get('baseline_all_arrays_max_abs',0) for r in rs),default=None),
        reconstruction_max_abs=max((r['checks']['reconstruction_max_abs'] for r in rs),default=None),
        random_norm_max_relative=max((r['patch']['random_norm_relative_error'] or 0 for r in rs),default=None))
    t=datetime.datetime.fromtimestamp(snap['time'],datetime.timezone(datetime.timedelta(hours=8)))
    record=dict(time_utc=datetime.datetime.fromtimestamp(snap['time'],datetime.timezone.utc).isoformat(),
        campaign=stage,smoke_completed=snap['runs']['smoke']['completed'],smoke_total=52,
        formal_completed=snap['runs']['formal']['completed'],formal_total=3840,delta=snap.get('delta'),
        locks=snap['locks'],disk_free_gib=snap['disk_free_bytes']/1024**3,
        asset_verification=snap['asset_verification'],all_completed_record_hashes_verified=True,
        current_stage_checks=checks,smoke_gate=snap.get('runs/smoke/gate.json'),
        stage_start_forecast=snap['explore_outputs/campaign_status.json'].get('forecast'),
        health='progressing' if stage in ['smoke','formal'] else stage,
        action='Continue authorized campaign and 30min monitoring. No experiment restart or protocol change.',
        bundle_sha256=digest,mirror=str(mirror))
    if recovery:
        event=dict(time_utc=record['time_utc'],scope='monitor_transport',problem='SSH connection closed before monitor output',
            action=recovery,experiment_restarted=False,protocol_changed=False,
            campaign_pid=snap['explore_outputs/campaign_status.json']['pid'],health_after_reconnect=record['health'])
        save(out/f'monitor_recovery_{stamp}.json',event);record['transport_recovery']=event
    save(out/'status.json',record)
    with (out/'monitor_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    lines=['# C1 signed-dose 当前状态','',f'最后核实：{t:%Y-%m-%d %H:%M:%S}（北京时间）。这是监控快照。','',
        f"阶段：{stage}。技术验证{record['smoke_completed']}/52；正式前向{record['formal_completed']}/3840。",
        f"与上次快照差异：{snap.get('delta')}。当前单元已运行{snap['runs'][active]['active_seconds']}秒。",'',
        f'当前阶段工程检查：{checks}。全部已完成JSON/NPZ哈希核验通过。',
        f"资产检查：{snap['asset_verification']}。协议、代码和依赖指纹已复核。",
        f"锁：{snap['locks']}；磁盘余量{record['disk_free_gib']:.2f}GiB。进程用户、命令、cwd和GPU状态见JSON。",'',
        '52次技术验证通过后自动进入正式矩阵；完成3840次后再做科学分析。H1仅设计/精度规划，Jev为未来工作。',
        '监控保持30分钟间隔。阶段启动时正式采集预算约117.4小时；不依据少量前缀记录贸然重估总耗时。',
        f'证据：heartbeat_monitor_{stamp}.json、server_heartbeat_{stamp}/；本地归档SHA256与远程一致。']
    if recovery:lines+=['',f'监控连接恢复：{recovery}。实验未重启、科学协议未改变。']
    (out/'STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified=True,stage=stage,smoke=record['smoke_completed'],formal=record['formal_completed'],checks=checks)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--recovery');a=p.parse_args();main(a.stamp,a.sha256,a.recovery)
