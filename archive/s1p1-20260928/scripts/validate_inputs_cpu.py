"""Verify restored tokenizer, exact historical tokens, and official cap assets."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import run_arr_acquisition as acq
from transformers import AutoTokenizer

paths=json.loads((acq.ROOT/'explore_outputs/asset_paths.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(paths['model'],local_files_only=True)
prompts={r['id']:r for r in acq.readl(acq.PACKAGE/'data/phase_validity_inputs/answer_inputs96.jsonl')}
traces=acq.readl(acq.ROOT/'inputs/frozen_traces192.jsonl')
anchors={r['trace_id']:r for r in acq.readl(acq.PACKAGE/'runs/fixed_trace_replay/formal-fixed-trace-seed0/results.jsonl')}
for trace in traces:
    text=tokenizer.apply_chat_template(prompts[trace['source_id']]['messages'],tokenize=False,
        add_generation_prompt=True,enable_thinking=True)
    ids=tokenizer.encode(text,add_special_tokens=False)
    assert acq.historical.object_sha256(text)==trace['prompt_sha256']
    assert acq.historical.object_sha256(ids)==trace['prompt_token_ids_sha256']
    assert acq.historical.object_sha256(trace['trace_prefix_token_ids'])==trace['trace_prefix_token_ids_sha256']
    assert acq.historical.object_sha256(ids+trace['trace_prefix_token_ids'])==trace['teacher_forced_input_token_ids_sha256']
    for key in ['prompt_sha256','trace_prefix_token_ids_sha256','teacher_forced_input_token_ids_sha256']:
        assert trace[key]==anchors[trace['trace_id']][key]
    for label,token in trace['candidate_token_ids'].items():
        assert tokenizer.encode(label,add_special_tokens=False)==[token]
caps,meta=acq.historical.e2run.load_p25_assets(acq.ROOT/'cache/hub')
assert tuple(caps)==tuple(range(46,54))
acq.write(acq.ROOT/'explore_outputs/CPU_INPUT_VALIDATION.json',dict(passed=True,
    traces=len(traces),tokenizer_and_prompt_token_identity_verified=True,
    submitted_replay_hashes_verified=True,official_cap=meta,scientific_evidence=False))
print('PASS: all 192 original prompt/prefix token hashes, option IDs, and official cap orientation verified.')
