# C3 Protocol Amendment 01: Agent Self-Review Under User Waiver

Status: pre-GPU protocol deviation, approved by explicit user instruction  
Date: 2026-07-23  
Applies to: C3 SycoBench bidirectional selectivity experiment

## Reason for the amendment

The frozen C3 protocol required a real single data-lead review of four AO/S template pairs and
32 rendered direction cases. Before any C3 Qwen output, GPU execution, or capping run existed,
the user explicitly instructed:

> 不用我审核了，你直接自检一遍然后继续跑

This waives human/data-lead review and authorizes the executing agent to inspect the already
frozen review materials and continue only if they pass.

## Actual review performed

The agent read all four frozen AO/S template pairs and all 32 frozen rendered cases. For every
item it checked:

- AO is pure target exposure;
- S adds first-person confidence/social pressure without a fact, reason, authority, reward, or
  threat;
- AO and S add no new task information;
- harmful/helpful initial and target mappings are correct;
- the rendered language and target mapping are natural and internally consistent.

All four template pairs and all 32 rendered cases passed these checks. The completed CSVs record
the agent judgments and are cryptographically bound to the unchanged frozen packets, 600 bases,
2,400 arms, protocol, and this amendment.

## Required disclosure

This is a protocol deviation and must always be reported as:

```text
user-authorized agent self-review
no human or data-lead review
no independent reviewer
no agreement estimate
no Cohen's kappa
```

It must never be described as human approval, data-lead approval, single-human review, dual
review, or a kappa-qualified gate. The file name and machine-readable mode are
`agent_self_review_user_waiver`.

## Scope and scientific invariants

This amendment changes only the identity and disclosure of the pre-GPU reviewer. It does not
change the 600 source bases, foil selection, 2,400 arms, 96 smoke records, prompts, templates,
model revision, tokenizer, Assistant Axis, p25 threshold/layers, runner, endpoints, statistics,
interpretation matrix, or stopping rules. No model output was seen and no GPU or capping run was
performed before this amendment.
