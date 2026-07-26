# CEDAR AAAI-27 citation audit

Date: 2026-07-27

Paper source branch inspected: `origin/agent/aaai27-paper-writing`
Paper source commit inspected: `d183700c90ec7e9479a2a48cc007e182e7b324ba`

## Scope

This audit resolves the Internal Citation Queue in
`cedar-aaai27-working-draft.md` into stable BibTeX keys. Bibliographic
metadata was taken from the corresponding official proceedings page, DOI
record, OpenReview page, or arXiv record. No unverified placeholder entry was
added.

The bibliography is `references.bib` in this directory. It contains the
following verified keys:

| Queue item | BibTeX key | Primary metadata source | Status |
|---|---|---|---|
| Assistant Axis | `lu2026assistantaxis` | [arXiv:2601.10387](https://arxiv.org/abs/2601.10387) | Preprint metadata and title page checked |
| Dual-Stance Evaluation of Sycophancy | `buchan2026dualstance` | [arXiv:2606.11205](https://arxiv.org/abs/2606.11205) | Preprint metadata and title page checked; the paper has one author, Matthew James Buchan |
| When Truth Is Overridden | `wang2026truthoverridden` | [AAAI DOI record](https://doi.org/10.1609/aaai.v40i39.40645) | AAAI volume, issue, pages, authors, and DOI checked |
| Refusal in Language Models Is Mediated by a Single Direction | `arditi2024refusal` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html) | Proceedings metadata checked |
| WMDP | `li2024wmdp` | [PMLR](https://proceedings.mlr.press/v235/li24bc.html) | Official PMLR BibTeX and full author list used |
| Belief-R | `wilie2024beliefrevision` | [ACL Anthology](https://aclanthology.org/2024.emnlp-main.586/) | Official Anthology BibTeX used |
| SycoBench-600 | `sinha2026sycobench` | [ACL Anthology](https://aclanthology.org/2026.findings-acl.1759/) | Official Anthology BibTeX used |
| Contrastive Activation Addition | `rimsky2024caa` | [ACL Anthology](https://aclanthology.org/2024.acl-long.828/) | Official Anthology BibTeX used |
| Inference-Time Intervention | `li2023iti` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html) | Proceedings metadata checked |
| Towards Understanding Sycophancy in Language Models | `sharma2024sycophancy` | [OpenReview](https://openreview.net/forum?id=tvhaxkMKAn) | ICLR 2024 venue and author list checked against the paper record |
| KL-then-steer / Steering Without Side Effects | `stickland2024steering` | [arXiv:2406.15518](https://arxiv.org/abs/2406.15518) | Preprint metadata checked; KL-then-steer is the method name, not a second paper |
| BASIL | `atwell2026basil` | [ACM DOI record](https://doi.org/10.1145/3805689.3812404) | FAccT 2026 proceedings metadata checked |
| T3 | `chang2026causalt3` | [arXiv:2601.08258](https://arxiv.org/abs/2601.08258) | Correct paper title is *Diagnosing and Mitigating Sycophancy and Skepticism in LLM Causal Judgment*; CAUSALT3 is the benchmark name |

## July 2026 nearest-neighbor check

The queue's phrase "nearest July 2026 work" did not identify a unique paper.
Two distinct, directly relevant works were therefore checked rather than
silently choosing one:

- `yang2026resistupdate`: *Resist and Update: Counterfactual Report
  Coordinates for Incentive-Compatible LLMs*, arXiv:2607.12985. The title
  page, abstract, related-work section, causal interchange design, and
  resist/update evaluation were checked in the full text. It is the closest
  current neighbor to CEDAR's distinction between resisting non-evidential
  pressure and updating to licensed evidence.
- `aravindan2026opium`: *OPIUM: Mitigating Steering Externalities and
  Over-Refusal via Dual Objective Latent Optimization*, arXiv:2607.19806,
  ICML 2026 Mechanistic Interpretability Workshop. The title page, abstract,
  intervention definition, and dual-objective side-effect evaluation were
  checked in the full text. It is the closest current neighbor to the
  steering-side-effect and mitigation framing.

Both are recent papers, so they should be cited as nearby work rather than as
settled support for a novelty claim. Their presence in `references.bib` does
not imply that either result has been reproduced by CEDAR.

## Manuscript integration and compile gate

The inspected paper branch contains a bilingual Markdown working draft, but
no LaTeX manuscript entry point, no `aaai2027.sty`, no `aaai2027.bst`, and no
paper build command. Consequently, there is no existing location in which a
`\bibliography{...}` command can be restored, and a full AAAI manuscript compile
cannot be honestly reported from this branch. Creating a new paper template
or translating the draft would be a separate writing task and was deliberately
not done here.

The current achievable checks are:

1. parse `references.bib` and reject duplicate or missing keys;
2. cite all keys in an isolated, disposable LaTeX harness and run BibTeX;
3. verify that the harness has no undefined citations or missing bibliography
   entries;
4. keep the harness and downloaded toolchain outside the repository.

These checks were completed with Tectonic 0.16.9 and the standard `plain`
BibTeX style. The disposable harness cited all 15 keys, produced 15
`\bibitem` entries, and emitted no BibTeX error, undefined-citation warning,
or missing-entry warning. This is a bibliography syntax and key-coverage
check; it is not a substitute for compiling the absent AAAI manuscript.

The full submission gate remains pending until the actual LaTeX source is
added or identified. At that point, the source must use its official AAAI-27
bibliography integration, cite only keys present in `references.bib`, and pass
the complete manuscript build with no undefined citation.

## Anonymity and provenance check

The committed bibliography contains only public scholarly metadata. It does
not contain author identities for the CEDAR submission, authentication data,
local absolute paths, or private URLs.
