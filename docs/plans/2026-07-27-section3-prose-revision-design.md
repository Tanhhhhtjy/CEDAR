# CEDAR Section 3 Prose Revision Design

Date: 2026-07-27

## Scope

Revise only Section 3 in the English submission, Chinese review companion, and bilingual working draft. Section 4 and later sections remain unchanged.

## Style decisions

- Replace defensive `not X but Y`, `rather than`, and `never` constructions with direct descriptions of measurements and analysis rules.
- Preserve technical exclusions when they define an experimental condition or formal guarantee.
- Prefer active verbs and ordinary research language over internal audit terminology and self-certifying labels.
- Keep the English and Chinese versions semantically aligned without translating English syntax word for word.
- Use completed-study prose. Unknown values and appendix locations remain narrow `TBD` slots.

## Approved Section 3 changes

1. Define answer revision directly and describe CEDAR-Evidence and CEDAR-Social through their respective paired contrasts.
2. Retitle and rewrite Figure 1's caption as `Controlled contrasts for answer revision`, retaining the statement that revision is formally warranted only under `V`.
3. Split the CEDAR-Evidence description into task construction, formal certification, and dataset scale/statistical unit.
4. Describe `AO` and `S` as sharing target exposure and a renewed answer request, with social framing added in `S`.
5. Add a pending two-researcher, full-set, output-blind audit of all 600 CEDAR-Social pairs; agreement and adjudication outcomes remain `TBD`.
6. Describe the cap operator positively and report Qwen, Llama, and Gemma settings separately by model.
7. Rename Section 3.5 to `Behavioral Outcomes and Logit Margins` and distinguish continuous answer geometry from realized four-state choices.
8. Rename Section 3.6 to `Statistical Inference and Quality Control` and express inclusion, resampling, and diagnostic rules directly.

## Verification

- Compile `main.tex` and `main_zh.tex`.
- Check for undefined references, overfull boxes, and encoding problems.
- Confirm that Section 4 text and headings are unchanged.
- Review the synchronized Section 3 prose in all three maintained files.
