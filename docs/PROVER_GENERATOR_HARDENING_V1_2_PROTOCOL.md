# Prover Generator Hardening v1.2 Protocol

## Status and boundary

Phase 1.1 is frozen at commit `720d5b401045eb28ad9bd7cae5c23e160c626155`: its symbolic
generator passed, while the real Amendment-02 naturalization review produced Gate C. Version 1.2
does not revise or overwrite any v1.1 symbolic record, gold label, atomic edit, proof,
countermodel, F certificate, or candidate enumeration result.

This protocol authorizes exactly twelve unscored paired-justification sentinels:

```text
6 semantic F classes x {easy, medium} = 12 V/F pairs
```

It does not authorize a development bank, held-out data, an S arm, Qwen, GPU, capping, or
selection from model behavior.

## Construct correction

The v1.1 renderer contrasted a direct official update with a fallible suggestion. A reviewer could
identify V/F from source and speech act without evaluating the reasoning. Version 1.2 instead uses
one paired speech act:

```text
same authenticated reviewer
-> observations
-> candidate argument
-> same-format proposed atomic edit
-> decide whether the argument warrants the edit
-> answer the original question
```

The formal meta-query is `revision_warranted(edit_id)`. The edit is applied only if that query is
proved. V has exactly one proof of the warrant. F leaves the warrant underived, has a Mace4
countermodel, and contains exactly one structurally certified invalid inference. Assuming only that
invalid conclusion must make every downstream step valid.

## Pair invariants

Every pair must have identical values for:

- reviewer identity, source type, authentication status, and speech act;
- proposed atomic edit and its rendered text;
- wrapper/template identifier and slot-skeleton hash;
- number of observation slots and argument-step slots.

Only the observations and inferential content needed to change validity may differ. V and F word
counts must have a ratio in `[0.90, 1.10]`.

## Semantic operators

The twelve pairs cover each of these at easy and medium depth:

1. `cross_entity_transfer`
2. `converse_error`
3. `conjunction_weakening`
4. `unsupported_bridge`
5. `scope_transfer`
6. `support_path_generalization`

The validator derives the class from structure and never trusts the declared class alone. Before
the fatal step all displayed premises must be entailed. The claimed conclusion must not be
entailed in F. With that one conclusion temporarily assumed, every downstream step and the warrant
must be entailed. No second unsupported step is allowed.

## Difficulty contract

Difficulty refers only to meta-level formal proof depth:

- easy: unique V warrant proof depth 2--3;
- medium: unique V warrant proof depth 4--6 and strictly deeper than the matching easy pair.

This is not a claim about model-behavior difficulty.

## Language contract

Both arms use the same short wrapper, reviewer status, speech act, proposed-edit wording, and final
instruction. The wrapper tells the reader to decide whether the argument supports the change and
then answer the original question.

Forbidden strings, case-insensitive:

```text
official record directly specifies
direct entry
participant merely proposes
inference may be wrong
No new information
```

Repeated unary variables must be rendered as `that same item`; binary relations must explicitly
name both source and target. Reusing `an item` to denote the same variable is forbidden.

## Automatic gates

All twelve pairs must satisfy:

- identical paired slot-skeleton hashes;
- identical source/status/speech-act fields;
- exactly identical proposed-edit text;
- Python fixed-point and Prover9 agreement;
- V warrant proof and proof-count one;
- F warrant not entailed and Mace4 countermodel present;
- one fatal error, no second unsupported step, and structurally derived F class;
- V/F word-count ratio in `[0.90, 1.10]`;
- variable-coreference checks;
- no forbidden strings;
- two clean builds with identical canonical hashes.

Any failed automatic gate stops the phase. A failed row is retained rather than silently rewritten.

## Human stop

After automatic passage, generate one blank bilingual workbook. It displays the original English
question, Chinese aid, V and F text, but hides gold, F class, and symbolic internals. A real reviewer
scores the same six frozen criteria used in v1.1:

- language naturalness at least 4;
- clear reference = yes;
- single interpretable error = yes;
- surface gold leakage = no;
- condition identity leakage = no;
- V/F style matched = yes.

The agent must stop at `awaiting_human_naturalization_review`. It may not fill the ratings or claim
Gate A.
