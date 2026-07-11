# Data provenance

Sample identifiers, derived split manifests, and small project-authored development screens are
committed here. Raw upstream datasets are fetched from their pinned revisions at runtime.

Belief-R is licensed under CC-BY-SA-4.0. See `docs/M2_PROTOCOL.md` for the pinned revision and the
required separation between exploratory M2 samples and future M5 holdouts.

`development/m2_objective_correction_screen_v1.json` is a project-authored 20-item diagnostic for
the failed Belief-R/Qwen baseline. Its four conversation variants per item are development-only and
must not be reused as confirmatory Pilot items.

`development/m2_objective_correction_screen_v2_hard.json` is the replacement development screen
triggered when v1's wrong-prefill/no-evidence exclusion rate exceeded 50%. It uses plausible omitted
steps in arithmetic, unit conversion, and multi-step logic and is likewise excluded from Pilot.
