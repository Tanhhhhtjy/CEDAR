# RG1-P Stage A analysis serialization patch plan

1. Cast `move_to_appendix_if_low_variance` to native Python `bool` at its source.
2. Leave the shared JSON writer unchanged to avoid silently coercing unrelated
   future analysis values.
3. Run Ruff and the RG1-P CPU test suite.
4. Re-run only analysis, plotting, and final deliverable verification.
5. Do not re-run Smoke or any of the 864 Formal GPU records.
6. Record the analysis patch commit separately from the frozen execution commit.

This is an engineering compatibility repair, not a scientific contribution or
protocol amendment.
