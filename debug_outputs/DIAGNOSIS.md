# RG1-P Stage A analysis serialization diagnosis

## Symptom

Formal generation completed at 864/864 and the blocking Formal audit passed, but
`scripts/analyze_rg1p_stagea.py` stopped while writing `ANALYSIS.json` with
`TypeError: Object of type bool is not JSON serializable`.

## Root cause

`pca.explained_variance_ratio_` is a NumPy array. Summing it yields a NumPy
scalar, and comparing that scalar with `0.1` yields `numpy.bool_`. Its class name
is displayed as `bool`, but the standard JSON encoder accepts only the native
Python `bool`.

## Scientific boundary

The failure occurred after all GPU records and the Formal audit were complete.
The fix changes only the runtime type of one descriptive output field. It does
not change records, tensors, bootstrap draws, statistics, route classification,
