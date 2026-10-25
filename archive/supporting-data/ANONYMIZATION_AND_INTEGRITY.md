# Anonymization and integrity

The archive is generated from frozen research blocks.  Structured text is
parsed where possible, project version-control fields and private machine data
are removed, local paths and reviewer identities are anonymized, and workbook
metadata is reset.  Public third-party model, tokenizer, and intervention-asset
revision identifiers are retained because they are required for replication
and do not identify the submission authors.

`SANITIZATION_LEDGER.csv` binds each source file hash to the delivered file
hash.  `MANIFEST.csv` is the authoritative integrity manifest for the anonymous
archive.  Internal scientific manifests describe frozen protocols and are not
used as delivery manifests after anonymization.
