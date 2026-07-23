# Engineering Failure Log

## Pre-delivery failures caught locally

1. The first readable build inherited corrupted quote glyphs from an imported v1.1 renderer
   function. Formal gates passed, but manual surface inspection caught `ĄŽ...ĄŻ`. Version 1.2 now
   renders record edits locally with Unicode quotation marks and hard-fails on known encoding
   artifacts.
2. The first medium rendering used mechanical `Review stage 1/2/3` text. Before human review it was
   replaced globally by a shorter semantic chain shared by V and F; no symbolic source or solver
   result changed.
3. The independent workbook validator initially opened the workbook in read-only mode and could
   not inspect hidden-column metadata. It now opens the blank packet normally and remains
   non-mutating.

No failed formal item was dropped, no seed was replaced, and no model output was consulted.
