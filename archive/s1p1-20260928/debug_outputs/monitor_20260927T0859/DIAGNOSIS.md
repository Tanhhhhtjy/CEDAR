# Monitoring failure diagnosis

The 2026-09-27 08:59 UTC inspection failed in monitor_snapshot.py while reading a blob mtime. The downloader completed and renamed a .incomplete shard after the monitor had collected paths and sizes. A later stat() of the old name raised FileNotFoundError. This was a monitoring race; the downloader and campaign remained active.

The original monitor was preserved as monitor_snapshot_before.py. This engineering repair does not change inference, inputs, model precision, numerical gates, or analysis. The skill's optional research-pitfall-checklist.md reference was absent; the available operating principles, research rigor principles, and debug policy were read.
