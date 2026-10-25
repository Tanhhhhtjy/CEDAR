# -*- coding: utf-8 -*-
"""Build references.bib for the CEDAR AAAI-27 submission from verified arXiv metadata.

Usage:  python make_bib.py            -> writes references_arxiv.bib + fetch_report.json
Nothing is hand-typed: every field comes from the arXiv API. Entries that are NOT on
arXiv are listed in TODO_NON_ARXIV and must be exported from ACL Anthology / DBLP.
"""
import json, re, sys, time, urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}

# (arxiv_id, suggested_key, why_we_cite_it, tier)
ITEMS = [
    # ---- tier 1: must cite (core object of study + nearest neighbours) ----
    ("2601.10387", "assistantaxis",      "the intervention we audit (Assistant Axis + capping)", 1),
    ("2606.11205", "buchan2026dual",     "nearest neighbour: readable-but-not-writable sycophancy steering", 1),
    ("2508.02087", "whentruth2026",      "sycophancy has deep representational origins; late-layer patching", 1),
    ("2312.06681", "caa2023",            "contrastive activation addition = the additive-steering baseline", 1),
    ("2406.11717", "refusaldirection",   "single-direction mediation; canonical directional-ablation method", 1),
    ("2507.21509", "personavectors",     "persona traits as linear directions", 1),
    ("2605.21006", "devilsadvocate2026", "per-role vectors vs sycophancy direction (orthogonality claim we bound)", 1),
    ("2509.21305", "sycophancynotone",   "sycophancy is not a single construct -> motivates V/F/AO/S split", 1),
    ("2605.27288", "muse2026",           "confidence-based separation of sycophantic vs rational yielding", 1),
    # ---- tier 2: should cite (measurement, selectivity, geometry) ----
    ("2409.05907", "cast2024",           "conditional activation steering: gating precedent for L1/L2 discussion", 2),
    ("2509.16820", "disco2025",          "selective steering with off-target preservation", 2),
    ("2602.15293", "dualsteering2026",   "information-geometric constraint on side effects", 2),
    ("2602.02315", "shapeofbeliefs",     "beliefs live on curved low-dim manifolds", 2),
    ("2604.02685", "beliefgeomsae",      "belief geometry via SAEs", 2),
    ("2603.14893", "llmsignaldetectors", "SDT framing for LLM decisions (d' / criterion)", 2),
    ("2303.08112", "tunedlens",          "layerwise readout tooling (methods / limitations)", 2),
    ("2305.08809", "boundlessdas",       "subspace causal analysis; why we did not claim circuit-level results", 2),
    ("2306.03341", "iti2023",            "inference-time intervention: the other canonical activation edit", 2),
    ("2310.13548", "sharma2023sycophancy", "sycophancy is systematic across RLHF'd assistants", 2),
    ("2212.09251", "perez2022evals",     "model-written evals; source of standard sycophancy probes", 2),
    # ---- tier 3: optional / appendix ----
    ("2408.05147", "gemmascope",         "SAE assets (future-work paragraph only)", 3),
    ("2406.15927", "sep2024",            "semantic entropy probes (uncertainty moderator, appendix)", 3),
    ("2402.10962", "instructioninstability", "drift grows with context length (motivation sentence)", 3),
    ("2605.10664", "gcad2026",           "attention-level gated intervention (future work)", 3),
]

TODO_NON_ARXIV = [
    ("Belief-R", "EMNLP 2024", "ACL Anthology", "why our formal domain is NOT Belief-R (methods footnote)"),
    ("SycoBench-600", "Findings of ACL 2026", "ACL Anthology", "source corpus for CEDAR-Social"),
    ("LOGICOM", "LREC-COLING 2024", "ACL Anthology", "external fallacy-persuasion benchmark (limitations)"),
    ("LoFa", "ACL 2026", "ACL Anthology", "external fallacy set we did not yet run (future work)"),
    ("HarmBench", "ICML 2024", "PMLR / DBLP", "harmful-behaviour benchmark used in the safety-leg attempt"),
    ("JailbreakBench", "NeurIPS 2024 D&B", "DBLP", "same"),
    ("WMDP", "ICML 2024", "PMLR / DBLP", "capability-unlearning benchmark (related work)"),
    ("GlobalOpinionQA", "Anthropic 2023", "arXiv 2306.16388", "value-domain topic pool (appendix only)"),
    ("Qwen3 technical report", "2025", "arXiv", "model card / reproducibility"),
    ("Prover9 / Mace4", "McCune, 2005-2010", "software URL", "solver certification of V/F and countermodels"),
    ("Jacobian Lens / global workspace", "Anthropic, 2026", "transformer-circuits.pub", "tool we explicitly did NOT use (limitations)"),
    ("Lakens, equivalence testing (TOST)", "SPPS 2017", "DOI", "ROPE / equivalence wording"),
    ("Macmillan & Creelman, Detection Theory", "book, 2005", "ISBN", "d' and criterion definitions"),
]


def fetch(arxiv_id, retries=3):
    url = "http://export.arxiv.org/api/query?id_list=" + arxiv_id
    for k in range(retries):
        try:
            raw = urllib.request.urlopen(url, timeout=25).read().decode("utf-8")
            entry = ET.fromstring(raw).find("a:entry", NS)
            if entry is None or entry.find("a:title", NS) is None:
                raise ValueError("no entry")
            title = " ".join(entry.find("a:title", NS).text.split())
            authors = [a.find("a:name", NS).text for a in entry.findall("a:author", NS)]
            published = entry.find("a:published", NS).text[:4]
            journal = entry.find("a:journal_ref", NS)
            doi = entry.find("a:doi", NS)
            comment = entry.find("{http://arxiv.org/schemas/atom}comment")
            primary = entry.find("{http://arxiv.org/schemas/atom}primary_category")
            return {
                "id": arxiv_id, "title": title, "authors": authors, "year": published,
                "journal_ref": journal.text.strip() if journal is not None else None,
                "doi": doi.text.strip() if doi is not None else None,
                "comment": " ".join(comment.text.split()) if comment is not None else None,
                "primary_class": primary.get("term") if primary is not None else "cs.CL",
            }
        except Exception as exc:
            if k == retries - 1:
                return {"id": arxiv_id, "error": str(exc)}
            time.sleep(3)


def bibtex(meta, key, note):
    authors = " and ".join(meta["authors"])
    title = meta["title"]
    # protect capitalised tokens so the AAAI .bst does not lowercase them
    title = re.sub(r"(?<![{\w])([A-Z][A-Za-z0-9\-]*[A-Z][A-Za-z0-9\-]*)", r"{\1}", title)
    lines = [
        "@misc{%s," % key,
        "  title         = {%s}," % title,
        "  author        = {%s}," % authors,
        "  year          = {%s}," % meta["year"],
        "  eprint        = {%s}," % meta["id"],
        "  archivePrefix = {arXiv},",
        "  primaryClass  = {%s}," % meta["primary_class"],
    ]
    if meta.get("doi"):
        lines.append("  doi           = {%s}," % meta["doi"])
    if meta.get("journal_ref"):
        lines.append("  note          = {%s}," % meta["journal_ref"])
    lines.append("}")
    head = "%% [%s] %s\n" % (key, note)
    if meta.get("journal_ref"):
        head += "%% PUBLISHED VERSION EXISTS -> replace with the proceedings entry: %s\n" % meta["journal_ref"]
    return head + "\n".join(lines) + "\n"


def main():
    out, report = [], []
    out.append("%% CEDAR AAAI-27 references -- arXiv entries auto-generated from the arXiv API.\n"
               "%% Every entry below was fetched, not typed. Entries whose journal_ref is set\n"
               "%% MUST be replaced by the peer-reviewed proceedings entry before submission.\n")
    for aid, key, note, tier in ITEMS:
        meta = fetch(aid)
        if "error" in meta:
            report.append({"id": aid, "key": key, "tier": tier, "status": "FAILED", "error": meta["error"]})
            out.append("%% !! FETCH FAILED for %s (%s) -- resolve manually\n" % (aid, key))
            continue
        report.append({"id": aid, "key": key, "tier": tier, "status": "ok",
                       "title": meta["title"], "year": meta["year"],
                       "journal_ref": meta["journal_ref"], "n_authors": len(meta["authors"])})
        out.append(bibtex(meta, key, note))
        time.sleep(1.0)  # be polite to the API

    out.append("\n%% ===================== NOT ON arXiv -- EXPORT MANUALLY =====================\n")
    for name, venue, where, why in TODO_NON_ARXIV:
        out.append("%% TODO  %-34s %-24s source: %-22s (%s)\n" % (name, venue, where, why))

    open("references_arxiv.bib", "w", encoding="utf-8").write("\n".join(out))
    json.dump(report, open("fetch_report.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    ok = sum(1 for r in report if r["status"] == "ok")
    print("fetched %d/%d arXiv entries; %d manual TODOs" % (ok, len(ITEMS), len(TODO_NON_ARXIV)))
    for r in report:
        if r["status"] == "ok" and r["journal_ref"]:
            print("  NOTE published version exists:", r["key"], "->", r["journal_ref"])
        if r["status"] != "ok":
            print("  FAILED:", r["id"], r["key"])


if __name__ == "__main__":
    main()