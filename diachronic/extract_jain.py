"""
extract_jain.py
===============
Extracts philosophical concept edges from Jain canonical PDFs.
Filters for English-language chunks (skips Devanagari/Prakrit-only pages).
Output: jain_edges.jsonl

Run from agentic-diffusion-sim directory:
    cp /path/to/jain/pdfs/*.pdf .
    python3 extract_jain.py
    python3 extract_jain.py --pdf Tattvarthasutra.pdf --limit 30
"""

import json, os, argparse, time, re
from pathlib import Path

GROQ_MODEL = "llama-3.3-70b-versatile"

FILE_TO_SCHOOL = {
    "tattva":     "jainism",
    "tattv":      "jainism",
    "umasv":      "jainism",
    "samaya":     "jainism",
    "acar":       "jainism",
    "ayaro":      "jainism",
    "agama":      "jainism",
    "aagam":      "jainism",
    "jain":       "jainism",
}

EXTRACTION_PROMPT = """You are extracting philosophical concept relationships from a Jain canonical text.

Text passage (Jain philosophy):
{passage}

Extract up to 8 philosophical concept pairs. For each:
- concept_a: first concept (use standard Jain/Sanskrit term where possible)
- concept_b: second concept
- relation: one of [IS_IDENTICAL_TO, IS_DISTINCT_FROM, IS_CAUSE_OF, LEADS_TO,
  PRESUPPOSES, SUBLATES, OBSTRUCTS, IS_MANIFESTATION_OF, IS_QUALIFIED_ASPECT_OF,
  CONTRADICTS_IN_SCHOOL]
- evidence_quote: exact phrase showing this relationship (max 100 chars)
- confidence: high/medium/low

Focus on Jain philosophical concepts:
ONTOLOGY: jiva (soul/living), ajiva (non-living), pudgala (matter), dharma (medium of motion),
adharma (medium of rest), akasha (space), kala (time), dravya (substance), tattva (reality)

SOTERIOLOGY: asrava (karma influx), bandha (bondage), samvara (stoppage), nirjara (shedding),
moksha (liberation), karma (karmic matter), kasaya (passion), lesya (soul-colouration)

ETHICS: ahimsa (non-violence), satya (truth), asteya (non-stealing), brahmacharya (celibacy),
aparigraha (non-possession), tapas (austerity), samyak darshana (right faith),
samyak jnana (right knowledge), samyak charitra (right conduct) — the three jewels

METAPHYSICS: anekantavada (many-sidedness), syadvada (conditional predication),
nayavada (doctrine of standpoints), nishchaya naya (absolute standpoint),
vyavahara naya (practical standpoint)

Return ONLY a JSON array. If no philosophical pairs found return [].
Example:
[
  {{"concept_a": "jiva", "concept_b": "ajiva",
    "relation": "IS_DISTINCT_FROM",
    "evidence_quote": "the soul and matter are fundamentally distinct",
    "confidence": "high"}}
]"""

def is_english_chunk(text, threshold=0.6):
    """Return True if chunk is mostly ASCII/English, not Devanagari."""
    if not text.strip():
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_chars / len(text)
    return ratio > threshold

def read_pdf_english(path, skip_pages=0):
    """Extract English-only text chunks from PDF."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        if i < skip_pages:
            continue
        text = page.extract_text() or ""
        if is_english_chunk(text):
            pages.append(text)
    return pages

def chunk_text(text, chunk_size=500, overlap=60):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def extract_from_chunk(chunk, groq_client):
    prompt = EXTRACTION_PROMPT.format(passage=chunk[:2000])
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=800, temperature=0.1)
        text = r.choices[0].message.content.strip()
        text = re.sub(r"```json|```","",text).strip()
        edges = json.loads(text)
        return edges if isinstance(edges,list) else []
    except json.JSONDecodeError:
        return []
    except Exception as e:
        print(f"    Groq error: {e}")
        time.sleep(2)
        return []

def get_school(filename):
    fn = filename.lower()
    for key, school in FILE_TO_SCHOOL.items():
        if key in fn:
            return school
    return "jainism"

def process_pdf(path, groq_client, limit=None, skip_pages=5):
    filename = Path(path).stem
    school   = get_school(filename)
    print(f"\n  [{filename}] -> school: {school}")

    pages = read_pdf_english(path, skip_pages=skip_pages)
    print(f"  English pages found: {len(pages)}")

    if not pages:
        print(f"  No English text — skipping")
        return []

    full_text = "\n".join(pages)
    chunks = chunk_text(full_text)
    print(f"  Chunks: {len(chunks)}")
    if limit:
        chunks = chunks[:limit]
        print(f"  (limited to {limit} chunks)")

    all_edges = []
    for i, chunk in enumerate(chunks):
        edges = extract_from_chunk(chunk, groq_client)
        for e in edges:
            e["school"]    = school
            e["source"]    = filename
            e["chunk_id"]  = i
            e["concept_a"] = str(e.get("concept_a","")).strip().lower()
            e["concept_b"] = str(e.get("concept_b","")).strip().lower()
        edges = [e for e in edges
                 if e["concept_a"] and e["concept_b"]
                 and e["concept_a"] != e["concept_b"]
                 and len(e["concept_a"]) > 2
                 and len(e["concept_b"]) > 2]
        all_edges.extend(edges)
        if (i+1) % 10 == 0:
            print(f"  Chunk {i+1}/{len(chunks)} — edges: {len(all_edges)}")
        time.sleep(0.3)

    print(f"  Done: {len(all_edges)} edges")
    return all_edges

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",    default=None)
    parser.add_argument("--dir",    default=".")
    parser.add_argument("--output", default="jain_edges.jsonl")
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--skip_pages", type=int, default=5)
    args = parser.parse_args()

    key = os.environ.get("GROQ_API_KEY","")
    if not key:
        print("ERROR: set GROQ_API_KEY"); return
    from groq import Groq
    client = Groq(api_key=key)
    print("Groq ready | school=jainism")

    if args.pdf:
        files = [args.pdf]
    else:
        d = Path(args.dir)
        files = sorted(d.glob("*.pdf"))

    # Priority order: Tattvarthasutra first, then Samayasara, then Acaranga
    priority = ["tattva","umasv","samaya","acar","ayaro","agama","aagam"]
    def sort_key(p):
        fn = Path(p).stem.lower()
        for i, k in enumerate(priority):
            if k in fn: return i
        return 99
    files = sorted(files, key=sort_key)

    print(f"\nFiles to process ({len(files)}):")
    for f in files:
        print(f"  {Path(f).name}")

    # Resume support
    existing = []
    existing_sources = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    existing.append(e)
                    existing_sources.add(e.get("source",""))
        print(f"\nResuming: {len(existing)} edges already done")

    all_edges = list(existing)
    with open(args.output, "a") as out_f:
        for path in files:
            source = Path(path).stem
            if source in existing_sources:
                print(f"  SKIP {source} (done)")
                continue
            edges = process_pdf(
                path, client,
                limit=args.limit,
                skip_pages=args.skip_pages)
            for e in edges:
                out_f.write(json.dumps(e) + "\n")
            out_f.flush()
            all_edges.extend(edges)

    # Summary
    from collections import Counter
    print(f"\n{'='*55}")
    print(f"Total Jain edges: {len(all_edges)}")
    concept_counts = Counter()
    for e in all_edges:
        concept_counts[e["concept_a"]] += 1
        concept_counts[e["concept_b"]] += 1
    print(f"\nTop 30 Jain concepts extracted:")
    for c, n in concept_counts.most_common(30):
        print(f"  {c:<35}: {n:>4}")

if __name__ == "__main__":
    main()
