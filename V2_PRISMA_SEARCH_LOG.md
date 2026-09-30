# V-2 — Systematic Literature Search Log (PRISMA template)

**Status:** TEMPLATE — search has NOT been executed by the agent.
**Gate:** Spec §24, §692 (blocks any novelty wording).
**Owner:** Human operator executes the search; agent may assist with search-engine queries, dedup, and inclusion checks.
**Last template revision:** 2026-09-04.

Per spec §692: "No novelty statement should use the phrases 'first,' 'only,' 'no prior
work,' or 'absent from the literature' until V-2 is completed. After V-2, any 'first'
claim must be tied to a documented search scope and dated search log."

---

## 1. Research question

> "Is there prior published work that formulates, for an autism-screening instrument
> (Q-CHAT-10, Q-CHAT-25, AQ-10, ADI-R, ADOS-2, M-CHAT, or equivalent), a sequential
> adaptive item-acquisition policy under an explicit finite question-count budget B,
> with a proper-scoring-rule terminal utility (Brier / log-loss / proper score) and a
> comparison against an exact backward-induction decision-theoretic reference over the
> empirical training support?"

**Sub-questions (positioning checks):**
- S1. CAT-Autism (Computerized Adaptive Testing for autism).
- S2. 2025 Q-CHAT compact-subset work (Sollis, Wall & Washington 2025; *Sci. Rep.* 15:39091).
- S3. Sequential feature-acquisition RL on autism or screening instruments.
- S4. Exact decision-tree / exact DP methods for adaptive testing (DL8.5, MurTree, STreeD, ID3-on-DP).
- S5. Item-response-theory (IRT) and item-information greedy policies applied to autism screening.
- S6. Q-CHAT-10 representation / binarisation rules (Sollis et al. mapping, §15).

## 2. Databases / search engines

| # | Database / Engine | Interface | Coverage | Access |
|---|---|---|---|---|
| 1 | PubMed | NLM eutils API + browser | biomedicine, psychology | open |
| 2 | PsycINFO (EBSCOhost) | EBSCO API | psychology, psychiatry | institutional |
| 3 | Web of Science Core Collection | Clarivate | multidisciplinary citations | institutional |
| 4 | IEEE Xplore | IEEE | engineering, ML applications | institutional |
| 5 | ACM Digital Library | ACM | computer science | open / institutional |
| 6 | arXiv (cs.LG, cs.AI, stat.ML) | arXiv API | preprints | open |
| 7 | Google Scholar | Google Scholar | broad, citation graph | open |
| 8 | Semantic Scholar | API | AI/ML + biomedicine with S2 API | open |
| 9 | DBLP | dblp.org | CS, esp. ML | open |
| 10 | ClinicalTrials.gov | CT.gov API | registered trials | open |

Hand-search: reference lists of all included full-text articles; forward-citation
through Google Scholar; check author homepages of Thabtah, Sollis, Allison, and
Wall for self-citation gaps.

## 3. Search strings (exact)

Each row is **one query**, executed as a single string in the database's default
field (title + abstract + keywords where supported). Strings below are illustrative;
record the EXACT string used (with field tags if the database requires them).

| # | Query (use exactly) | Database |
|---|---|---|
| Q1 | `"Q-CHAT" AND ("adaptive" OR "sequential" OR "computerized adaptive testing")` | PubMed, PsycINFO, WoS, Scholar |
| Q2 | `"autism screening" AND ("item selection" OR "item acquisition" OR "information gain" OR "Shannon" OR "entropy")` | all |
| Q3 | `"autism" AND ("Q-CHAT-10" OR "Q-CHAT-25") AND ("machine learning" OR "reinforcement learning" OR "decision tree")` | all |
| Q4 | `"backward induction" AND ("assessment" OR "screening") AND ("item" OR "question")` | all |
| Q5 | `"Brier" AND ("adaptive testing" OR "sequential" OR "policy") AND ("screening" OR "diagnosis")` | all |
| Q6 | `"exact dynamic programming" AND ("decision tree" OR "OR tree" OR "AND/OR graph") AND ("psychometrics" OR "assessment")` | all |
| Q7 | `("Q-CHAT" OR "M-CHAT" OR "AQ-10") AND ("item response theory" OR "IRT" OR "Graded Response") AND ("adaptive" OR "personalized")` | all |
| Q8 | `"reinforcement learning" AND ("adaptive testing" OR "question selection" OR "item selection")` | all |
| Q9 | `"Computerized Adaptive Testing" AND ("autism" OR "ASD")` | all |
| Q10 | `"Thabtah" AND ("autism" OR "Q-CHAT")` | author search |
| Q11 | `"Sollis" AND "Q-CHAT"` | author search |
| Q12 | `("DL8.5" OR "MurTree" OR "STreeD") AND ("adaptive testing" OR "screening" OR "assessment")` | IEEE, ACM, Scholar |

**Date range:** 2000-01-01 through the search-execution date (no closing date yet —
record the date on the worksheet).

**Language filter:** None (do not exclude non-English; machine-translate in
screening).

**Document types:** all (peer-reviewed, conference, preprint, thesis, report).

## 4. Inclusion / exclusion criteria

### Inclusion (must satisfy ALL)
- I1. Targets an autism-screening instrument OR a sequential assessment/IRT/CAT framework comparable in structure.
- I2. Specifies or applies a sequential item-acquisition policy under a question-count or time budget.
- I3. Defines a terminal utility (or terminal objective) and compares it against a non-trivial reference (static subset, alternative policy, or exact method).
- I4. Reports reproducible metric(s) (Brier, AUROC, accuracy, sensitivity/specificity, decision-utility, or comparable).

### Exclusion (any one is sufficient)
- E1. Pure static feature selection (no sequential acquisition, no budget) — log under "background only" not "in-scope".
- E2. Pure instrument-validation study (psychometric, factor analysis) with no acquisition policy.
- E3. Imaging / EEG / genetic / wearable-only modalities (out of project scope per §15 — facial images, neuroimaging).
- E4. Diagnostic (post-assessment) classification only, no item-acquisition policy.
- E5. Duplicate of an already-included record (resolved by DOI/title match).

### Tie-break rules
- T1. If a paper applies a method to a non-autism domain but the method is **directly applicable** to Q-CHAT-10 (e.g., CAT framework, exact DP for sequential assessment), include under S3/S4 and note the transfer in the worksheet.
- T2. If a paper claims a result that is contradicted by a later publication, include both and record the contradiction in the worksheet.
- T3. If a paper is a preprint with a published version, prefer the published version but keep the preprint as a duplicate record.

## 5. PRISMA flow

```
Identification
  Records identified through database searching (Q1..Q12 across §2)
        = N_iden
  Records identified through hand-search / citation chasing
        = N_hand
  Duplicates removed (DOI + title match)
        = N_dedup
  Records after deduplication
        = N_screened

Screening
  Records screened (title + abstract)
        = N_screened
  Records excluded at screening
        = N_screened_excl
  Full-text articles assessed for eligibility
        = N_fulltext
  Full-text articles excluded, with reasons
        = N_fulltext_excl
  (Reasons: E1 static-only; E2 validation-only; E3 modality; E4 no policy; E5 dup)

Included
  Studies included in qualitative synthesis
        = N_incl_qual
  Studies included in quantitative comparison
        = N_incl_quant
  Studies contributing to novelty claim
        = N_novelty
```

Fill in all N_ values when search executes. The "novelty claim" tier is
narrowest: a study only contributes to novelty if it is directly comparable
to **our exact finite-budget objective formulation** with a Brier-based
terminal utility and exact decision-theoretic reference.

## 6. Screening worksheet schema

One row per record. Stored as CSV under `docs/prisma/screening_worksheet.csv`
(search execution populates it). Columns:

```
record_id         stable internal id (e.g., P0001)
title             exact title from database
authors           "First, Second, Third" semicolon-separated
year             publication year
venue            journal / conference / preprint server
doi              DOI (or arXiv id, or URL)
database_source  which of the 10 databases in §2 produced the hit
query_id         which of Q1..Q12 (or "hand")
search_date      YYYY-MM-DD
screening_stage  identification | screening | eligibility | included
decision         include | exclude | background | duplicate
exclusion_reason one of E1..E5 if exclude; empty otherwise
rationale        one-sentence justification (≤ 200 chars)
notes            free-form, optional
contributes_to   one or more of S1..S6 (semicolon-separated)
novelty_anchor   true if the record is in the novelty-claim tier (N_novelty)
```

The agent populates only the **schema and one example row**; the human operator
executes the search and fills the rest.

## 7. Required outputs (after search execution)

- `docs/prisma/PRISMA_flow_<YYYY-MM-DD>.png` — flow diagram
- `docs/prisma/PRISMA_flow_<YYYY-MM-DD>.md` — same in markdown
- `docs/prisma/screening_worksheet.csv` — full row-level data
- `docs/prisma/included_studies.csv` — included-only with full citation
- `docs/prisma/positioning_summary.md` — narrative comparing this project to each included study, especially S1..S4
- **Append to repo** `AGENT_PROGRESS.md`: V-2 status, search date, N_incl_qual, N_novelty, signature.

## 8. No-claim rule (locked in)

Until V-2 is completed, the following phrases are FORBIDDEN in any artifact under
this repo, in any commit message, or in any documentation:

- "first"
- "only"
- "no prior work"
- "absent from the literature"
- "to our knowledge" (when qualifying novelty)

Replacements while V-2 is open:

- "first" → "preliminary" or "in this implementation"
- "only" → rephrase
- "no prior work" → "no prior work has been identified by the V-2 search as of <date>"
- "absent from the literature" → rephrase
- "to our knowledge" → "to the best of the V-2 search's finding as of <date>"

This rule applies to README, code docstrings, results/, and any markdown under
`docs/`. The agent is configured to reject such wording on review.

### 2026-10-01 note — practical effect of the rule

The rule is enforced mechanically by `tests/test_v2_no_claim_rule.py`, which scans every
non-allowlisted markdown file in the repo. Because the trigger is a bare word match
(`\bonly\b`, `\bfirst\b`), it fires on ordinary exclusivity usage, not just novelty claims.
When writing new documents, expect to rephrase rather than add to the allowlist:

- `only` → "the single", "solely", or restructure the sentence
- `first` → "initial", "preliminary", "in this implementation"

Real examples of this from the 2026-10-01 pass, for reference:
- "the first real training runs" → "initial real training runs"
- "the only cohort" → "the single cohort"
- "documentation only" → "documented, not built"

Note the test also skips lines containing audit/policy tokens (`TEST`, `AUDIT`, `PENDING`,
`SCREENING`, …) and skips fenced code blocks, so some legitimate uses pass automatically.
This is a blunt instrument by design: it is cheaper to rephrase than to adjudicate whether a
given usage constitutes a novelty claim.

**The rule remains in force. No novelty claim is permitted until V-2 is completed.**

## 9. Example row (synthetic, for template validation only)

The CSV in `docs/prisma/screening_worksheet.csv` carries this exact one-row header
plus the synthetic example below. The full sheet is **populated by the human
operator**; the agent provides only the schema and a single illustrative example.

```
record_id,title,authors,year,venue,doi,database_source,query_id,search_date,screening_stage,decision,exclusion_reason,rationale,notes,contributes_to,novelty_anchor
P0001,Q-CHAT-10 fixed subset for autism screening,Sollis; Wall; Washington,2025,Scientific Reports 15:39091,10.1038/s41598-025-39091-3,Scholar,Q1,2026-XX-XX,included,include,,Four-item Saudi-trained fixed subset; AUROC 87 +/- 11 at τ=0.3,Polish external evaluation 252,Four-item fixed subset; sequential adaptive policy is not the contribution; static baseline reference,S2,true
```
