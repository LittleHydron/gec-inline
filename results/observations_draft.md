# Qualitative observations — draft (JFLEG dev, BEA-dev numbers pending)

## Format learning (base → SFT)

- **Base zero-shot fails the task wholesale**: 27.5 % parse-failure rate;
  it answers in free-form prose, restates the sentence, or wraps the
  whole output in one giant brace pair. Exact-match 0.5 % — effectively
  zero. This confirms the format is not in the base model's prior.
- **3-shot prompting fixes the format but not the corrections**:
  parse failures drop to 3.8 %, but every exact match (13.3 %) is a
  trivial copy — the few-shot base scores *only* on sentences that were
  already correct. It mimics the bracket syntax while contributing no
  actual correction quality (and litters 0.35 no-op edits/sentence,
  e.g. `{I=>I}` — copying the few-shot examples' surface pattern).
- **SFT learns both format and task**: parse failures ≈ 0 (0.4 %),
  exact-match 29.3 % (2.2× the base+3-shot), and — unlike the base —
  most of its exact matches come from *actual corrections*, not copies
  (trivial-copy 12.9 % < exact-match 29.3 %). No-op edits all but
  vanish (0.08/sentence). Errors that remain are mostly missed
  insertions of short closed-class words ("the", "a", commas) — the
  model is confident but conservative, as expected from the literature.

## DPO negative result (first run) — preference-data bias

The first DPO run **degraded** the model dramatically: JFLEG exact-match
collapsed 29.3 % → 2.1 %, parse failures rose to 9.3 %, and edit density
exploded from 2.79 to **11.22 edits/sentence** with 2.11 *no-op* edits
(`{could=>could}`, `{sex=>sex}`) per sentence — plus gratuitous
paraphrasing of already-correct text (`{purchase=>buy}`,
`{consumers=>customers}`).

Root cause — three compounding biases in the preference-pair builder,
all pushing the same direction ("more edits ⇒ preferred"):

1. Already-correct sentences were skipped, so `chosen` always contained
   at least one edit; an edit-free output was never shown as preferable.
2. The programmatic corruption only ever *removed* an edit from gold,
   so `rejected` systematically had fewer edits than `chosen`.
3. Cross-sentence rejections (gold of a different sentence) carried the
   same fewer-relevant-edits signal.

DPO optimized exactly the proxy it was given: brace density. A clean
demonstration that preference optimization amplifies dataset bias far
more aggressively than SFT does — SFT on the same gold data never
developed this pathology because its targets include unedited spans.

[IF RETRAINED: results of the fixed second DPO run — symmetric
corruptions (add ∨ remove edits), ~30 % already-correct chosen pairs —
go here, with before/after table.]

## Misc

- Base+3-shot sometimes "corrects" proper nouns and dialect spellings
  it shouldn't touch; SFT learned BEA's conservative editing norms.
- The bracket format survives batched greedy decoding fine at 3 B scale;
  parse failures in the SFT model are mostly truncations of very long
  sentences (max_new_tokens), not malformed braces.
- JFLEG (fluency-oriented, 4 refs) systematically under-rewards the
  BEA-trained models: they make minimal grammatical edits where JFLEG
  references rewrite for fluency. ERRANT F0.5 on BEA dev is the metric
  that matches the training distribution; JFLEG exact-match is a
  format-independent sanity check, not the headline number.
