# Adversarial Motions Exercise — Husidic v. FR8 Solutions, Inc.

A three-round simulated adversarial drafting exercise based on the filed Motion for
Sanctions (Doc. 145) and Response in Opposition (Doc. 151) in *Husidic v. FR8
Solutions, Inc.*, No. 3:24-cv-00963 (M.D. Fla.). Two AI advocate agents (Plaintiffs'
counsel and Defendants' counsel) iteratively revised their briefs; a neutral AI judge
agent gave brainstorming feedback after Rounds 1 and 2 and issued a reasoned final
order after Round 3.

**All documents are simulation drafts — not legal advice and not for filing.**

## Final deliverables

| Document | Path |
|---|---|
| Plaintiffs' final motion for sanctions | `final/final_motion_for_sanctions.md` |
| Defendants' final response (with proposed order, App. A) | `final/final_response_to_motion.md` |
| Judge's final order + closing commentary | `final/final_order.md` |
| Plaintiffs' counsel change log & reflections (Rounds 1–3) | `logs/plaintiff_log.md` |
| Defendants' counsel change log & reflections (Rounds 1–3) | `logs/defense_log.md` |

## Round-by-round artifacts

- `source/` — text extracted from the two filed PDFs
- `round1/` — motion v1, response v1, judge's Round 1 feedback
- `round2/` — motion v2, response v2, judge's Round 2 (final pre-submission) feedback
- `round3/` — (final drafts live in `final/`)

## Citation discipline

Agents were restricted to authorities appearing in the two filed documents; anything
added from model knowledge had to be tagged `[VERIFY]`. Final drafts resolved all
`[VERIFY]` tags by deletion rather than silent laundering. Verify every citation
independently before any real-world use.
