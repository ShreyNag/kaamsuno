# KaamSuno Research Pack v1.1

## Files
- `research/schema.json`: report fields, required fields, enums, and safety/validation rules.
- `research/fillers.json`: 1178 phrase candidates for hesitation, continuation, uncertainty, correction, confirmation, regional-influenced speech, and spelling/script variants.
- `research/number_pairs.csv`: 1800 number/unit/ID cases. Includes cardinal numbers, quantities, spoken digit IDs, ambiguous contrasts, dates/times, decimals, fractions, ranges and units.
- `research/dialogue_scripts.json`: 92 candidate dialogue scripts for missing fields, clarification, read-back, confirmation, correction, emergency and fallback.
- `research/scenarios.jsonl`: 1000 scenario-level tests, with development/holdout split labels.

## Important interpretation
This is a broad synthetic starter/evaluation pack, not a claim that every real-world dialect, filler, asset name or site scenario has been covered. A generated phrase/test is a test prompt, not proof that people naturally say it. Review regional phrases with speakers from the target sites and replace synthetic prompts with consented real recordings before making performance claims.

## Integration rules
1. Treat phrases as contextual cues, not deterministic labels.
2. Never guess critical IDs, quantities, units, location, cause or urgency.
3. Preserve leading zeros and alphanumeric prefixes in IDs.
4. Keep the original spoken number/phrase when normalization may be ambiguous.
5. Ask one targeted clarification at a time.
6. Do not submit routine reports until read-back is explicitly confirmed.
7. Follow a site-approved emergency escalation policy for imminent hazards; do not wait for perfect metadata.
8. Holdout cases should not be used to tune prompts, thresholds or model settings.

## Splits and evaluation
`scenarios.jsonl` marks every fifth scenario as `holdout` and the rest as `development`. This is a simple reproducible split, not a guarantee of statistical representativeness. For a stronger evaluation, split by speaker/recording session and avoid near-duplicate variants crossing splits.

## Metrics to report
- Field exact-match accuracy for incident type, location, asset ID, quantity, unit and urgency.
- Critical-slot error rate, especially asset IDs and quantities.
- Clarification precision: whether a clarification was actually needed.
- Unnecessary clarification rate.
- Correct turn-end detection / false end-of-turn rate.
- Confirmation compliance: routine reports submitted only after explicit confirmation.
- Emergency escalation recall on the labelled hazard set.
- End-to-end latency and failure rate.
- Results separately for Hindi, Hinglish, English, regional-influenced speech and noisy audio.

## Before production
- Confirm schema names with the code owner.
- Replace generic site terms with the real site's approved asset/material glossary.
- Review local-language entries with native speakers.
- Use consented, licensed audio and retain its source/licence information.
- Keep personal identifiers out of test data unless strictly necessary.
