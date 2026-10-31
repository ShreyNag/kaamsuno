# End-of-turn delay: current policy replayed on logged sessions

Policy: fast 400 ms only if Smart Turn >= 0.8 AND text arrived AND no incomplete ending; completion 400; hesitation/connector/postposition 3500/3500/3500; number/ID 3000/3000; unsure or no text 2500/2500; tone < 0.5 3000; cap 4000 ms.

Delay = last speech -> turn end, on turns that really ended (not cut-ins). "Then" is what each session's live policy did (older sessions ran older policies). Sessions without Smart Turn verdicts in the log (fixed_timeout runs, logs before 2026-10-25) are skipped.

| session | pauses | turn ends then / now | delay p50 then -> now (ms) | delay p95 then -> now (ms) | cut-ins then -> now | live cut-ins now held |
|---|---:|---:|---:|---:|---:|---:|
| 20261025-123951-2cd53e | 9 | 3 / 3 | 200 -> 2500 | 2725 -> 2950 | 2 -> 0 | 2/2 |
| 20261025-124129-619497 | 8 | 2 / 2 | 886 -> 2500 | 890 -> 2500 | 1 -> 0 | 1/1 |
| 20261025-155234-3c9570 | 7 | 5 / 5 | 544 -> 542 | 1844 -> 3400 | 1 -> 1 | 0/1 |
| 20261025-155355-d70b60 | 12 | 9 / 9 | 576 -> 2500 | 1682 -> 3500 | 1 -> 0 | 1/1 |
| 20261025-155519-f798e8 | 11 | 2 / 2 | 865 -> 1762 | 1170 -> 2876 | 2 -> 0 | 2/2 |
| 20261025-155613-19941d | 10 | 4 / 4 | 526 -> 3000 | 627 -> 3000 | 2 -> 2 | 0/2 |
| 20261031-181054-255b7f | 15 | 13 / 12 | 626 -> 487 | 2577 -> 2500 | 0 -> 0 | 0/0 |
| 20261031-181259-9128d5 | 2 | 2 / 2 | 519 -> 1473 | 565 -> 2397 | 0 -> 0 | 0/0 |
| 20261031-181332-a82717 | 8 | 6 / 6 | 538 -> 542 | 642 -> 2765 | 1 -> 1 | 0/1 |
| 20261031-181435-4d0466 | 7 | 6 / 6 | 774 -> 543 | 1386 -> 643 | 1 -> 1 | 0/1 |
| 20261031-181558-208daa | 11 | 4 / 4 | 1201 -> 2500 | 1201 -> 2925 | 2 -> 2 | 0/2 |
| 20261031-181658-46f83c | 13 | 3 / 3 | 467 -> 465 | 2750 -> 3196 | 2 -> 1 | 1/2 |
| 20261031-181858-02f41e | 2 | 1 / 1 | 594 -> 579 | 594 -> 579 | 1 -> 1 | 0/1 |

**All: delay p50 585 -> 673 ms, p95 2054 -> 3500 ms (n 60 -> 59); cut-ins 16 -> 9; live cut-ins now held 7/16.**

Turn ends now, by rule: other+smart_turn_complete 36, other+smart_turn_unsure 7, no_text 5, short_answer 5, id_in_progress 5, other+smart_turn_incomplete 4, continuation 2, postposition 2, hesitation 1, number_in_progress 1
