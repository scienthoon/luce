# Phishing split IDs

Item IDs (the `id` column of PhishNChips, as used by anisselbd/jev-phishing-bench `prepare_data.py`) for the four
disjoint splits used in Luce E17. IDs only, no e-mail content. Source `AreLit/PhishNChips` at revision
`89afcc39610084298c4679159cb2e27d9ffffa46`. Together the four lists cover exactly the benchmark's 2,000 e-mails.

| split | items | phishing | legitimate | sha256 of our converted split file |
|---|---|---|---|---|
| train | 1000 | 500 | 500 | `4df854068f73ec04679ac412aaffee4bb0ec29ea44bdd02e27b6dd2fbcfaae71` |
| val | 250 | 125 | 125 | `1088160a3d1495b7f224fd933ecfa919f3b20fe4aa50c308fbb486ebceecef9e` |
| calibration | 250 | 125 | 125 | `d552e1165dfa9430340c69d0fdac4fe3958bc294f3b5bd1ae4b13d597e18ea90` |
| test | 500 | 250 | 250 | `a5444bf25cbe653a990f45549f764e95934c7ad7c5f819b69323285def94edc4` |

Training used `train` only; `val` chose the checkpoint and temperature, `calibration` refit the temperature, and
`test` was evaluated once. No label from `val`, `calibration` or `test` was used for gradient updates.

Scope: all four splits come from the same corpus, so the test result is in-distribution. It does not show
generalisation to other phishing sources.
