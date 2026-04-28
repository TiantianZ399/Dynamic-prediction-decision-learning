# arXiv submission package

Main file: `main.tex`

Suggested arXiv category: q-fin.PM, with cs.LG as a possible cross-list.

Build locally:

```bash
latexmk -pdf main.tex
```

Important checks before submission:
1. Confirm permission to mention Boke Simu / Boke City and to describe the Wind data source.
2. Replace proprietary ETF data with a public replication dataset if a reproducible code package is released.
3. Review all empirical numbers and rerun the full 2019-2024 rolling experiment before claiming final performance.
4. Consider adding coauthors or supervisors if they contributed materially.
5. Keep the disclaimer and data availability statement unless your institution requests a different format.
