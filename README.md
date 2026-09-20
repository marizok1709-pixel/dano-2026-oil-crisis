# DANO Тбанк Hackathon 2026: The Summer 2026 Oil Crisis

I took part in **DANO**, a data-analytics hackathon by **Т-Банк (T-Bank)**, where I got to work with a massive amount of real-world-scale data on the **2026 summer oil crisis** and investigate what it meant for banks, drivers, and the wider economy.

The brief: a national fuel shortage hit in summer 2026, gas stations started rationing sales, and queues formed everywhere. T-Bank handed teams ~430,000 rows of anonymized client, fuel-transaction, and traffic-fine data and asked a sharp question — **did the fuel crisis actually cause a spike in driving violations** (the kind you'd expect from stressed, queue-weary drivers: illegal parking, crossing stop lines, lane violations, phone use)? And more broadly, what does this mean for how a bank should think about client risk and product design during a supply shock?

This repo contains the full pipeline: raw data → cleaning → exploratory analysis → causal hypothesis testing → final report.

## TL;DR result

There **is** a statistical association between the crisis period and a rise in fuel-related violations (IRR ≈ 2.1, p = 0.0005) — but after running it through six independent robustness checks (event-study pre-trends, negative controls, placebo dates, geographic splits, offset models, and difference-in-differences), the association **does not hold up as a causal effect**. The seatbelt-violation control group — which queues at a gas station cannot plausibly affect — grew *even faster* than the target violations; 55% of randomly chosen placebo dates produce the same "significant" jump; and drivers who were actually bound by the fuel-rationing limit show **no** extra violations relative to those who weren't (DiD IRR = 0.97, p = 0.31).

**Bottom line for the jury/bank:** the honest finding is "association observed, causal mechanism not supported" — the crisis very plausibly changed fuel demand and pricing behavior sharply (fill-up volumes dropped ~37%, top-decile fill-ups nearly vanished), but the data can't isolate a queue-driven spike in bad driving from ordinary seasonal/registration trends. That distinction matters for a bank deciding whether to act on this signal (e.g., risk-scoring or product interventions) — a real effect would justify one thing, a spurious correlation would justify another.

Full writeup: [`cleaned dataset/FINAL_results_RU.md`](cleaned%20dataset/FINAL_results_RU.md) (Russian, this was a Russian-language competition).

## What's in here

```
given data/          Raw files handed out by T-Bank: client demographics, 2026 fines,
                      fuel transactions, plus the task brief and grading criteria (PDFs).
cleaned dataset/      The cleaned, documented, analysis-ready version of the data —
                      panel dataset, anomaly register, assumptions log, and the final
                      hypothesis-testing notebook/report.
solution/             The full cleaning + validation pipeline (notebooks + script) that
                      turns "given data" into "cleaned dataset".
some visuals/         Key charts: the fuel-rationing shock, event-study plots, model
                      comparison across 10 specifications, violation structure, etc.
```

### Data at a glance

| File | Rows | What it is |
|---|---:|---|
| `clients_demographics.csv` | 28,237 | client × vehicle |
| `fines_2026.csv` | 77,978 | traffic-fine records |
| `fuel_transaction.csv` | 379,699 | fuel purchase transactions |

Fixed analysis cohort: 23,452 clients subscribed before the crisis window began. Client IDs are pseudonymous hashes — no names or raw PII.

### What actually happened to fuel supply

| Metric | Before rationing | After hard cap | Change |
|---|---:|---:|---:|
| p90 fill-up volume | 55.9 L | 35.62 L | −36% |
| Share of fill-ups > 50L | 20.9% | 0.7% | 29× drop |
| Daily sales volume | 84,100 L | 54,200 L | −35.5% |
| Median price | ₽68.50/L | ₽70.20/L | +2.5% |

This was a **quantity shock** (rationing), not primarily a price shock.

### The modeling approach

Daily counts of six violation categories plausibly linked to fuel-station stress (parking, stop-line, road-shoulder, dedicated-lane, marking, phone-while-driving), modeled with Poisson / Negative-Binomial interrupted time series around two event dates (crisis start: May 1; hard volume cap: June 19), then stress-tested with:

- **Event study** — weekly coefficients, checking for pre-trends
- **Negative controls** — violation categories a fuel queue *shouldn't* affect (e.g. seatbelt)
- **Placebo dates** — rerunning the model with the "event" moved to arbitrary dates
- **Offset model** — is this a rise in violations, or just a rise in overall fine-issuance volume?
- **Geographic split** — does the effect hold outside Moscow/St. Petersburg?
- **Difference-in-differences** — do clients actually bound by the fuel cap see more violations than those who weren't?

10 model specifications × 2 event dates, all reported side by side rather than cherry-picked.

## Reproducing it

```python
import pandas as pd
D = "cleaned dataset/data/"

panel = pd.read_csv(D+"05_panel_client_month.csv", sep=';')
fuel  = pd.read_csv(D+"04_fuel_transactions_clean.csv", sep=';', parse_dates=['order_dt'])
fines = pd.read_csv(D+"03_fines_clean.csv", sep=';', parse_dates=['offence_dt'])
```

See [`cleaned dataset/README.md`](cleaned%20dataset/README.md) for the full data dictionary and the five rules that keep you from drawing false conclusions from this dataset (e.g. why August/September fine counts are a registration-lag artifact, not a real drop). The end-to-end statistical pipeline is reproducible from [`cleaned dataset/FINAL_hypothesis_pipeline.ipynb`](cleaned%20dataset/FINAL_hypothesis_pipeline.ipynb) (~22s runtime, 0 errors).

## Why I'm sharing this

Hackathons like DANO are a good showcase of working end-to-end with messy, large-scale real-world data: cleaning it, documenting every assumption, building a causal-inference pipeline, and — maybe most importantly — being willing to report a negative/null result instead of forcing a good story out of a p-value.
