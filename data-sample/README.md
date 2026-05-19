# Data Samples

This folder contains 20-row Parquet samples for the main project datasets. The full local data files are intentionally kept outside Git tracking, while these small samples make the schema inspectable on GitHub.

Sample rows are filtered to avoid blank key fields. Raw samples require non-empty `post_id`, `datetime`, and `text`; cleaned samples require non-empty `cleaned_text`; classified samples require successful `classification_status = ok` rows with non-empty narrative labels.

| Sample | Source File | Rows | Notes |
| --- | --- | ---: | --- |
| `trump_truth_social_sample.parquet` | `data/raw/trump_truth_social.parquet` | 20 | Raw post, engagement, media/link, category, and market-price columns. |
| `cleaned_events_sample.parquet` | `data/processed/cleaned_events.parquet` | 20 | ETL output with normalized dates, cleaned text, engagement features, and daily return columns. |
| `classified_events_sample.parquet` | `data/processed/classified_events.parquet` | 20 | Main classified fact table, including LLM labels and rule-based selected tickers. |
| `classified_events_gpt5mini_full_sample.parquet` | `data/processed/classified_events_gpt5mini_full.parquet` | 20 | Full-run LLM classification artifact before ticker-selection enrichment. |

Quick local inspection:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_parquet("data-sample/classified_events_sample.parquet")
print(df.shape)
print(df.head())
print(df.columns.tolist())
PY
```
