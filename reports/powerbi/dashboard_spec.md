# Power BI Dashboard Specification

## Data Sources

- `reports/powerbi/tables/narrative_topic_counts.csv`
- `reports/powerbi/tables/posts_over_time_weekly.csv`
- `reports/powerbi/tables/tone_distribution.csv`
- `reports/powerbi/tables/policy_direction_distribution.csv`
- `reports/powerbi/tables/market_reaction_by_topic_ticker.csv`
- `reports/powerbi/tables/selected_ticker_distribution.csv`
- `reports/powerbi/tables/high_engagement_posts.csv`
- `reports/powerbi/tables/similar_event_search_output.csv`
- `reports/powerbi/tables/data_quality_summary.csv`

## Pages

1. Narrative Overview: topic counts, posts over time, tone distribution, policy direction distribution.
2. Market Reaction: average/median daily returns by topic and ticker, selected ticker counts, high-engagement posts.
3. Similar Event Search Output: retrieved posts, selected tickers, similarity scores, corresponding daily returns.
4. Data Quality: classification status, fallback rate, missing return rates, empty text count.

## Current Data Note

The local classified artifact is fallback-only unless live LLM classification has been run with `OPENAI_API_KEY`.
Current topic/tone/policy-direction distributions should therefore be interpreted as pipeline validation, not final analysis.
