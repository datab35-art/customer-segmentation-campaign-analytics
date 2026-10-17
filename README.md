# Customer Segmentation & Campaign Analytics

K-Means clustering on healthcare member data to build actionable customer segments, paired with campaign performance analysis to measure outreach effectiveness across segments.

## Business Problem

Healthcare payers need to understand member populations to personalize outreach, reduce churn, and improve health outcomes. Generic communications have low engagement; segmentation enables targeted campaigns that drive measurable results.

## Approach

1. **Feature Engineering** — Derive behavioral features from utilization, engagement, demographics, and satisfaction data
2. **K-Means Clustering** — Segment members into distinct behavioral groups using scaled features
3. **Segment Profiling** — Characterize each segment by demographics, utilization patterns, and engagement levels
4. **Campaign Simulation** — Generate realistic campaign outreach data with segment-dependent response rates
5. **Campaign Analytics** — Measure open rates, click rates, conversion, and ROI by segment and channel

## Key Findings

| Segment | Size | Avg Age | Engagement | Campaign Response |
|---------|------|---------|------------|-------------------|
| High-Value Engaged | ~25% | 45-55 | High | 18-22% |
| Cost-Sensitive At-Risk | ~20% | 35-50 | Low | 8-12% |
| Young Digital-First | ~25% | 25-35 | Medium | 15-20% |
| Stable Low-Touch | ~30% | 50-65 | Low | 5-8% |

## Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **Clustering** — K-Means with elbow method and silhouette analysis
- **Visualization** — matplotlib, seaborn
- **Notebook** — Jupyter for interactive analysis

## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── member_profiles.csv
│   └── campaign_results.csv
├── src/
│   ├── generate_data.py
│   ├── feature_engineering.py
│   ├── segmentation.py
│   └── campaign_analytics.py
└── notebooks/
    └── segmentation_analysis.ipynb
```

## How to Run

```bash
pip install -r requirements.txt
python src/generate_data.py
jupyter notebook notebooks/segmentation_analysis.ipynb
```

## Skills Demonstrated

- Unsupervised learning (K-Means clustering)
- Feature engineering for behavioral segmentation
- Campaign A/B analytics and ROI measurement
- Healthcare member population analysis
- Data-driven marketing strategy
