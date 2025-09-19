# Deliverable 1 — Hybrid Credibility Scoring Prototype

## Overview
This project implements a prototype hybrid credibility scoring function that evaluates the trustworthiness of online sources. It combines:

- **Rule-Based Scoring**  
  Uses heuristics such as HTTPS, DOI detection, domain priors, and academic/government TLDs.  
  Optionally boosts scores for highly cited sources if a `citations` column is present in the dataset.

- **Machine Learning Scoring**  
  A simple Linear Regression model trained on TF-IDF sentence embeddings from article text.  
  The ML model is trained to approximate the rule-based scores and adds content sensitivity.

- **Hybrid Score**  
  Combines both components using a tunable weight parameter:  
  `hybrid = alpha * rule_score + (1-alpha) * ml_pred`  
  Default is `alpha=0.7` (70% rule-based, 30% ML).

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
- pandas
- numpy
- scikit-learn
- scipy

## Input Format
The script expects a **CSV file** with these required columns:
- `id` (unique identifier for each row)
- `url` (URL of the source to evaluate)
- `text` (article or content text to vectorize)

Optional column:
- `citations` (integer count of academic citations — boosts score up to +0.20)

## Running the Script
In Colab or terminal:
```bash
python deliverable1.py --input sample_data.csv --output hybrid_scores.csv --alpha 0.7 --debug
```

### Arguments
- `--input` : Path to input CSV  
- `--output` : Path to save results (default: `hybrid_scores.csv`)  
- `--alpha` : Weight for hybrid scoring (default: 0.7)  
- `--debug` : Print detailed signals from rule-based evaluation  

## Output Format
The script generates a CSV with:
- `id`
- `url`
- `rule_score`
- `citation_boost` (if applicable)
- `ml_pred` (ML-predicted score)
- `hybrid_score` (final blended score)

### Example Output
```csv
id,url,rule_score,citation_boost,ml_pred,hybrid_score
1,https://www.bbc.com/news/world-00000000,0.8977,0.0277,0.8977,0.8977
2,https://doi.org/10.1038/s41586-020-2649-2,0.8520,0.0700,0.5415,0.6968
3,https://www.reuters.com/world/europe/europe-ec...,0.9164,0.0300,0.9164,0.9164
4,https://medium.com/@randomauthor/opinion,0.6300,0.0000,0.6300,0.6300
5,notaurl,0.0000,0.0000,0.0000,0.0000
```

## Design Decisions
- **Base Score = 0.6**: Ensures unknown sites don’t start too high.  
- **Domain Priors**: Strongest signal, weighted at 80%. Includes academic, government, and reputable outlets.  
- **DOI / TLD Signals**: Provide boosts for `.edu`, `.gov`, and DOI references.  
- **Citation Boost**: Simulates academic credibility, with diminishing returns (`log1p`).  
- **Hybrid Weighting**: Default `alpha=0.7` favors rule-based interpretability.  

## Limitations
- Domain priors are hand-selected and incomplete (bias possible).  
- ML model is trained only on small toy datasets → poor generalization.  
- Content features are limited to TF-IDF; richer signals (author metadata, link analysis, social credibility) are future work.  
- Scores are heuristic, not absolute truth; users should interpret them as guidelines.  

n.  

