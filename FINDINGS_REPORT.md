# Genomic Language Model Findings Report

## 1. Scope of this report
This report summarizes findings from the current baseline pipeline outputs and explains what each result means.

## 2. Executive summary
1. The DNA masked language model trained successfully and improved steadily across 12 epochs.
2. Embedding-based repeat detection is strong with high discrimination power.
3. Motif discovery pipeline is operational and produces interpretable per-position probabilities and motif candidate tables.
4. Current results are strong for baseline stage, but biological validation and interval-level evaluation are still pending.

## 3. Language model training findings
From [outputs/baseline_cnn/training_history.json](outputs/baseline_cnn/training_history.json):

1. Validation loss improved from 1.1567 (epoch 1) to 1.0734 (epoch 12).
2. Validation masked accuracy improved from 0.4878 to 0.5201.
3. Validation perplexity improved from 3.1794 to 2.9253.
4. Average epoch runtime was about 31 seconds on GPU.

Interpretation:
1. The model is learning contextual nucleotide dependencies.
2. No obvious overfitting trend is seen yet because validation metrics generally improve with training.
3. Improvement rate is becoming smaller after later epochs, which suggests approach toward a local performance plateau for current data/model settings.

## 4. Repeat detection findings
From the repeat baseline run on embeddings extracted to [outputs/baseline_cnn/embeddings.npz](outputs/baseline_cnn/embeddings.npz):

1. Samples: 5000 windows
2. Class counts: repeats 2393, non-repeats 2607 (reasonably balanced)
3. Metrics:
- Accuracy: 0.9100
- F1: 0.9041
- ROC-AUC: 0.9670
- PR-AUC: 0.9691

Interpretation:
1. The representation space learned by the gLM is strongly informative for repeat/non-repeat separation.
2. High ROC-AUC and PR-AUC indicate robust ranking/separation quality, not just threshold luck.
3. This is a strong baseline result and supports continuing to stricter analyses.

## 5. Motif discovery findings
From [outputs/baseline_cnn/motif_discovery/run_metadata.json](outputs/baseline_cnn/motif_discovery/run_metadata.json):

1. Windows scanned: 50
2. Position-level prediction rows: 25600
3. Candidate motif contexts observed: 22142 unique (before top-k selection)
4. Runtime: about 12.7 seconds

From [outputs/baseline_cnn/motif_discovery/report/report_summary.json](outputs/baseline_cnn/motif_discovery/report/report_summary.json):

1. Mean true-base probability: 0.3348
2. Mean surprise (nats): 1.2153
3. Mean entropy (bits): 1.7534
4. Mean top1-top2 margin: 0.1560

Interpretation:
1. The model has informative but not fully deterministic confidence at masked sites.
2. Entropy below the maximum 2.0 bits indicates non-random structure in predictions.
3. Margin around 0.156 suggests moderate confidence separation between top nucleotide choices on average.
4. Very high-surprise motifs in [outputs/baseline_cnn/motif_discovery/motif_candidates.csv](outputs/baseline_cnn/motif_discovery/motif_candidates.csv) often have low frequency (count 1-2), so they should be treated as candidate anomalies or rare contexts, not final biological motifs yet.

## 6. Graph-by-graph explanation
### A. Position probability heatmap
File: [outputs/baseline_cnn/motif_discovery/report/position_probability_heatmap.png](outputs/baseline_cnn/motif_discovery/report/position_probability_heatmap.png)

What it shows:
1. Mean predicted probabilities for A/C/G/T at each masked position index.
2. Brighter cells represent higher model-assigned probability.

How to interpret:
1. Smooth or repeating structures across positions may indicate positional/context effects in the CNN receptive field.
2. Large base-specific shifts at certain positions can highlight context regions where nucleotide preference is strongest.
3. If all bases remain near uniform across positions, motif signal is weak.

### B. Entropy and confidence trend
File: [outputs/baseline_cnn/motif_discovery/report/position_entropy_margin.png](outputs/baseline_cnn/motif_discovery/report/position_entropy_margin.png)

What it shows:
1. Entropy curve: uncertainty at each position (lower is more confident).
2. Top1-top2 margin curve: confidence gap between best and second-best nucleotide (higher is more confident).

How to interpret:
1. Low entropy with high margin indicates strong model confidence and potentially informative sequence contexts.
2. High entropy with low margin indicates ambiguous contexts.
3. Regions where these two curves diverge are useful targets for deeper motif inspection.

### C. Top motif counts plot
File: [outputs/baseline_cnn/motif_discovery/report/top_motif_counts.png](outputs/baseline_cnn/motif_discovery/report/top_motif_counts.png)

What it shows:
1. Most frequent motif contexts among extracted candidates.

How to interpret:
1. High-count motifs are more stable and reproducible than count-1 motifs.
2. Combine frequency with surprise/confidence statistics to prioritize motifs for biological follow-up.

## 7. Tables produced and how to use them
1. [outputs/baseline_cnn/motif_discovery/report/top_surprising_sites.csv](outputs/baseline_cnn/motif_discovery/report/top_surprising_sites.csv)
- Use to identify sites where the true nucleotide was unexpectedly low probability.
- Good for anomaly/error analysis and rare-context mining.

2. [outputs/baseline_cnn/motif_discovery/report/top_confident_sites.csv](outputs/baseline_cnn/motif_discovery/report/top_confident_sites.csv)
- Use to identify high-confidence predictions for clearer motif context interpretation.

3. [outputs/baseline_cnn/motif_discovery/position_summary.csv](outputs/baseline_cnn/motif_discovery/position_summary.csv)
- Use for aggregate per-position trends across the scanned windows.

## 8. Biological and methodological caution
1. Current repeat labels are weakly supervised through soft-masked lowercase signal, not yet curated RepeatMasker interval labels.
2. Motif candidates are computationally ranked contexts and are not yet validated against known TF motif databases.
3. Current motif scan used one sequence ID and limited windows, so broader genome coverage is needed before strong biological claims.

## 9. Conclusions
1. Baseline objective is achieved: end-to-end pipeline from training to downstream repeat/motif analysis is functional.
2. Repeat detection performance is strong and supports utility of learned embeddings.
3. Motif discovery pipeline is producing meaningful intermediate signals and report artifacts.
4. Next value comes from stronger biological validation and interval-level benchmarking.

## 10. Recommended next actions
1. Run robustness sweep and report mean +- std from the generated robustness summary CSV once it is created.
2. Expand motif scan windows and stratify by repeat-rich versus non-repeat windows.
3. Add interval-overlap evaluation against external repeat annotations.
4. Add multi-class repeat classification and compare against non-neural baselines.
