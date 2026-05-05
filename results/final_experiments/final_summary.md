# Final Experiment Summary

## Retrieval
- TF-IDF: R@5=0.8032, R@10=0.8564, MRR=0.6570
- Dense MPNet: R@5=0.8723, R@10=0.9309, MRR=0.7429

## Error Analysis
- Top-5 success: 98 claims (52.1%)
- Top-5 retrieval_miss: 24 claims (12.8%)
- Top-5 stance_error: 66 claims (35.1%)

## Sentence Evidence Selection
- Rationale Hit@1=0.4737
- Rationale Hit@2=0.6507
- Rationale Coverage@2=0.5019

## Oracle Verification
- logreg_abstract: Accuracy=0.8361, Macro-F1=0.3589, F1(Supports)=0.1140, F1(Refutes)=0.0370
- scibert_abstract: Accuracy=0.9134, Macro-F1=0.6762, F1(Supports)=0.6691, F1(Refutes)=0.3906
- scibert_sentence_top2: Accuracy=0.9049, Macro-F1=0.6536, F1(Supports)=0.6113, F1(Refutes)=0.3846
- pubmedbert_abstract: Accuracy=0.9035, Macro-F1=0.6739, F1(Supports)=0.6577, F1(Refutes)=0.4000

## Dense End-to-End Pipeline
- dense_plus_logreg: JointHit@3=0.0532, JointHit@5=0.0585
- dense_plus_scibert_abstract: JointHit@3=0.4840, JointHit@5=0.5213
- dense_plus_scibert_sentence_top2: JointHit@3=0.4362, JointHit@5=0.4734
- dense_plus_pubmedbert_abstract: JointHit@3=0.5266, JointHit@5=0.5638

## Training Notes
- scibert_abstract: loaded_existing=True, best internal dev Macro-F1=0.7823
- scibert_sentence_top2: loaded_existing=False, best internal dev Macro-F1=0.8015
- pubmedbert_abstract: loaded_existing=False, best internal dev Macro-F1=0.8075

## Runtime
- load_data: 0.03 seconds
- build_tfidf_and_pairs: 7.72 seconds
- train_logreg: 26.64 seconds
- retrieval: 1.51 seconds
- load_scibert: 0.53 seconds
- baseline_evaluation: 6.76 seconds
- error_analysis: 2.95 seconds
- sentence_selection: 3.50 seconds
- train_or_load_sentence_scibert: 209.17 seconds
- sentence_scibert_evaluation: 5.30 seconds
- train_or_load_pubmedbert: 217.10 seconds
- pubmedbert_evaluation: 5.88 seconds