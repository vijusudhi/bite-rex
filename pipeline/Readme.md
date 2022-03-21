This folder contains files for the following tasks.

## Data
- script to get data from EPO, [get_epo_data.ipynb](./get_epo_data.ipynb)
- spiders to crawl automotive webpages, [crawl_auto](./crawl_auto)
- script to fetch scraped content from automotive webpages, [get_scraped_data.ipynb](./get_scraped_data.ipynb)

## Pre-process
- script to pre-process data, [preprocess_data.ipynb](./preprocess_data.ipynb)

## Auxiliary
- script to generate data for Giza++, [generate_data_for_giza.ipynb ](./generate_data_for_giza.ipynb )
- script to read the dictionary from Giza++, [read_giza_dict.ipynb](./read_giza_dict.ipynb)
- script to generate simulated text retrieval dataset from EPO, [generate_data_for_phase2.ipynb](./generate_data_for_phase2.ipynb)

## Training models
- training scripts for static and contextual embeddings: 
  - **Static:**
    - word2vec with offline alignment, [train_offline-alg_word2vec.ipynb](./train_offline-alg_word2vec.ipynb)
    - word2vec with post-hoc alignment, [train_post-hoc_word2vec.ipynb](./train_post-hoc_word2vec.ipynb)
    - fastText with offline alignment, [train_offline-alg_fasttext.ipynb](./train_offline-alg_fasttext.ipynb)
    - fastText with post-hoc alignment, [train_post-hoc_fastText.ipynb](./train_post-hoc_fastText.ipynb)
   - **Contextual:**
     - pre-training mBERT with MLM and NSP on EPO dataset, [pre_training_MLM_NSP_mBERT.ipynb](./pre_training_MLM_NSP_mBERT.ipynb) 
     - **Note:** Sentence-BERT is trained right from the implementation found [here](https://github.com/UKPLab/sentence-transformers).

## Evaluation 
- evaluation script for static, contextual and baseline embeddings, [evaluation.ipynb](./evaluation.ipynb)

## Other experiments
- experiments to arrive at explanations using:
  - attention weights, [explanations_experiment_attention_weights.ipynb](./explanations_experiment_attention_weights.ipynb)
  - syntactic associations, [explanations_experiment_syn_associations.ipynb](./explanations_experiment_syn_associations.ipynb)



> **Note:** Due to confidentiality, the data used (in the scripts/notebooks) is not added in the repository. Please contact [me](mailto:vjusudhi@gmail.com) in case of further queries.
