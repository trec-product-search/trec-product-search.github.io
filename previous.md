---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

# Past Websites

* 2024: <a href="2024.html">Website</a>
* 2023: <a href="2024.html">Website</a>

# Previous Data
Visit [here](https://huggingface.co/trec-product-search) for all datasets.

---

## 2024 Datasets

| Type | Filename | File Size | Num Records | Description | Format |
|------|---------|----------|-------------|-------------|--------|
| Query to Query ID | [Query2QueryID](https://huggingface.co/datasets/trec-product-search/Product-Search-Corpus-v0.1/resolve/main/data/qid2query.tsv) | 946 KB | 30,734 | TREC style QueryID to Query Text | tsv: qid, query |
| Collection | [Collection (TREC Format)](https://huggingface.co/datasets/trec-product-search/Product-Search-Corpus-v0.1/resolve/main/data/trec/collection.trec.gz) | 1.81 GB (568 MB compressed) | 1,661,907 | TREC style corpus collection | tsv: docid, Title, Description |
| Train QREL (ESCI) | [Train QRELS (TREC Format)](https://huggingface.co/datasets/trec-product-search/Product-Search-Qrels-v0.1/resolve/main/data/train/product-search-train.qrels.gz) | 6.8 MB (2.1 MB compressed) | 392,119 | Train QRELs | tsv: qid, 0, docid, relevance label |
| Dev QREL (ESCI) | [Dev QRELS (TREC Format)](https://huggingface.co/datasets/trec-product-search/Product-Search-Qrels-v0.1/resolve/main/data/dev/product-search-dev.qrels.gz) | 2.9 MB (906 KB compressed) | 169,952 | Dev QRELs | tsv: qid, 0, docid, relevance label |
| Training Triples | [Train Triples JSONl](https://huggingface.co/datasets/trec-product-search/Product-Search-Triples/resolve/main/train.jsonl.gz) | 6.23 GB (1.28 GB compressed) | 20,888 | Training Triples json format | json: qid, query, positive passages, negative passages |

---

## 2023 Datasets

| Type | Filename | File Size | Num Records | Description | Format |
|------|---------|----------|-------------|-------------|--------|
| Test Queries | [2023 Test Queries (TREC Format)](https://huggingface.co/datasets/trec-product-search/product-search-2023-queries) | 12 KB (7 KB compressed) | 186 | 2023 Test Queries | tsv: qid, query text |
| Test QREL Synthetic (Non NIST) | [2023 Test QREL Synthetic (Non NIST)](https://huggingface.co/datasets/trec-product-search/Product-Search-Qrels-v0.1/blob/main/data/test/product-search-test.qrels.gz) | 18 KB (6 KB compressed) | 998 | 2023 Test QREL Synthetic (Non NIST) | tsv: qid, 0, docid, relevance label |
| Test QRELS (NIST Judged) | [2023 Test QREL (TREC Format)](https://huggingface.co/datasets/trec-product-search/product-search-2023-queries) | 2.1 MB (460 KB compressed) | 115,490 | 2023 Test Qrels | tsv: qid, 0, docid, relevance label |

---

## Getting Started/Tevatron Usage

To allow quick experimentation, we have made the datasets compatible with the popular [Tevatron](https://github.com/texttron/tevatron/) library. To train, index, and retrieve from the product search, researchers can take the [Tevatron MSMARCO Example Guide](https://github.com/texttron/tevatron/blob/main/examples/example_msmarco.md), update the dataset names, and run with their favorite model variant. For simplicity, an example is shown below.

### Steps

1. **Train a Model**
   ```bash
   python -m tevatron.driver.train \
       --output_dir product_search_bi_encoder_baseline \
       --model_name_or_path bert-base-uncased \
       --dataset_name trec-product-search/Product-Search-Triples
   ```

2. **Encode the Corpus**
   ```bash
   python -m tevatron.driver.encode \
       --output_dir=temp \
       --model_name_or_path product_search_bi_encoder_baseline \
       --dataset_name trec-product-search/product-search-corpus \
       --encoded_save_path corpus_emb.pkl \
       --encode_num_shard 1 \
       --encode_shard_index 0
   ```

3. **Create Query Embeddings for 2023 Queries**
   ```bash
   python -m tevatron.driver.encode \
       --output_dir=temp \
       --model_name_or_path product_search_bi_encoder_baseline \
       --dataset_name trec-product-search/Product-Search-Triples/test \
       --encoded_save_path query_emb.pkl \
       --q_max_len 32 \
       --encode_is_qry
   ```

4. **Retrieve Top Results**
   ```bash
   python -m tevatron.faiss_retriever \
       --query_reps query_emb.pkl \
       --passage_reps corpus_emb.pkl \
       -depth 100 \
       --batch_size -1 \
       --save_text \
       --save_ranking_to run.txt
   ```

5. **Convert Ranking to TREC Format**
   ```bash
   python -m tevatron.utils.format.convert_result_to_trec \
       --input run.txt \
       --output run.trec
   ```

6. **Evaluate with TREC Eval or ir_measures**
   ```bash
   ir_measures product_qrel.trec run.trec \
       NDCG@1 NDCG@3 NDCG@5 NDCG@10 NDCG@100 NDCG@1000 \
       AP@1 AP@3 AP@5 AP@10 AP@100 AP@1000
   