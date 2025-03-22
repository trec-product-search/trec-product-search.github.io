---
layout: default
---

**Welcome to the TREC 2025 Product Search and Recommendations Track!**

# Overview

Online shopping has drastically reshaped consumer expectations, emphasizing speed, convenience, and accuracy. The TREC Product Search and Recommendations Track aims to advance research in product search and recommendation systems by creating robust, **high-quality datasets** that enable the evaluation of end-to-end multimodal retrieval and nuanced product recommendation algorithms. The dataset spans diverse products across multiple languages and includes both textual descriptions and visual content, designed specifically to reflect real-world e-commerce search and recommendation challenges.


# Guidelines
## Important Dates

- **Data release:** TBD  
- **Submission period:** TBD  
- **Conference:** November 18th @ Gaithersburg, MD  

## Subscribe for Track Updates

<a href="subscribe.html" style="display: inline-block; background-color:rgb(15, 56, 100); color: white; padding: 12px 20px; font-size: 16px; font-weight: bold; border-radius: 5px; text-decoration: none; text-align: center;">Click Here to Subscribe</a>

## Tasks

### Search Task: Query Expansion for Product Search

Participants are tasked with enhancing retrieval effectiveness through advanced query expansion techniques. Teams will focus on reformulating user queries to close the vocabulary gap and significantly boost product retrieval performance.

- Improve query realism and diversity using synthetic data generation techniques.
- Focus specifically on task-oriented queries (e.g., "birthday party essentials").

### Recommendation Task: Item-to-Item Relations

This task addresses the nuanced challenge of identifying item relationships, such as complementary or substitute products, through explicit annotation and evaluation. Participants will build models capable of discerning detailed product relationships to enhance recommendations, supporting sophisticated e-commerce scenarios like complementary product recommendations and basket completion.

- Identify nuanced relationships (complementary, substitute, compatible).
- Develop ranked lists for complementary and substitute products based on a seed product.

## Instructions and Datasets
Detailed instructions and datasets will be released soon. Please subscribe and wait.

## Submission, Evaluation, and Judging

We will be following the classic TREC submission formatting, which is outlined below. White space is used to separate columns. The width of the columns in the format is not important, but it is crucial to have exactly six columns per line with at least one space between the columns.

```txt
1 Q0 pid1 1 2.73 runid1
1 Q0 pid2 1 2.71 runid1
1 Q0 pid3 1 2.61 runid1
1 Q0 pid4 1 2.05 runid1
1 Q0 pid5 1 1.89 runid1
```

Where:
- The **first column** is the topic (query) number.
- The **second column** is currently unused and should always be `"Q0"`.
- The **third column** is the official identifier of the retrieved passage (in passage ranking) or document (in document ranking).
- The **fourth column** is the rank at which the passage/document is retrieved.
- The **fifth column** shows the score (integer or floating point) that generated the ranking. This score **must** be in descending (non-increasing) order.
- The **sixth column** is the ID of the run being submitted.

### Evaluation Process

As the official evaluation set, we provide a set of **926 queries**, where **50 or more** will be judged by NIST assessors. For this purpose, NIST will be using **depth pooling** with separate pools for each task. Products in these pools will then be labeled by NIST assessors using **multi-graded judgments**, allowing us to measure **NDCG**.

### Submission Types

The main type of TREC submission is **automatic**, which means that there is no manual intervention when running the test queries. This means:
- You **should not** adjust your runs, rewrite the query, retrain your model, or make any manual adjustments after seeing the test queries.
- Ideally, you should only check the test queries to verify that they ran properly (i.e., no bugs) before submitting your automatic runs.

However, if you want to have **a human in the loop** for your run or make any manual adjustments to the model or ranking after seeing the test queries, you can mark your run as **manual** and provide a description of the types of alterations performed.
