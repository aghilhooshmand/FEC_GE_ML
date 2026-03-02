# Abstract and One-Page Summary for Journal Submission

**Title (suggested):** *Fitness Evaluation Caching for Grammatical Evolution in Interpretable Clinical Classification: Speedup with Comparable Accuracy*

---

## Abstract

Interpretability of machine learning models is increasingly critical in healthcare and medicine, where decisions must be auditable and understandable. Grammatical Evolution (GE) addresses this need by evolving *interpretable classifiers*: each solution is an explicit expression (formula) over input features, defined by a grammar, so that predictions are explained by a combination of thresholds and operations on those features. A major drawback of GE for classification is computational cost: fitness evaluation over the full training set at every generation is expensive and limits scalability. We propose a **Fitness Evaluation Cache (FEC)** that reduces this cost while preserving accuracy. Under FEC, each individual is identified by a fingerprint derived from its *behaviour on a small sample* of the data (e.g. predictions on representative points). When this fingerprint matches a previous evaluation (cache hit), the stored fitness is reused; when it does not (cache miss), fitness is computed on the *full training set* and the result is cached. We use **multiple sampling strategies**—including k-means, k-medoids, farthest-point sampling, and stratified sampling—to select the fraction of data used for building the cache key, and we compare FEC against a baseline that always uses the full training set. Experiments on **several clinical and biomedical datasets** (including breast cancer recurrence and heart disease) show that FEC achieves a substantial **speedup** (fewer full fitness evaluations) while maintaining **accuracy comparable to training on the full data**. The choice of dataset and sampling method influences the trade-off between speedup and accuracy, demonstrating that the approach generalises across domains and can be tuned for different applications. We conclude that FEC makes GE-based interpretable classification more practical for clinical use without sacrificing the explainability that makes GE suitable for healthcare.

**Keywords:** Grammatical Evolution; interpretability; explainable AI; fitness evaluation cache; clinical classification; sampling methods; computational efficiency.

---

## One-Page Summary

### 1. Motivation and explainability

In healthcare and medicine, model **explainability** is essential for trust, regulation, and clinical adoption. Black-box classifiers are often unsuitable because practitioners need to understand *why* a prediction was made. **Grammatical Evolution (GE)** for classification is well aligned with this requirement: it evolves **interpretable models** in the form of explicit expressions over input features. Each individual is a formula (e.g. logical and arithmetic combinations of features) specified by a context-free grammar, so that the decision rule is inspectable and can be stated in terms of thresholds and feature combinations. This makes GE a strong candidate for clinical decision support, provided that its computational cost can be managed.

### 2. Computational challenge and FEC idea

A central bottleneck in GE is **fitness evaluation**: every individual in every generation must be evaluated on the training data. With large datasets or long runs, this becomes prohibitively time-consuming. We address this with a **Fitness Evaluation Cache (FEC)**. The goal is to obtain **accuracy comparable to using the full training set**, while performing **fewer full evaluations** and thus achieving a significant **speedup**. The key idea is to reuse fitness values when we can safely treat two individuals as equivalent. We do this by (i) building a **cache key** (fingerprint) for each individual using only a **small sample** of the data (e.g. predictions on representative points), and (ii) on a **cache hit**, returning the previously stored fitness; on a **cache miss**, computing fitness on the **full training set**, storing the result, and then returning it. Thus, full training data is used only when necessary (on misses), while cache hits avoid expensive re-evaluation entirely.

### 3. Sampling methods and cache behaviour

We use **several sampling methods** to select the subset of data used for constructing the cache fingerprint: **k-means**, **k-medoids**, **farthest-point sampling**, and **stratified sampling**, as well as combinations (e.g. union of several methods). For each configuration we vary the **sample fraction** (e.g. 10%–60% of the training set). When an individual is evaluated, its behaviour on this sampled subset (and optionally its syntactic form) defines the cache key. If the key is found in the cache, we return the cached fitness (hit); otherwise we evaluate on the **full training data**, store the fitness with that key, and return it (miss). Optional **fake-hit** checks can re-evaluate on the full set for a subset of hits to monitor drift; the main evolutionary loop still benefits from the cache. This design ensures that all *stored* fitness values correspond to evaluations on the full training set, so that selection pressure remains aligned with true performance.

### 4. Experimental scope and findings

Experiments are conducted on **multiple datasets** (including clinical breast cancer and heart disease data) to assess how well FEC generalises and how **different datasets and sampling strategies** affect the trade-off between **speedup** and **accuracy**. We compare (i) a **baseline** (no cache; every evaluation on the full training set) with (ii) **FEC-enabled runs** using different sample fractions and sampling methods. Metrics include test-set accuracy (or equivalent fitness), cache hit rate, number of full vs sample-based evaluations, and wall-clock or generation time where applicable. Results show that FEC can achieve a **substantial reduction in full fitness evaluations** (and thus runtime) while keeping **accuracy comparable to the full-data baseline**. The degree of speedup and the small impact on accuracy depend on the dataset and the choice of sampling method and sample size, indicating that the technique is applicable across different clinical and biomedical settings and can be tuned for each.

### 5. Conclusion

We demonstrate that a **Fitness Evaluation Cache** combined with **diverse sampling strategies** makes Grammatical Evolution more scalable for interpretable clinical classification without sacrificing the explainability that is essential in healthcare. By using a fraction of the data only for cache indexing and performing full training-set evaluation on cache misses, we obtain **near–full-data accuracy** with **significant speedup**, and we show that this holds across **multiple datasets** and sampling methods. This supports the use of GE-based interpretable models in practice where both explainability and efficiency matter.

---

*Document prepared for journal submission (abstract + one-page). Adjust title, keywords, and dataset names to match the target journal and final experiments.*
