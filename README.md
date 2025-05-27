# NLP Final Project - Misinformation Detection

This repository contains a project focused on detecting misinformation in Brazilian Portuguese WhatsApp forwards. The project compares different NLP models applied to a binary text classification task: identifying whether a news piece is fake or not.

## Project Overview

The main goal of this project was to evaluate and compare different approaches for text classification, ranging from traditional machine learning methods to pre-trained language models.

### Step-by-Step Process

1. **Data Preprocessing**: 
   - Loaded and cleaned the dataset of WhatsApp messages.
   - Performed basic text preprocessing such as lowercasing and removal of special characters.

2. **Tokenization and Vectorization**:
   - For the traditional machine learning model, applied TF-IDF vectorization to convert text into numerical features.
   - For the transformer-based models, used pre-trained tokenizers compatible with BERT models.

3. **Model Training**:
   - **TF-IDF + Logistic Regression**: A baseline model using TF-IDF features with Logistic Regression.
   - **BERT-base-uncased + Logistic Regression**: Used embeddings from the English pre-trained BERT-base model, followed by Logistic Regression.
   - **BERTimbau + Logistic Regression**: Employed the BERTimbau model, a BERT variant pre-trained on Brazilian Portuguese, combined with Logistic Regression.

4. **Evaluation**:
   - Assessed model performance using standard metrics: F1-Score, Accuracy, Precision, and Recall.
   - Visualized performance metrics to compare model effectiveness.

## Results Analysis

The following table summarizes the performance of each model:

| Model                      | F1-Score | Accuracy | Precision | Recall  |
|--------------------------- |---------- |--------- |---------- |-------- |
| TF-IDF + LogisticRegression| 0.9518    | 0.9519   | 0.9526    | 0.9519  |
| BERT-base-uncased + LR     | 0.9314    | 0.9315   | 0.9330    | 0.9315  |
| BERTimbau + LR             | **0.9593**| **0.9593**| **0.9594**| **0.9593**|

The graphic below show this information:

![alt text](imgs/graficos.png)

### Key Insights:

- The **BERTimbau + Logistic Regression** model achieved the best performance across all metrics, confirming the advantage of using a model pre-trained on Brazilian Portuguese data for this task.
- The **TF-IDF + Logistic Regression** baseline performed surprisingly well, achieving over 95% accuracy, demonstrating that even traditional methods can be highly effective with good feature engineering.
- The **BERT-base-uncased + Logistic Regression** underperformed compared to the other models, likely due to the language mismatch, as it was pre-trained on English data.

## Conclusion

The results clearly indicate that using a language-specific model like BERTimbau significantly enhances performance in misinformation detection tasks in Brazilian Portuguese. However, traditional models such as TF-IDF combined with Logistic Regression can still deliver competitive results.

## Reference

Monteiro R.A., Santos R.L.S., Pardo T.A.S., de Almeida T.A., Ruiz E.E.S., Vale O.A. (2018) Contributions to the Study of Fake News in Portuguese: New Corpus and Automatic Detection Results. In: Villavicencio A. et al. (eds) Computational Processing of the Portuguese Language. PROPOR 2018. Lecture Notes in Computer Science, vol 11122. Springer, Cham.
