# Transformers
This project showcases various natural language processing (NLP) applications utilizing popular Hugging Face models and datasets. The applications included are text classification, token classification, question answering, summarization, and translation.

# Introduction
This project leverages state-of-the-art NLP models provided by Hugging Face, such as BERT, ALBERT, T5, DistilBERT, and Roberta, to build a suite of NLP applications. By utilizing these powerful models and rich datasets, this project aims to demonstrate the capabilities of text classification, token classification, question answering, summarization, and translation tasks.

# Models and Datasets
1. This project utilizes the following Hugging Face models and datasets:
 * BERT (Bidirectional Encoder Representations from Transformers)
 * ALBERT (A Lite BERT for Self-supervised Learning of Language Representations)
 * T5 (Text-to-Text Transfer Transformer)
 * DistilBERT (Distilled BERT)
 * Roberta (Robustly Optimized BERT Approach)

2. The models are fine-tuned on various datasets, including but not limited to:
 * SetFit/tweet_sentiment_extraction
 * SpeedOfMagic/ontonotes_english
 * ncbi_disease
 * squad_v2

For more information on these models and datasets, please refer to the Hugging Face documentation.

# Application Examples
1. Text Classification
Text classification is the task of assigning predefined categories or labels to text documents. Using 
BERT variants, I have gained experience in text classification tasks, including sentiment analysis. I have learned how to 
fine-tune BERT models like Albert, RoBERTa, DistillBERT, BART on labeled datasets, perform model evaluations, and 
interpret the results. The dataset I used for this was https://huggingface.co/datasets/SetFit/tweet_sentiment_extraction

2. Token Classification
Token classification involves labeling individual tokens within a text 
sequence. This task is commonly used for named entity recognition, part-of-speech tagging, and other sequence labeling 
tasks. I have worked on token classification using BERT models like BERT, ALBERT, RoBERTa learned about tokenlevel predictions and trained models to identify and classify tokens in various contexts. The dataset I used for this was: SpeedOfMagic/ontonotes_english and ncbi_disease.

3. Question Answering
 I delved into the techniques and methodologies involved in Question Answering using BERT. This includes fine-tuning BERT models like RoBERTa, T5, BigBird, Longformer on QA 
datasets, understanding the format of input data (context and question), and generating precise answers.The data I used : https://huggingface.co/datasets/squad_v2 and squad

4. Summarization abd Translation
BERT variants have also been utilized for text summarization and translation tasks. I have explored techniques for abstractive summarization, where BERT models are fine-tuned to generate concise summaries of longer texts. Additionally, the BERT model I learned are T5, Pegasus, BigBirdPegasus can be applied to machine translation tasks, enabling the conversion of text between different languages. The dataset I used: https://www.kaggle.com/datasets/sunnysai12345/news-summary and Translation: 
https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset
