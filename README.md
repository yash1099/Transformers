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
The token classification application performs Named Entity Recognition (NER) on text. It employs the ALBERT model and is trained on the CoNLL-2003 dataset. Given an input sentence, the model identifies and tags specific entities such as persons, organizations, and locations.

3. Question Answering
The question answering application utilizes the T5 model and is trained on the SQuAD dataset. It enables the model to answer questions based on a given context or passage. Given a context and a question, the model generates the most relevant answer.

4. Summarization
The summarization application utilizes the DistilBERT model and is trained on the CNN/Daily Mail dataset. It can generate a summary of a given input text or article, condensing the content into a shorter, more concise form.

5. Translation
The translation application employs the Roberta model and is trained on the WMT dataset. It enables the translation of text between different languages. Given an input text and a desired target language, the model generates the translated output.
