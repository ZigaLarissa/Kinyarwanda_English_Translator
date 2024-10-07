# Machine Translation with LSTM - Tourism Dataset

This project focuses on building a machine translation model using an LSTM (Long Short-Term Memory) neural network. The goal is to translate between two languages, English and Kinyarwanda, using a tourism-related dataset. The notebook walks through data preprocessing, model building, and evaluation.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training & Evaluation](#training-and-evaluation)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Requirements](#requirements)
9. [How to Run](#how-to-run)
10. [Contributors](#contributors)

## Overview
This project builds a machine translation model to convert text from English to Kinyarwanda (or vice versa). Using an LSTM-based encoder-decoder architecture, the model is trained to predict sequences in the target language based on input sequences from the source language.

## Dataset
- **Input Dataset**: A tourism dataset containing English and Kinyarwanda phrases. [mbazaNLP/NMT_Tourism_parallel_data_en_kin](!mbazaNLP/NMT_Tourism_parallel_data_en_kin)
- **File Format**: The data is stored in TSV format (`tourism_train_data.tsv`) and contains columns for `source` (English) and `phrase` (Kinyarwanda).
- **Preprocessing**: The text data is cleaned by lowercasing, removing punctuation, and stripping extra whitespace.

## Preprocessing
Before training the model, the data goes through several preprocessing steps:
1. **Text Cleaning**: Lowercasing, removing punctuation, and extra spaces.
2. **Tokenization**: Using Keras' `Tokenizer` to convert text to sequences of integers.
3. **Padding**: Padding sequences to ensure uniform input size using `pad_sequences`.
4. **Vocabulary Size**: Vocabulary sizes for both languages are determined based on the tokenized sequences.

## Model Architecture
The translation model is built using an encoder-decoder architecture with LSTM layers:
- **Encoder**: Converts source sequences (English) into hidden states.
- **Decoder**: Takes the encoder's states and generates target sequences (Kinyarwanda).
- **Embedding Layer**: Used in both encoder and decoder to handle word representations.
- **LSTM Layers**: Used to learn the temporal dependencies between words in sequences.
- **Output Layer**: A dense layer with softmax activation to predict the next word in the sequence.

## Training and Evaluation
- The model is trained with **sparse categorical cross-entropy** loss and **Adam optimizer**.
- The data is split into 80% training and 20% validation sets.
- **Batch Size**: 250
- **Epochs**: 5
- Training and validation accuracy and loss are tracked.

## Results
After training, the model achieved the following:
- **Validation Loss**: `1.3120923042297363`
- **Validation Accuracy**: `0.8434032797813416`

The model's performance can be improved by fine-tuning the hyperparameters or increasing the dataset size, *which I reduced before training due to time and ram limit*.

## Visualizations
- **Word Clouds**: Visualized the most common words in both the English and Kinyarwanda datasets using word clouds.
- **Accuracy and Loss Plots**: Plotted training and validation accuracy/loss over the epochs.

[Image]

[Image]

## Future Work
- Hyperparameter tuning (embedding dimension, hidden units, batch size).
- Use of more advanced models like Transformers for better performance.
- Increase the number of epochs for more thorough training.
- Try using data augmentation techniques for the text.

## Requirements
To run the notebook, you'll need the following dependencies:
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy 1.x, *avoid using 2.x, it might couse conflicts*
- Matplotlib
- WordCloud
- scikit-learn
- Re (Regular Expressions)
- Pickle (for saving tokenizers)

Install the required packages using:
```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/username/KINYARWANDA_ENGLISH_TRANSLATOR.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset (`tourism_train_data.tsv`) in the project directory.
4. Open and run the notebook:
   ```bash
   jupyter notebook eng_kiny_trans.ipynb
   ```

## Contributors
- **Larissa Bizimungu** - Developer

Feel free to contribute by raising issues or submitting pull requests for improvements!
