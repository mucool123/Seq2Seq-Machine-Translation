# English to Marathi Translation with Transformers

This repository contains the implementation of a Transformer model that translates English text to Marathi. The project is built using PyTorch and demonstrates the application of advanced machine learning techniques in natural language processing, specifically in the domain of machine translation.

## Project Overview

The goal of this project was to build a neural machine translation (NMT) system from scratch that can accurately translate English sentences into Marathi. The system utilizes a Transformer architecture, renowned for its efficiency and scalability in handling sequence-to-sequence tasks without the need for recurrent neural networks.

### Features

- **Custom Transformer Model**: Implementation of a Transformer model including encoder and decoder components.
- **Data Preprocessing**: Scripts to preprocess text data for training and evaluation.
- **Modular Code Structure**: Organized codebase into modules for ease of testing and extension.
- **Training Pipeline**: Comprehensive training and validation loops with detailed logging.
- **Inference Interface**: Script for translating new sentences using the trained model.

## Directory Structure

```plaintext
MarathiTranslationProject/
│
├── data/
│   ├── train.en    # English training data
│   ├── train.mr    # Marathi training data
│
├── models/
│   ├── __init__.py
│   ├── transformer.py    # Transformer model implementation
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # Data loading utilities
│   ├── tokenization.py   # Text tokenization utilities
│
├── configs/
│   ├── __init__.py
│   ├── training_config.py    # Configuration settings for training
│
├── train.py    # Script for training the model
├── validate.py    # Script for validating the model
├── translate.py    # Script for translating text using the trained model
└── requirements.txt    # Project dependencies



## Setup and Installation

### Clone the repository:

```bash
git clone https://github.com/mucool123/MarathiTranslationProject.git
cd MarathiTranslationProject

## Installation
pip install -r requirements.txt

## Prepare the Data:
Ensure that your data files are placed in the data/ directory. Update the paths in configs/training_config.py if necessary.

## Usage
Training the Model
Run the training script with:
python train.py
This will train the model using the data provided and save the trained model weights in the models/ directory.

## Validating the Model
To validate the model on a validation set, run:
python validate.py

## Translating Text
To translate English text to Marathi using the trained model:
python translate.py

## Contributing
Contributions to this project are welcome! Here are a few ways you can help:

Report bugs and issues.
Suggest improvements or new features.
Improve documentation or comments.

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contact
For any queries or discussions, feel free to contact me at gharpuremukul.work@gmail.com


