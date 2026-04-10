# BERT News Topic Classifier

This project fine-tunes a BERT model to classify news headlines into categories using the AG News dataset.

## Categories
- World
- Sports
- Business
- Sci/Tech

## Technologies
- Hugging Face Transformers
- PyTorch
- Gradio
- Scikit-learn

## Model
BERT (bert-base-uncased)

## Dataset
AG News Dataset

## Results
Accuracy: ~81%  
F1 Score: ~0.80

## Run the Project

Install dependencies:

pip install -r requirements.txt

Train the model:

python train.py

Run the demo:

python app.py

## Demo

Example:

Input:
Stock markets fall after inflation concerns

Output:
Business

## Screenshots
attached to view

![Demo](screenshots/demo.png)
