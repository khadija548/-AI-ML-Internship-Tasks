from transformers import pipeline
import gradio as gr

# Load trained model
classifier = pipeline(
    "text-classification",
    model="models/news_classifier"
)

labels = {
    "LABEL_0": "World",
    "LABEL_1": "Sports",
    "LABEL_2": "Business",
    "LABEL_3": "Sci/Tech"
}

def predict(text):
    result = classifier(text)[0]
    return labels[result["label"]]

interface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="BERT News Topic Classifier",
    description="Enter a news headline and the model will predict its topic."
)

interface.launch()