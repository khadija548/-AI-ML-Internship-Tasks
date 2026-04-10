import gradio as gr
from transformers import pipeline
import time
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
def predict_news(text):
    if not text.strip():
        return "Please enter a headline.", None
    
    # Get prediction
    result = classifier(text)[0]
    label_name = labels[result["label"]]
    score = result["score"]
    
    # Return both the label and a dict for the Label component (shows confidence bars)
    return label_name, {label_name: score}
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
).set(
    body_background_fill="*neutral_50",
    block_border_width="2px",
    block_shadow="*shadow_drop_lg"
)

with gr.Blocks(theme=theme, title="NewsIntel AI") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # 📰 NewsIntel AI 
                ### Enterprise-Grade Topic Classification
                Identify the category of any global news headline instantly using a fine-tuned BERT model.
                """
            )
            
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Headline Analysis",
                placeholder="e.g., SpaceX launches new satellite into orbit...",
                lines=4,
                interactive=True
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Analyze Headline", variant="primary")
            
            # Example inputs 
            gr.Examples(
                examples=[
                    ["Federal Reserve announces interest rate hike for next quarter."],
                    ["Manchester United secures victory in final minutes of the match."],
                    ["Quantum computing reaches new milestone in error correction."]
                ],
                inputs=input_text
            )

        with gr.Column(scale=1):
            output_label = gr.Textbox(label="Primary Category", interactive=False)
            output_conf = gr.Label(label="Confidence Score", num_top_classes=1)


    submit_btn.click(
        fn=predict_news, 
        inputs=input_text, 
        outputs=[output_label, output_conf]
    )
    
    clear_btn.click(lambda: [None, None, None], outputs=[input_text, output_label, output_conf])

if __name__ == "__main__":
    demo.launch()
