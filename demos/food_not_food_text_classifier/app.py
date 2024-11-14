# 1. Importamos las librerías necesarias
import torch
import gradio as gr

from typing import Dict
from transformers import pipeline

# 2. Deifnimos nuestra función para usar el modelo
food_not_food_classifier = pipeline(task="text-classification",
                                    model="tonicanada/learn_hf_food_not_food_text_classifier-distilbert-base-uncased",
                                    top_k=1,
                                    device="cuda" if torch.cuda.is_available() else "cpu",
                                    batch_size=32)

def classify_text(text):
    # Usa el clasificador
    result = food_not_food_classifier(text)
    # Ahora accedemos al primer diccionario en la primera lista
    return result[0][0]['label'], result[0][0]['score']
    

# 3. Create a Gradio interface
description = """
A text classifier to determine if a sentence is about food or not food. 

Fine-tuned from [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) on a [small dataset of food and not food text](https://huggingface.co/datasets/mrdbourke/learn_hf_food_not_food_image_captions).

See [source code](https://github.com/mrdbourke/learn-huggingface/blob/main/notebooks/hugging_face_text_classification_tutorial.ipynb).
"""

demo = gr.Interface(
    fn = classify_text,
    inputs = "text",
    outputs=[gr.Label(num_top_classes=2), gr.Textbox()],
    title="🍗🚫🥑 Food or Not Food Text Classifier",
    description=description,
    examples=[["I whipped up a fresh batch of code, but it seems to have a syntax error."],
                       ["A delicious photo of a plate of scrambled eggs, bacon and toast."]])


# 4. Launch the interface
if __name__ == "__main__":
    demo.launch()
