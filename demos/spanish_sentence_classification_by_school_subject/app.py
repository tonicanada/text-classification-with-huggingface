# 1. Import the required libraries
import torch
import gradio as gr

from typing import Dict
from transformers import pipeline

# 2. Define our function to use with our model
spanish_sentence_classification_by_school_subject_pipeline = pipeline(task="text-classification",
                                    model="tonicanada/learn_hf_spanish_sentence_classification_by_school_subject",
                                    top_k=1,
                                    device="cuda" if torch.cuda.is_available() else "cpu",
                                    batch_size=32)    

def classify_text(text):
    # Usa el clasificador
    result = spanish_sentence_classification_by_school_subject_pipeline(text)
    # Extrae la etiqueta y la puntuación (score)
    label = result[0][0]['label']
    score = result[0][0]['score']
    return {label: score}  # Devuelve un diccionario con la etiqueta y la puntuación


# 3. Create a Gradio interface
description = """
Un clasificador de texto que indica a qué asignatura se refiere la frase. 

Fine-tuned from [DistilBERT](https://huggingface.co/distilbert/distilbert/distilbert-base-multilingual-cased) on a [small dataset of food and not food text](https://huggingface.co/datasets/mrdbourke/learn_hf_food_not_food_image_captions).
"""

demo = gr.Interface(
    fn = classify_text,
    inputs = "text",
    outputs=gr.Label(num_top_classes=10),
    title="📚🔍 Clasificador de asignaturas",
    description=description,
    examples=[["Matemáticas: 5 al cuadrado es 25"],
                       ["Geografía: París es la capital de Francia"]])


# 4. Launch the interface
if __name__ == "__main__":
    demo.launch()
