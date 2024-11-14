# Clasificadores de Texto con Transfer Learning usando Modelos de Hugging Face

Este repositorio contiene dos notebooks prácticos que ilustran cómo crear clasificadores de texto utilizando modelos preentrenados de Hugging Face. En ellos, aprenderás a aplicar transfer learning y fine-tuning para entrenar clasificadores personalizados con datos propios.

## Contenido del repositorio

1. **Tutorial para crear un clasificador de texto binario**  
   Un notebook paso a paso que explica cómo entrenar un modelo para clasificar texto en dos clases. En concreto, se toma como base el modelo `distilbert/distilbert-base-uncased`, un modelo optimizado y ligero de BERT, y se entrena con 250 frases. Algunas de las frases son sobre comida, y otras no. El modelo final clasifica cada frase en inglés como `food` o `not food`.

2. **Tutorial para crear un clasificador de texto multiclase**  
   Una guía práctica para entrenar un modelo capaz de clasificar texto en múltiples categorías. Se utiliza `distilbert/distilbert-base-multilingual-cased` como base, entrenándolo con 250 frases relacionadas con asignaturas escolares (Religión, Lengua y literatura, Educación Física, Artes, Idiomas extranjeros, Historia, Geografía, Física y química, Matemáticas). El modelo final predice la asignatura a la que pertenece una frase en español y, si no encaja en ninguna, también lo detecta.

## ¿Por qué usar Transfer Learning?

El uso de transfer learning con modelos de Hugging Face preentrenados y su posterior fine-tuning con datos personalizados abre un sinfín de posibilidades:

- **Personalización con pocos datos**: los modelos pueden ser entrenados con un conjunto relativamente pequeño de ejemplos, manteniendo un rendimiento sólido.
- **Tamaño manejable**: el modelo resultante es relativamente ligero, lo que facilita su almacenamiento en un servidor y su integración en aplicaciones más amplias.
- **Ahorro en recursos computacionales**: al aprovechar los modelos preentrenados, los tiempos de entrenamiento y los recursos necesarios son menores en comparación con un modelo entrenado desde cero.
- **Adaptación a datos específicos**: el fine-tuning permite personalizar un modelo para dominios especializados, como salud, derecho o finanzas, mejorando la precisión para estos sectores.

Esta flexibilidad permite optimizar recursos, reducir los tiempos de desarrollo y aprovechar al máximo el poder de la inteligencia artificial en proyectos de NLP (Natural Language Processing).

## Versatilidad de Hugging Face

Hugging Face no solo se limita a modelos de texto. Ofrece una amplia gama de modelos preentrenados en distintas modalidades, lo que permite a los desarrolladores crear soluciones multimodales de forma unificada:

- **Clasificación y generación de texto**
- **Procesamiento de imágenes**
- **Modelos de audio**: reconocimiento de voz, conversión de texto a voz, etc.
- **Modelos de video**

Esta biblioteca diversa facilita la creación de soluciones integradas en un solo ecosistema, permitiendo el acceso a modelos de última generación y fomentando la reutilización en distintos proyectos y aplicaciones.

## Requisitos previos

Para ejecutar los notebooks, asegúrate de tener instaladas las siguientes dependencias. Puedes instalarlas fácilmente usando el archivo `requirements.txt` proporcionado.