# Instalar dependencias (si aún no las tienes)
# pip install sentence-transformers faiss-cpu

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1. Preparación de Documentos Médicos
# Supongamos que tienes una lista de documentos con la información médica.
documents = [
    "La diabetes se caracteriza por niveles altos de glucosa en sangre, fatiga, sed excesiva y aumento en la frecuencia urinaria. Se trata con insulina o medicamentos orales, y se monitoriza con tests de glucosa.",
    "La hipertensión se define por una presión arterial elevada de forma persistente. Su manejo incluye cambios en el estilo de vida y medicación antihipertensiva.",
    # Agrega más documentos según tu base de datos
]

# 2. Generación de Embeddings usando Sentence Transformers
# Se utiliza un modelo preentrenado de código abierto para convertir los textos en vectores.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedding_model.encode(documents)

# 3. Creación del Índice FAISS
# Determinamos la dimensión del embedding y construimos un índice de similitud.
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings, dtype=np.float32))

# 4. Definir la Consulta del Paciente
# Ejemplo de input: síntomas y sospecha de enfermedad.
query = (
    "El paciente presenta fatiga crónica, sed excesiva y aumento de la frecuencia urinaria. "
    "Estos síntomas sugieren diabetes. Basado en la siguiente información, "
    "explícame en detalle sobre la enfermedad, qué evidencias respaldan ese diagnóstico, "
    "cuál es el tratamiento posible (incluyendo ejemplos de medicamentos y dosis generales), "
    "qué resultados esperar de los tests clínicos y cómo deberían ser los próximos seguimientos."
)

# Obtener el embedding de la consulta
query_embedding = embedding_model.encode([query])

# 5. Recuperar Documentos Relevantes con FAISS
k = 2  # Número de documentos a recuperar
distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)

# Combinar el contenido de los documentos recuperados para crear un contexto
retrieved_context = " ".join([documents[i] for i in indices[0]])
print("Contexto recuperado:")
print(retrieved_context)
print("\n---\n")

# 6. Generación de Respuesta con un Modelo de Lenguaje Abierto
# Usaremos un modelo de Hugging Face para generación. En este ejemplo, usamos 'google/flan-t5-base'.
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Crear un pipeline de generación
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Construimos un prompt combinando el contexto y la consulta
prompt = (
    f"Medical Information: {retrieved_context}\n\n"
    f"Query: {query}\n\n"
    "Provide a detailed response explaining the disease, the evidence supporting the diagnosis, the treatment (including examples of medications and general dosages), the expected test results, and the follow-up plan."
)

# Generar la respuesta
generated = generator(prompt, max_length=300, do_sample=False)
respuesta = generated[0]['generated_text']
print("Generated Answer:")
print(respuesta)
