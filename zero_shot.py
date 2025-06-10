from transformers import pipeline
from multiprocessing import Pool, cpu_count

# Define your possible classes
possible_classes = [
    "trial",
    "guidelines",
    "paper"
]

# Initialize zero-shot classification pipeline globally in each process
def init_classifier():
    global classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_single_document(doc):
    # classifier is global in each process
    return classifier(doc, possible_classes)

def classify_documents(documents, candidate_labels=possible_classes):
    """
    Classify a list of documents into one of the candidate labels using zero-shot learning, in parallel.

    Args:
        documents (list of str): The documents to classify.
        candidate_labels (list of str): The possible classes.

    Returns:
        list of dict: Each dict contains 'sequence', 'labels', and 'scores'.
    """
    with Pool(processes=min(cpu_count(), len(documents)), initializer=init_classifier) as pool:
        results = pool.map(classify_single_document, documents)
    return results['labels'][0]

"""This script uses the Hugging Face Transformers library to perform zero-shot classification on a list of documents.
It defines a function `classify_documents` that takes a list of documents and classifies each one into predefined categories using a zero-shot classification model.
"""





"""
# Esempio di utilizzo con classificazione parallela:
if __name__ == "__main__":
    docs = [
        "This study enrolled 200 patients to test the efficacy of a new drug.",
        "The following recommendations are for the management of hypertension.",
        "We present a review of recent advances in cancer immunotherapy."
    ]
    # Classificazione parallela dei documenti
    classified = classify_documents(docs)
    for res in classified:
        print(f"Testo: {res['sequence']}\nClasse predetta: {res['labels'][0]} (score: {res['scores'][0]:.2f})\n")
"""
"""This example demonstrates how to classify a list of documents into predefined categories using zero-shot classification."""
