import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util
import torch
THRESHOLD = 0.95
MODEL_NAME = 'yikuan8/Clinical-Longformer'

# Load ICD codes and save their names in icd_name_list
d_icd_diagnoses = pd.read_csv('/path/to/D_ICD_DIAGNOSES.csv.gz', compression='gzip')
icd_name_list = d_icd_diagnoses['LONG_TITLE'].tolist()

# Lab value rules dictionary
lab_value_rules = {
    'Anion Gap': {'unit': 'meq/l', 'low': 3, 'normal': (3, 11), 'high': 11},
    'Apnea Interval': {'unit': 's', 'normal': 10},
    'Glucose': {'unit': 'mg/dl', 'low': 70, 'normal': (70, 99), 'prediabetic': (100, 125), 'diabetic': 125},
    'Heart Rate': {'unit': 'bpm', 'bradycardia': 60, 'normal': (60, 100), 'tachycardia': 100},
    'Sodium': {'unit': 'meq/l', 'low': 135, 'normal': (135, 145), 'high': 145},
    'SpO2': {'unit': 'percent', 'low': 95, 'normal': (95, 100)},
    'Creatinine': {'unit': 'mg/dl', 'low': 0.7, 'normal': (0.7, 1.3), 'high': 1.3},
    'Potassium': {'unit': 'meq/l', 'low': 3.5, 'normal': (3.5, 5), 'high': 5},
    'Magnesium': {'unit': 'mg/dl', 'low': 1.7, 'normal': (1.7, 2.2), 'high': 2.2},
    'Temperature F': {'unit': 'deg f', 'low': 96.8, 'normal': (96.8, 99.5), 'high': 99.5},
    'Wbc count': {'unit': 'k/ul', 'low': 4000, 'normal': (4000, 11000), 'high': 11000},
    'Hemoglobin': {'unit': 'g/dl', 'low': 12, 'normal': (12, 16), 'high': 16},
    'Urea Nitrogen': {'unit': 'mg/dl', 'low': 7, 'normal': (7, 20), 'high': 20},
    'Bicarbonate': {'unit': 'meq/l', 'low': 22, 'normal': (22, 29), 'high': 29},
    'Chloride': {'unit': 'meq/l', 'low': 96, 'normal': (96, 106), 'high': 106},
    'Calcium': {'unit': 'mg/dl', 'low': 8.5, 'normal': (8.5, 10.2), 'high': 10.2},
    'Phosphate': {'unit': 'mg/dl', 'low': 2.5, 'normal': (2.5, 4.5), 'high': 4.5},
    'Oxygen [Partial pressure] in Blood': {'unit': 'mmHg', 'low': 75, 'normal': (75, 100), 'high': 100},
    'Bicarbonate [in Blood]': {'unit': 'mmol/l', 'low': 22, 'normal': (22, 29), 'high': 29},
    'Temperature C': {'unit': 'deg c', 'low': 36.1, 'normal': (36.1, 37.2), 'high': 37.2}
}

# Generate all possible query combinations of lab values
lab_values_query_list = []
for key, rules in lab_value_rules.items():
    if 'low' in rules:
        lab_values_query_list.append(f"low {key}")
    if 'high' in rules:
        lab_values_query_list.append(f"high {key}")
    if 'normal' in rules:
        lab_values_query_list.append(f"normal {key}")
    if 'prediabetic' in rules:
        lab_values_query_list.append(f"prediabetic {key}")
    if 'diabetic' in rules:
        lab_values_query_list.append(f"diabetic {key}")
    if 'bradycardia' in rules:
        lab_values_query_list.append(f"bradycardia {key}")
    if 'tachycardia' in rules:
        lab_values_query_list.append(f"tachycardia {key}")

# Load the list of documents
with open('data/primekg/documents.pkl', 'rb') as f:
    documents = pickle.load(f)

# Load the SBERT and Clinical-Longformer models
longformer_model = SentenceTransformer(MODEL_NAME)

# Encode the documents
document_embeddings = longformer_model.encode(documents, convert_to_tensor=True)

# Function to process queries in batches
def process_queries_in_batches(queries, batch_size=100):
    retrieved_documents = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        query_embeddings = longformer_model.encode(batch_queries, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_embeddings, document_embeddings)
        
        for j, query in enumerate(batch_queries):
            relevant_docs = []
            for k, doc in enumerate(documents):
                if cosine_scores[j][k] > THRESHOLD:
                    relevant_docs.append((doc, float(cosine_scores[j][k])))
            if relevant_docs:
                retrieved_documents.append((query, relevant_docs))
    return retrieved_documents

# Process queries in batches
retrieved_documents = process_queries_in_batches(lab_values_query_list)

# Concatenate all retrieved documents into a single string
all_retrieved_docs_str = ""
for query, docs in retrieved_documents:
    for doc, score in docs:
        all_retrieved_docs_str += doc + " "

# Save the retrieved_documents to retrieved_documents.pkl
with open('/path/to/retrieved_documents.pkl', 'wb') as file:
    pickle.dump(retrieved_documents, file)

# Optionally, save the concatenated string to a text file
with open('/path/to/all_retrieved_docs.txt', 'w') as file:
    file.write(all_retrieved_docs_str)

print("Retrieved documents have been saved.")

