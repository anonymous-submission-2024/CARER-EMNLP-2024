import pickle
import pandas as pd
from datetime import timedelta
import openai
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
# Set up OpenAI API key
openai.api_key = 'insert-openai-api-key'

# Load the retrieved documents and patient data
with open('data/mimic3/retrieved_documents.pkl', 'rb') as f:
    retrieved_documents = pickle.load(f)

with open('data/mimic3/all_patient_data.pkl.pkl', 'rb') as f:
    patient_dict = pickle.load(f)

# Function to generate reasoning using GPT-3.5-turbo
def generate_reasoning(patient_data, retrieved_docs):
    input_text = f"Demographic: Patient is a {patient_data['gender']}.\n"
    input_text += f"At the first visit, the patient was {patient_data['age']} years old.\n"
    input_text += "The following is this patient’s diagnosis and treatment history:\n"

    visit_level_reasoning = ""
    for visit_key, visit_data in patient_data['visit'].items():
        visit_level_reasoning += f"- {visit_key}:\n"
        visit_level_reasoning += f"  ICD Diagnoses: {', '.join(visit_data[f'{visit_key}_icd_diagnoses'])}\n"
        visit_level_reasoning += "  Lab Values:\n"
        for lab_value in visit_data[f'{visit_key}_lab_values']:
            visit_level_reasoning += f"    - {lab_value[0]}: {lab_value[1]} ({lab_value[2]})\n"
        visit_level_reasoning += "\n"

    input_text += visit_level_reasoning

    input_text += "Retrieved Documents:\n"
    for query, docs in retrieved_docs:
        for doc, score in docs:
            input_text += f"{doc}\n"

    input_text += "\nGenerate reasoning based on the above information:\n\n"

    # Use OpenAI GPT-3.5-turbo to generate the reasoning
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are given a list of hospital admissions information of a patient, sorted by admissions time. "
                    "The information includes the patient’s demographic, diagnoses and lab values and some clinical note segments. "
                    "You are required to summarize the health status and progression of that patient visit by visit in less than 2000 words.\n"
                    "1. Visit-level reasoning: First, analyze the information from each visit separately. Look through their diagnosed diseases and "
                    "summarize the main conditions they are suffering from. Next, take a look at some of the lab values, pay attention to abnormal "
                    "or fluctuating lab values, generate knowledge on the typical range of those lab values, and what the patients' results indicate "
                    "about their health.\n"
                    "2. Progression Reasoning: Afterward, summarize and analyze the health progression of the patient’s in-between visits. Pay attention to "
                    "which types of conditions are persistent, which types of conditions are cured, which types are emerging, and the progression of lab "
                    "values, especially abnormal ones, and generate reasoning on what those progressions mean to their health condition. Finally, draw the "
                    "most important conclusions on the patients' health state.\n"
                    "Structure your answer in the following format:\n"
                    "[Start Visit-level reasoning]\n"
                    "- Visit 1 (Reasoning on visit 1)\n"
                    "- Visit 2 (Reasoning on visit 2)\n"
                    "…\n"
                    "- Visit n (Reasoning on visit n)\n"
                    "[End Visit-level reasoning]\n"
                    "[Start Progression Reasoning]\n"
                    "Persistent Conditions (Reasoning on persistent conditions)\n"
                    "*** Emerging Conditions (Reasoning on emerging conditions)\n"
                    "*** Resolved Conditions (Reasoning on resolved conditions)\n"
                    "*** Lab values progression (Reasoning on lab values progression)\n"
                    "*** Conclusion (Final conclusion) ***\n"
                    "[End Progression Reasoning]"
                )
            },
            {"role": "user", "content": input_text}
        ],
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.7
    )

    reasoning_text = response.choices[0].message['content']
    return reasoning_text

MODEL_NAME = 'yikuan8/Clinical-Longformer'
# model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


all_reasoning = {}
all_tokenized = {}
for patient_id, patient_data in tqdm(patient_dict.items()):
    retrieved_docs = retrieved_documents.get(patient_id, [])
    reasoning = generate_reasoning(patient_data, retrieved_docs)
    all_reasoning[patient_id] = reasoning
    # Save the reasoning output to a file
    with open('data/mimic3/reasoning_output.pkl', 'wb') as f:
        pickle.dump(all_reasoning, f)

    # Tokenize the reasoning text
    inputs =  tokenizer.encode_plus(
            reasoning,
            add_special_tokens=True,
            max_length=4096,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )# Generate reasoning for each patient and save the output

    all_tokenized[patient_id] = inputs

    # Save the tokenized outputs to a file
    with open('data/mimic3/tokenized_reasoning.pkl', 'wb') as f:
        pickle.dump(all_tokenized, f)

