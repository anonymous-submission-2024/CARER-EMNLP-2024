import pandas as pd
import numpy as np
import pickle
import json
from collections import OrderedDict
from tqdm import tqdm
from datetime import datetime

from openai import OpenAI
import openai
# client = OpenAI()

openai.api_key = ''
import os
os.environ["OPENAI_API_KEY"] = ''


def to_standard_icd9(code):    
    code = str(code)
    if code == '':
        return code
    split_pos = 4 if code.startswith('E') else 3
    icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
    return icd9_code


with open('data/mimic3/parsed/patient_admission.pkl','rb') as f:
    patient_admission = pickle.load(f)

with open('data/mimic3/parsed/admission_codes.pkl','rb') as f:
    admission_codes = pickle.load(f)

admission_df = pd.read_csv('data/mimic3/raw/ADMISSIONS.csv.gz')

patient_df = pd.read_csv('data/mimic3/raw/PATIENTS.csv.gz')
patient_df = patient_df.merge(admission_df,how='left',on='SUBJECT_ID')

icd_df = pd.read_csv('data/mimic3/raw/D_ICD_DIAGNOSES.csv.gz')
add_icd =  pd.read_csv('add_icd.csv')

diagnoses = pd.read_csv('/mnt/henryng/dungnt/Early-Disease-Prediction/data/mimic3/raw/DIAGNOSES_ICD.csv.gz')

admission_ids = diagnoses['HADM_ID'].unique()
admission_codes = {}
for id in admission_ids:
    # print(id)
    adm_rows = diagnoses[diagnoses['HADM_ID'] == id]
    all_adm_icds = adm_rows['ICD9_CODE'].tolist()
    admission_codes[id] = all_adm_icds


print('started processing lab df')
lab_df = pd.read_csv('data/mimic3/raw/LABEVENTS.csv.gz')
d_lab_items = pd.read_csv('data/mimic3/raw/D_LABITEMS.csv.gz')
lab_df_merge = lab_df.merge(d_lab_items,on = 'ITEMID', how= 'left')
lab_df_filter = lab_df_merge[lab_df_merge['SUBJECT_ID'].isin(list(patient_admission.keys()))]
lab_df_filter[lab_df_filter['LABEL'] == 'Glucose'].columns

lab_values = [50809,50931,51084,51478,50852,50822,50971,50824,50983,50862,50912]
lab_df_final = lab_df_filter[lab_df_filter['ITEMID'].isin(lab_values)]
lab_df_final = lab_df_final.dropna(subset=['HADM_ID'])
lab_df_final = lab_df_final.sort_values(by=['SUBJECT_ID','CHARTTIME'])

lab_df_final['FLAG'] = lab_df_final['FLAG'].fillna('normal')

print('finished processing lab df')


patient_progression = OrderedDict()
patient_meta = {}
patient_info =  {}
i = 0
for patient_id in tqdm(patient_admission):
    
    patient_row = patient_df[patient_df['SUBJECT_ID'] == patient_id]


    # patient
    gender = patient_row['GENDER'].tolist()[0]
    dob = patient_row['DOB'].tolist()[0]
    dob = datetime.strptime(dob,'%Y-%m-%d %H:%M:%S')

    patient_meta[patient_id] = {'gender':gender, 'dob':dob}
    # print(gender,dob)
    admissions = patient_admission[patient_id]
    patient_info[patient_id] = {}
    for admission in admissions:
        admission_info = {}
        admission_id, admission_time = admission['adm_id'], str(admission['adm_time'])
        # print(type(dob), type(admission_time))
        # print(dob, admission_time)
        admission_time = datetime.strptime(admission_time,'%Y-%m-%d %H:%M:%S')
        # print(dob)
        current_age = (admission_time - dob).days // 365

        if current_age > 95:
            current_age = 95
        if current_age == 0:
            current_age = 1

# totalDays = 120

# years = totalDays//365
        # print(current_age)
        # print(admission_time)
        admission_icd_list = admission_codes[admission_id]
        icd_diagnoses = []
        for icd_code in admission_icd_list:
            # print(icd_code)
            try:
                icd_title = icd_df[icd_df['ICD9_CODE']==icd_code]['LONG_TITLE'].tolist()[0].lower()
                # print(icd_title)
            except:
                # print(icd_df[iccd_df['ICD9_CODE']==icd_code])
                # print(icd_code)
                try:
                    icd_title = add_icd[add_icd['ICD9_CODE']==icd_code]['LONG_TITLE'].tolist()[0].lower()
                except:
                    print('can not get title for: ', icd_code)

            icd_diagnoses.append(icd_title)        
        # print(admission_icd)
        admission_rows = admission_df[admission_df['HADM_ID'] == admission_id]
        race = str(admission_rows['ETHNICITY'].tolist()[0]).lower()
        patient_meta[patient_id]['race'] = race

        adm_lab_values = lab_df_final[lab_df_final['HADM_ID'] == admission_id]
        # print(adm_lab_values)
        lab_values_lists = []

        for _,row in adm_lab_values.iterrows():
            # print(row)
            fluid, label, value, value_metric, abnormal = row['FLUID'], row['LABEL'], row['VALUENUM'], row['VALUEUOM'], row['FLAG']
            space = ' '
            row_string = f'{fluid} {label} is {value} {value_metric}, considered as {abnormal}'
            lab_values_lists.append(row_string)
            # print(row_string)
        # print(lab_values)
        admission_info[admission_id] = {'age':current_age, 'diagnosis': icd_diagnoses ,'lab_values':lab_values_lists}
        patient_info[patient_id][admission_id] = {'age':current_age, 'diagnosis': icd_diagnoses ,'lab_values':lab_values_lists}
    # patient_main[patient]
        # admission_in
        # print(len(admission_rows), race, marital_status)
        # current_date = admission_rows
        # break
    if i == 20:
        break
    i += 1
        # print(admission_rows)
        # break

        # print(admission_rows)
        # print(admission_id)
        # pati
        # print(admission)
        # pass
        # print(admission)
    # print(admission)
    # print(patient_id)


initial_prompt = """You are given a list of hospital admissions information of a patient, sorted by admissions time. 
The information includes the patients demographic, diagnoses and lab values. 
Summarize the health status and progression of that patient. 
Afterwards, extract the information that might suggest future risk of diabetes 
(if there are not any, answer Diabetes:No Symptom) \n"""

count = 0

# gpt_results = {}
client = OpenAI()

if not os.path.exists('data/mimic3/gpt_reasoning_1401.json'):
    gpt_results = {}
else:
    with open('data/mimic3/gpt_reasoning_1401.json') as f:
        gpt_results = json.load(f)


count_inference = len(gpt_results)
count = 0
for patient_id in tqdm(patient_info.keys()):
    
    if count >= count_inference:
        prompt = ''
        gender, race =patient_meta[patient_id]['gender'], patient_meta[patient_id]['race']
        gender_translate = {'F':'female','M':'male','f':'female','m':'male'}
        gender = gender_translate[gender]
        pronoun = {'female':'she', 'male':'he'}

        demographic_prompt = f'\nDemographic: Patient is a {race} {gender}.'

        prompt += demographic_prompt

        # Throughout the treatment process, he is diagnosed with the following diseases
        diagnoses_prompt = '\nThe following is this patients diagnosis and treatment history: \n'
        prompt += diagnoses_prompt
        patient_dict = patient_info[patient_id]

        adm_count = 0
        for adm_id in patient_dict.keys():
            adm_count +=1
            age, diagnoses, lab_values = patient_dict[adm_id]['age'], patient_dict[adm_id]['diagnosis'], patient_dict[adm_id]['lab_values']
            
            if len(lab_values) > 10:
                lab_values = lab_values[-10:-1]
            diagnoses_string = '\n +'.join(diagnoses)
            lab_values_string = '\n +'.join(lab_values)
            prompt += f'- Visit {adm_count}, patient was {age} years old. \n'
            prompt += f'{pronoun[gender]} was diagnosed with the following: \n +{diagnoses_string} \n'
            prompt += f'{pronoun[gender]} had the following blood test results:\n +{lab_values_string} \n \n'

        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
            "role": "system",
            "content": initial_prompt
            },
            {
            "role": "user",
            "content": prompt
            },
        ],
        temperature=1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        answer = response.choices[0].message.content
        gpt_results[patient_id] = answer

        with open('data/mimic3/gpt_reasoning.json','w') as f:
            json.dump(gpt_results,f)
        # print(answer)
    count +=1
