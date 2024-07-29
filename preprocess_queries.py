import pandas as pd
from datetime import timedelta

# Load the CSV files into DataFrames
raw_path = 'data/mimic3/raw'
admissions = pd.read_csv(raw_path + '/ADMISSIONS.csv.gz', compression='gzip')
patients = pd.read_csv(raw_path + '/PATIENTS.csv.gz', compression='gzip')
diagnoses_icd = pd.read_csv(raw_path + '/DIAGNOSES_ICD.csv.gz', compression='gzip')
labevents = pd.read_csv(raw_path +'/LABEVENTS.csv.gz', compression='gzip')
noteevents = pd.read_csv(raw_path +'/NOTEEVENTS.csv.gz', compression='gzip')
d_icd_diagnoses = pd.read_csv(raw_path + '/D_ICD_DIAGNOSES.csv.gz', compression='gzip')

# Dictionary for lab value rules
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

# Function to determine lab value comment based on rules
def get_lab_value_comment(value, rules, obs_name):
    if 'low' in rules and value < rules['low']:
        return f"low {obs_name} {value} {rules['unit']}"
    if 'high' in rules and value > rules['high']:
        return f"high {obs_name} {value} {rules['unit']}"
    if 'normal' in rules and rules['normal'][0] <= value <= rules['normal'][1]:
        return f"normal {obs_name} {value} {rules['unit']}"
    if 'prediabetic' in rules and rules['prediabetic'][0] <= value <= rules['prediabetic'][1]:
        return f"prediabetic {obs_name} {value} {rules['unit']}"
    if 'diabetic' in rules and value > rules['diabetic']:
        return f"diabetic {obs_name} {value} {rules['unit']}"
    if 'bradycardia' in rules and value < rules['bradycardia']:
        return f"bradycardia {obs_name} {value} {rules['unit']}"
    if 'tachycardia' in rules and value > rules['tachycardia']:
        return f"tachycardia {obs_name} {value} {rules['unit']}"
    return f"abnormal {obs_name} {value} {rules['unit']}"

# Function to get patient data excluding the last visit
def get_patient_data(patient_id):
    # Extract demographic information
    patient_info = patients[patients['SUBJECT_ID'] == patient_id]
    if not patient_info.empty:
        gender = patient_info.iloc[0]['GENDER']
        dob = pd.to_datetime(patient_info.iloc[0]['DOB'])
    else:
        gender, dob = None, None
    
    # Extract admission details to determine visits
    admission_info = admissions[admissions['SUBJECT_ID'] == patient_id]
    if not admission_info.empty:
        admission_info['ADMITTIME'] = pd.to_datetime(admission_info['ADMITTIME'])
        admission_info = admission_info.sort_values(by='ADMITTIME')
        last_admission = admission_info.iloc[-1]
        admission_info = admission_info.iloc[:-1]  # Exclude the last admission
        excluded_hadm_id = last_admission['HADM_ID']
    else:
        excluded_hadm_id = None

    # Extract ICD diagnoses excluding the last visit
    if excluded_hadm_id:
        diagnoses = diagnoses_icd[(diagnoses_icd['SUBJECT_ID'] == patient_id) & (diagnoses_icd['HADM_ID'] != excluded_hadm_id)]
    else:
        diagnoses = diagnoses_icd[diagnoses_icd['SUBJECT_ID'] == patient_id]
    
    # Map ICD9_CODE to long title
    icd_mapping = d_icd_diagnoses.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()
    
    # Organize ICD codes by visit
    icd_by_visit = {}
    for i, (hadm_id, group) in enumerate(diagnoses.groupby('HADM_ID'), start=1):
        icd_codes = group['ICD9_CODE'].map(icd_mapping).tolist()
        icd_by_visit[f'visit_{i}_icd_diagnoses'] = icd_codes

    # Extract lab values and group by hadm_id
    if excluded_hadm_id:
        lab_values = labevents[(labevents['SUBJECT_ID'] == patient_id) & (labevents['HADM_ID'] != excluded_hadm_id)]
    else:
        lab_values = labevents[labevents['SUBJECT_ID'] == patient_id]

    lab_values['CHARTTIME'] = pd.to_datetime(lab_values['CHARTTIME'])
    lab_values_grouped = []
    for hadm_id, group in lab_values.groupby('HADM_ID'):
        hadm_admission_time = admission_info[admission_info['HADM_ID'] == hadm_id]['ADMITTIME'].iloc[0]
        for _, row in group.iterrows():
            item_id = row['ITEMID']
            value = row['VALUENUM']
            unit = row['VALUEUOM']
            charttime = row['CHARTTIME']
            for obs_name, rules in lab_value_rules.items():
                if obs_name in item_id and unit == rules['unit']:
                    query = get_lab_value_comment(value, rules, obs_name)
                    lab_values_grouped.append((hadm_id, obs_name, value, query))
                    break
            else:
                # If lab value has no hadm_id, group it with closest admission time within 1 week
                closest_adm = admission_info.loc[(admission_info['ADMITTIME'] - charttime).abs() < timedelta(weeks=1)]
                if not closest_adm.empty:
                    closest_hadm_id = closest_adm.iloc[0]['HADM_ID']
                    for obs_name, rules in lab_value_rules.items():
                        if obs_name in item_id and unit == rules['unit']:
                            query = get_lab_value_comment(value, rules, obs_name)
                            lab_values_grouped.append((closest_hadm_id, obs_name, value, query))
                            break

    lab_values_by_visit = {}
    for hadm_id, group in pd.DataFrame(lab_values_grouped, columns=['HADM_ID', 'OBS_NAME', 'VALUE', 'QUERY']).groupby('HADM_ID'):
        visit_number = list(icd_by_visit.keys()).index(f'visit_{hadm_id}_icd_diagnoses') + 1
        visit_key = f'visit_{visit_number}_lab_values'
        lab_values_by_visit[visit_key] = group[['OBS_NAME', 'VALUE', 'QUERY']].values.tolist()

    # Extract clinical notes and concatenate them excluding the last visit
    if excluded_hadm_id:
        clinical_notes = noteevents[(noteevents['SUBJECT_ID'] == patient_id) & (noteevents['HADM_ID'] != excluded_hadm_id)]
    else:
        clinical_notes = noteevents[noteevents['SUBJECT_ID'] == patient_id]
    clinical_notes_concat = ' '.join(clinical_notes['TEXT'].fillna(''))
    
    # Calculate age at the time of the first admission
    if not admission_info.empty:
        admittime = pd.to_datetime(admission_info.iloc[0]['ADMITTIME'])
        age = (admittime - dob).days / 365.25 if dob is not None else None
    else:
        age = None
    
    # Combine results
    patient_data = {
        'gender': gender,
        'age': age,
        'visit': {}
    }
    for visit_key in icd_by_visit.keys():
        visit_number = visit_key.split('_')[1]
        patient_data['visit'][f'visit {visit_number}'] = {
            visit_key: icd_by_visit[visit_key],
            f'visit_{visit_number}_lab_values': lab_values_by_visit.get(f'visit_{visit_number}_lab_values', [])
        }
    patient_data['clinical_notes'] = clinical_notes_concat
    
    return patient_data

# Iterate through each patient in the dataset and store the data in a dictionary
all_patient_data = {}
for patient_id in patients['SUBJECT_ID'].unique():
    all_patient_data[patient_id] = get_patient_data(patient_id)

import pickle
# Save the dictionary to a pkl file
with open('data/mimic3/all_patient_data.pkl', 'wb') as file:
    pickle.dump(all_patient_data, file)