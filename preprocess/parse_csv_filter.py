import os
from datetime import datetime
from collections import OrderedDict

import pandas
import pandas as pd
import numpy as np


class EHRParser:
    pid_col = 'pid'
    adm_id_col = 'adm_id'
    adm_time_col = 'adm_time'
    cid_col = 'cid'

    def __init__(self, path):
        self.path = path

        self.skip_pid_check = False

        self.patient_admission = None
        self.admission_codes = None
        self.admission_procedures = None
        self.admission_medications = None

        self.parse_fn = {'d': self.set_diagnosis}

    def set_admission(self):
        raise NotImplementedError

    def set_diagnosis(self):
        raise NotImplementedError

    @staticmethod
    def to_standard_icd9(code: str):
        raise NotImplementedError

    def parse_admission(self):
        print('parsing the csv file of admission ...')
        filename, cols, converters = self.set_admission()
        admissions = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        admissions = self._after_read_admission(admissions, cols)
        all_patients = OrderedDict()
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(admissions)), end='')
            pid, adm_id, adm_time = row[cols[self.pid_col]], row[cols[self.adm_id_col]], row[cols[self.adm_time_col]]
            if pid not in all_patients:
                all_patients[pid] = []
            admission = all_patients[pid]
            admission.append({self.adm_id_col: adm_id, self.adm_time_col: adm_time})
        print('\r\t%d in %d rows' % (len(admissions), len(admissions)))

        patient_admission = OrderedDict()
        for pid, admissions in all_patients.items():
            if len(admissions) >= 2:
                patient_admission[pid] = sorted(admissions, key=lambda admission: admission[self.adm_time_col])

        self.patient_admission = patient_admission

    def _after_read_admission(self, admissions, cols):
        return admissions

    def _parse_concept(self, concept_type):
        assert concept_type in self.parse_fn.keys()
        filename, cols, converters = self.parse_fn[concept_type]()
        concepts = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        concepts = self._after_read_concepts(concepts, concept_type, cols)
        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id, code = row[cols[self.adm_id_col]], row[cols[self.cid_col]]
                if code == '':
                    continue
                if adm_id not in result:
                    result[adm_id] = []
                codes = result[adm_id]
                codes.append(code)
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        return result

    def _after_read_concepts(self, concepts, concept_type, cols):
        return concepts

    def parse_diagnoses(self):
        print('parsing csv file of diagnosis ...')
        self.admission_codes = self._parse_concept('d')

    def parse_notes(self, use_summary=False) -> dict:
        patient_admission = self.patient_admission
        # print(self.patient_admission)
        print('parsing NOTEEVENTS.csv ...')
        notes_path = os.path.join(self.path, 'NOTEEVENTS.csv.gz')
        print("reading noteevents")
        notes = pd.read_csv(
            notes_path,
            usecols=['HADM_ID', 'TEXT', 'CATEGORY'],
            converters={'HADM_ID': lambda x: int(x) if x != '' else -1, 'TEXT': str, 'CATEGORY': str}
        )
        print('finished reading noteevents')
        patient_note = OrderedDict()
        # for i, (pid, admissions) in enumerate(patient_admission.items()):
        for i, (pid, admissions) in enumerate(patient_admission.items()):
            # if i == 10: 
            #     break
            print('\r\t%d in %d patients' % (i + 1, len(patient_admission)), end='')
            admission_id = admissions[-1]['adm_id']
            if use_summary:
                note = [row['TEXT'] for _, row in notes[notes['HADM_ID'] == admission_id].iterrows()
                        if row['CATEGORY'] == 'Discharge summary']
            else:
                # note = notes[notes['HADM_ID'] == admission_id]['TEXT'].tolist()
                note = [row['TEXT'] for _, row in notes[notes['HADM_ID'] == admission_id].iterrows()
                        if row['CATEGORY'] != 'Discharge summary']
            note = ' '.join(note)
            if len(note) > 0:
                patient_note[pid] = note
        print('\r\t%d in %d patients' % (len(patient_admission), len(patient_admission)))
        self.patient_note = patient_note
        # print(self.patient_note.keys())
        # print('Patient note', self.patient_note)
        print("Done patient note")
        return patient_note
    

    def calibrate_patient_by_notes(self):
        patient_admission = self.patient_admission
        admission_codes = self.admission_codes
        patient_note = self.patient_note
        print('calibrating patients by notes ...')
        del_pids = [pid for pid in patient_admission if pid not in patient_note]
        print("number of patients w/o notes: ",del_pids)
        for pid in del_pids:
            # print('\tpatient %d doesn\'t have notes' % pid)
            admissions = patient_admission[pid]
            for admission in admissions:
                del admission_codes[admission['adm_id']]
            del patient_admission[pid]

    def calibrate_patient_by_admission(self):
        print('calibrating patients by admission ...')
        del_pids = []
        for pid, admissions in self.patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.admission_codes:
                    break
            else:
                continue
            del_pids.append(pid)
        for pid in del_pids:
            admissions = self.patient_admission[pid]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                for concepts in [self.admission_codes]:
                    if adm_id in concepts:
                        del concepts[adm_id]
            del self.patient_admission[pid]

    def calibrate_admission_by_patient(self):
        print('calibrating admission by patients ...')
        adm_id_set = set()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id_set.add(admission[self.adm_id_col])
        del_adm_ids = [adm_id for adm_id in self.admission_codes if adm_id not in adm_id_set]
        for adm_id in del_adm_ids:
            del self.admission_codes[adm_id]

    def calibrate_patient_by_target_disease(self,target_prefix = '250'):
        print('calibrating patients by target prefix ...')
        del_pids = []

        def check_target_in_visit(visit):
            target_in_visit = False
            for d in visit:
                # print(d)
                if target_prefix in d:
                    print
                    target_in_visit = True
                    break
            return target_in_visit
        count_valid = 0
        count_valid_2 = 0
        count_valid_3 = 0
        for pid, admissions in self.patient_admission.items():
            # print('pid, admissions: ', pid, admissions)
            i = 0

            target_in_first_visits = False
            target_in_later_visits = False
            target_in_third_visits = False
            target_in_4th_visits = False
    
            for admission in admissions:
                # print('admission: ', admission)
                adm_id = admission[self.adm_id_col]
                admission_diagnoses = self.admission_codes[adm_id]

                if i == 0:
                    target_in_first_visits = check_target_in_visit(admission_diagnoses)
                    if target_in_first_visits:
                        del_pids.append(pid)
                    # print(target_in_first_visits)
                elif i  >= 1:
                    if check_target_in_visit(admission_diagnoses):
                        target_in_later_visits  =True
                        if i >=2:
                            target_in_third_visits = True
                        if i >=3:
                            target_in_4th_visits = True

                    # print(target_in_first_visits)
                # print(admission_diagnoses)
    
                i +=1
            if not  target_in_first_visits and target_in_later_visits:
                count_valid +=1
            if not  target_in_first_visits and target_in_third_visits:
                count_valid_2 +=1
            if not  target_in_first_visits and target_in_4th_visits:
                count_valid_3 +=1

        print('Count valid: ' , count_valid)
        print('Count valid 2: ' , count_valid_2)
        print('Count valid 3: ' , count_valid_3)

        for pid in del_pids:
            admissions = self.patient_admission[pid]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                for concepts in [self.admission_codes]:
                    if adm_id in concepts:
                        del concepts[adm_id]
            del self.patient_admission[pid]

    def sample_patients(self, sample_num, seed):
        np.random.seed(seed)
        keys = list(self.patient_admission.keys())
        selected_pids = np.random.choice(keys, sample_num, False)
        self.patient_admission = {pid: self.patient_admission[pid] for pid in selected_pids}
        admission_codes = dict()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                admission_codes[adm_id] = self.admission_codes[adm_id]
        self.admission_codes = admission_codes

    def parse_with_notes(self, sample_num=None, seed=6669):
        self.parse_admission()
        self.parse_diagnoses()
        # self.parse_notes()
        self.calibrate_patient_by_admission()
        self.calibrate_admission_by_patient()
        self.calibrate_patient_by_target_disease(target_prefix='401')
        """"copd (49...): 1450 / 648 / 316 MIMIC III - 4935 / 2845 / 1800 MIMIC IV
            diabetes (250): 504 / 200 / 112 MIMIC III - 2455 / 1554 / 1005 MIMIC IV
            heart failure (428): 1277 / 540 / 257 MIMIC III - 1896 / 1245 / 839 MIMIC IV
            CKD (585): 1079 / 538 / 306 MIMIC III - 2967 / 2041 / 1468 MIMIC IV
            Hypertension(401): 1097 / 396 /171 MIMIC III - 4842 / 2832 / 1853 MIMIC IV"""
        self.calibrate_patient_by_notes()

        if sample_num is not None:
            self.sample_patients(sample_num, seed)
        return self.patient_admission, self.patient_note, self.admission_codes

    def parse(self, sample_num=None, seed=6669):
        self.parse_admission()
        self.parse_diagnoses()
        # self.parse_notes()
        self.calibrate_patient_by_admission()
        self.calibrate_admission_by_patient()
        # self.calibrate_patient_by_notes()
        if sample_num is not None:
            self.sample_patients(sample_num, seed)
        return self.patient_admission,self.admission_codes


class Mimic3Parser(EHRParser):
    def set_admission(self):
        filename = 'ADMISSIONS.csv.gz'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.adm_time_col: 'ADMITTIME'}
        converter = {
            'SUBJECT_ID': int,
            'HADM_ID': int,
            'ADMITTIME': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'DIAGNOSES_ICD.csv.gz'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.cid_col: 'ICD9_CODE'}
        converter = {'SUBJECT_ID': int, 'HADM_ID': int, 'ICD9_CODE': Mimic3Parser.to_standard_icd9}
        return filename, cols, converter

    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code


class Mimic4Parser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.icd_ver_col = 'icd_version'
        self.icd_map = self._load_icd_map()
        self.patient_year_map = self._load_patient()

    def _load_icd_map(self):
        print('loading ICD-10 to ICD-9 map ...')
        filename = 'icd10-icd9.csv'
        cols = ['ICD10', 'ICD9']
        converters = {'ICD10': str, 'ICD9': str}
        icd_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        icd_map = {row['ICD10']: row['ICD9'] for _, row in icd_csv.iterrows()}
        return icd_map

    def _load_patient(self):
        print('loading patients anchor year ...')
        filename = 'patients.csv.gz'
        cols = ['subject_id', 'anchor_year', 'anchor_year_group']
        converters = {'subject_id': int, 'anchor_year': int, 'anchor_year_group': lambda cell: int(str(cell)[:4])}
        patient_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        patient_year_map = {row['subject_id']: row['anchor_year'] - row['anchor_year_group']
                            for i, row in patient_csv.iterrows()}
        return patient_year_map

    def set_admission(self):
        filename = 'admissions.csv.gz'
        cols = {self.pid_col: 'subject_id', self.adm_id_col: 'hadm_id', self.adm_time_col: 'admittime'}
        converter = {
            'subject_id': int,
            'hadm_id': int,
            'admittime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnoses_icd.csv.gz'
        cols = {
            self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            self.cid_col: 'icd_code',
            self.icd_ver_col: 'icd_version'
        }
        converter = {'subject_id': int, 'hadm_id': int, 'icd_code': str, 'icd_version': int}
        return filename, cols, converter

    def _after_read_admission(self, admissions, cols):
        print('\tselecting valid admission ...')
        valid_admissions = []
        n = len(admissions)
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t\t%d in %d rows' % (i + 1, n), end='')
            pid = row[cols[self.pid_col]]
            year = row[cols[self.adm_time_col]].year - self.patient_year_map[pid]
            if year > 2012:
                valid_admissions.append(i)
        print('\r\t\t%d in %d rows' % (n, n))
        print('\t\tremaining %d rows' % len(valid_admissions))
        return admissions.iloc[valid_admissions]

    def _after_read_concepts(self, concepts, concept_type, cols):
        print('\tmapping ICD-10 to ICD-9 ...')
        n = len(concepts)
        if concept_type == 'd':
            def _10to9(i, row):
                if i % 100 == 0:
                    print('\r\t\t%d in %d rows' % (i + 1, n), end='')
                cid = row[cid_col]
                if row[icd_ver_col] == 10:
                    if cid not in self.icd_map:
                        code = self.icd_map[cid + '1'] if cid + '1' in self.icd_map else ''
                    else:
                        code = self.icd_map[cid]
                    if code == 'NoDx':
                        code = ''
                else:
                    code = cid
                return Mimic4Parser.to_standard_icd9(code)

            cid_col, icd_ver_col = cols[self.cid_col], self.icd_ver_col
            col = np.array([_10to9(i, row) for i, row in concepts.iterrows()])
            print('\r\t\t%d in %d rows' % (n, n))
            concepts[cid_col] = col
        return concepts

    @staticmethod
    def to_standard_icd9(code: str):
        return Mimic3Parser.to_standard_icd9(code)


class EICUParser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.skip_pid_check = True

    def set_admission(self):
        filename = 'patient.csv.gz'
        cols = {
            self.pid_col: 'patienthealthsystemstayid',
            self.adm_id_col: 'patientunitstayid',
            self.adm_time_col: 'hospitaladmitoffset'
        }
        converter = {
            'patienthealthsystemstayid': int,
            'patientunitstayid': int,
            'hospitaladmitoffset': lambda cell: -int(cell)
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnosis.csv.gz'
        cols = {self.pid_col: 'diagnosisid', self.adm_id_col: 'patientunitstayid', self.cid_col: 'icd9code'}
        converter = {'diagnosisid': int, 'patientunitstayid': int, 'icd9code': EICUParser.to_standard_icd9}
        return filename, cols, converter

    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        code = code.split(',')[0]
        c = code[0].lower()
        dot = code.find('.')
        if dot == -1:
            dot = None
        if not c.isalpha():
            prefix = code[:dot]
            if len(prefix) < 3:
                code = ('%03d' % int(prefix)) + code[dot:]
            return code
        if c == 'e':
            prefix = code[1:dot]
            if len(prefix) != 3:
                return ''
        if c != 'e' or code[0] != 'v':
            return ''
        return code

    def parse_diagnoses(self):
        super().parse_diagnoses()
        t = OrderedDict.fromkeys(self.admission_codes.keys())
        for adm_id, codes in self.admission_codes.items():
            t[adm_id] = list(set(codes))
        self.admission_codes = t
