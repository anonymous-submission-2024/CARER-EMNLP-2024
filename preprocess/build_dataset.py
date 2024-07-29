import numpy as np

from preprocess.parse_csv import EHRParser
import math

def split_patients(patient_admission, admission_codes, code_map, train_num, test_num, seed=6669):
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission[EHRParser.adm_id_col]]
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    valid_num = len(patient_admission) - train_num - test_num
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids


def build_code_xy(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num):
    n = len(pids)
    x = np.zeros((n, max_admission_num, code_num), dtype=bool)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n,), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            x[i, k, codes] = 1
        codes = np.array(admission_codes_encoded[admissions[-1][EHRParser.adm_id_col]])
        y[i, codes] = 1
        lens[i] = len(admissions) - 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens

def build_note_x(pids: np.ndarray,
                 patient_note_encoded: dict,
                 max_word_num_in_a_note: int) -> (np.ndarray, np.ndarray):
    print('building train/valid/test notes features and labels ...')
    n = len(pids)
    x = np.zeros((n, max_word_num_in_a_note), dtype=int)
    lens = np.zeros((n, ), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        note = patient_note_encoded[pid]
        length = max_word_num_in_a_note if max_word_num_in_a_note < len(note) else len(note)
        x[i][:length] = note[:length]
        lens[i] = length
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, lens


def build_note_x(pids: np.ndarray,
                 patient_note_encoded: dict,
                 max_word_num_in_a_note: int) -> (np.ndarray, np.ndarray):
    print('building train/valid/test notes features and labels ...')
    n = len(pids)
    x = np.zeros((n, max_word_num_in_a_note), dtype=int)
    lens = np.zeros((n, ), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        note = patient_note_encoded[pid]
        length = max_word_num_in_a_note if max_word_num_in_a_note < len(note) else len(note)
        print(note,length)
        x[i][:length] = note[:length]
        lens[i] = length
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, lens

def build_note_x_bert(pids: np.ndarray,
                 patient_note_encoded: dict,
                 max_word_num_in_a_note: int) -> (np.ndarray, np.ndarray):
    print('building train/valid/test notes features and labels ...')
    n = len(pids)
    x = np.zeros((n, max_word_num_in_a_note), dtype=int)
    masks = np.zeros((n, max_word_num_in_a_note), dtype=int)
    lens = np.zeros((n, ), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        note, mask = patient_note_encoded[pid]['input_ids'], patient_note_encoded[pid]['attention_mask']
        # # length = max_word_num_in_a_note if max_word_num_in_a_note < len(note) else len(note)
        # # print(note,length)
        # length = max_word_num_in_a_note
        x[i] = note
        masks[i] = mask
        # lens[i] = length
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, masks


def calculate_tf_idf(note_encoded: dict, word_num: int) -> dict:
    n_docs = len(note_encoded)
    tf = dict()
    df = np.zeros((word_num + 1, ), dtype=np.int64)
    print('calculating tf and df ...')
    for i, (pid, note) in enumerate(note_encoded.items()):
        print('\r\t%d / %d' % (i + 1, n_docs), end='')
        note_tf = dict()
        for word in note:
            note_tf[word] = note_tf.get(word, 0) + 1
        wset = set(note)
        for word in wset:
            df[word] += 1
        tf[pid] = note_tf
    print('\r\t%d / %d patients' % (n_docs, n_docs))
    print('calculating tf_idf ...')
    tf_idf = dict()
    for i, (pid, note) in enumerate(note_encoded.items()):
        print('\r\t%d / %d patients' % (i + 1, n_docs), end='')
        note_tf = tf[pid]
        note_tf_idf = [note_tf[word] / len(note) * (math.log(n_docs / (1 + df[word]), 10) + 1)
                      for word in note]
        tf_idf[pid] = note_tf_idf
    print('\r\t%d / %d patients' % (n_docs, n_docs))
    return tf_idf


def build_tf_idf_weight(pids: np.ndarray, note_x: np.ndarray, note_encoded: dict, word_num: int) -> np.ndarray:
    print('build tf_idf for notes ...')
    tf_idf = calculate_tf_idf(note_encoded, word_num)
    weight = np.zeros_like(note_x, dtype=float)
    for i, pid in enumerate(pids):
        note_tf_idf = tf_idf[pid]
        weight[i][:len(note_tf_idf)] = note_tf_idf
    weight = weight / weight.sum(axis=-1, keepdims=True)
    return weight


def build_heart_failure_y(hf_prefix, codes_y, code_map):
    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map),), dtype=int)
    hfs[hf_list] = 1
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y
