import os
import _pickle as pickle

from preprocess import save_sparse, save_data, save_data_notes
from preprocess.parse_csv import Mimic3Parser, Mimic4Parser, EICUParser
from preprocess.encode import encode_code, encode_note_train, encode_note_test
from preprocess.build_dataset import split_patients, build_code_xy, build_heart_failure_y, build_note_x
from preprocess.auxiliary import generate_code_code_adjacent, generate_neighbors, normalize_adj, divide_middle, generate_code_levels,generate_code_and_target_prior

import argparse
def read_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='Random seed', type=int, default=42)
    parser.add_argument('--dataset', help='name of dataset', type=str, default='mimic3') #mimic3 or mimic4
    parser.add_argument('--max_note_len', help='max length of clinical note', type=int, default=5000) #mimic3 or mimic4
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    conf = {
        'mimic3': {
            'parser': Mimic3Parser,
            'train_num':3600,
            'test_num': 1850,
            'threshold': 0.01
        },
        'mimic4': {
            'parser': Mimic4Parser,
            'train_num': 14000,
            'test_num': 3000,
            'threshold': 0.01,
            'sample_num': 20000
        },
        'eicu': {
            'parser': EICUParser,
            'train_num': 8000,
            'test_num': 1000,
            'threshold': 0.01
        }
    }

    max_note_len = 50000
    from_saved = False
    data_path = 'data'
    dataset = 'mimic3'  # mimic3, eicu, or mimic4
    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    if dataset == 'mimic4':
        raw_path = os.path.join(raw_path,'2.2/hosp')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/%s/raw`' % dataset)
        exit()
    parsed_path = os.path.join(dataset_path, 'parsed')
    if from_saved:
        patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
        admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
    else:
        parser = conf[dataset]['parser'](raw_path)
        sample_num = conf[dataset].get('sample_num', None)
        patient_admission, patient_note, admission_codes = parser.parse_with_notes(sample_num)
        print('saving parsed data ...')
        if not os.path.exists(parsed_path):
            os.makedirs(parsed_path)
        pickle.dump(patient_admission, open(os.path.join(parsed_path, 'patient_admission.pkl'), 'wb'))
        pickle.dump(patient_note, open(os.path.join(parsed_path, 'patient_note.pkl'), 'wb'))

        pickle.dump(admission_codes, open(os.path.join(parsed_path, 'admission_codes.pkl'), 'wb'))

    patient_num = len(patient_admission)
    max_admission_num = max([len(admissions) for admissions in patient_admission.values()])
    avg_admission_num = sum([len(admissions) for admissions in patient_admission.values()]) / patient_num
    max_visit_code_num = max([len(codes) for codes in admission_codes.values()])
    avg_visit_code_num = sum([len(codes) for codes in admission_codes.values()]) / len(admission_codes)
    print('patient num: %d' % patient_num)
    print('max admission num: %d' % max_admission_num)
    print('mean admission num: %.2f' % avg_admission_num)
    print('max code num in an admission: %d' % max_visit_code_num)
    print('mean code num in an admission: %.2f' % avg_visit_code_num)

    print('encoding code ...')
    admission_codes_encoded, code_map = encode_code(patient_admission, admission_codes)
    code_num = len(code_map)
    print('There are %d codes' % code_num)

    code_levels = generate_code_levels(data_path, code_map)
    pickle.dump({
        'code_levels': code_levels,
    }, open(os.path.join(parsed_path, 'code_levels.pkl'), 'wb'))

    train_pids, valid_pids, test_pids = split_patients(
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=code_map,
        train_num=conf[dataset]['train_num'],
        test_num=conf[dataset]['test_num']
    )
    
    print(len(train_pids), len(valid_pids), len(test_pids))
    # train_note_encoded, dictionary = encode_note_train(patient_note, train_pids, max_note_len=max_note_len)
    train_note_encoded, dictionary = encode_note_train(patient_note, train_pids, max_note_len=max_note_len)
    # print(train_note_encoded.shape)
    valid_note_encoded = encode_note_test(patient_note, valid_pids, dictionary, max_note_len=max_note_len)
    test_note_encoded = encode_note_test(patient_note, test_pids, dictionary, max_note_len=max_note_len)

    def max_word_num(note_encoded: dict) -> int:
        return max(len(note) for note in note_encoded.values())

    max_word_num_in_a_note = max([max_word_num(train_note_encoded), max_word_num(valid_note_encoded), max_word_num(test_note_encoded)])
    print('max word num in a note:', max_word_num_in_a_note)

    print('There are %d train, %d valid, %d test samples' % (len(train_pids), len(valid_pids), len(test_pids)))
    code_adj = generate_code_code_adjacent(pids=train_pids, patient_admission=patient_admission,
                                           admission_codes_encoded=admission_codes_encoded,
                                           code_num=code_num, threshold=conf[dataset]['threshold'])
    print('code_adj',code_adj.shape)
    
    
    diabetes_prior = generate_code_and_target_prior(
                                         target_prefix = '250',
                                         code_map = code_map,
                                         code_adj = code_adj)

    hf_prior = generate_code_and_target_prior(
                                         target_prefix = '428',
                                         code_map = code_map,
                                         code_adj = code_adj)


    common_args = [patient_admission, admission_codes_encoded, max_admission_num, code_num]
    print('building train codes features and labels ...')
    (train_code_x, train_codes_y, train_visit_lens) = build_code_xy(train_pids, *common_args)
    print('building valid codes features and labels ...')
    (valid_code_x, valid_codes_y, valid_visit_lens) = build_code_xy(valid_pids, *common_args)
    print('building test codes features and labels ...')
    (test_code_x, test_codes_y, test_visit_lens) = build_code_xy(test_pids, *common_args)

    print('generating train neighbors ...')
    train_neighbors = generate_neighbors(train_code_x, train_visit_lens, code_adj)
    print('generating valid neighbors ...')
    valid_neighbors = generate_neighbors(valid_code_x, valid_visit_lens, code_adj)
    print('generating test neighbors ...')
    test_neighbors = generate_neighbors(test_code_x, test_visit_lens, code_adj)

    print('generating train middles ...')
    train_divided = divide_middle(train_code_x, train_neighbors, train_visit_lens)
    print('generating valid middles ...')
    valid_divided = divide_middle(valid_code_x, valid_neighbors, valid_visit_lens)
    print('generating test middles ...')
    test_divided = divide_middle(test_code_x, test_neighbors, test_visit_lens)

    print('building note data for training')
    train_note_x, train_note_lens = build_note_x(train_pids, train_note_encoded, max_word_num_in_a_note)
    valid_note_x, valid_note_lens = build_note_x(valid_pids, valid_note_encoded, max_word_num_in_a_note)
    test_note_x, test_note_lens = build_note_x(test_pids, test_note_encoded, max_word_num_in_a_note)
    
    print('building train heart failure labels ...')
    train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
    print('building valid heart failure labels ...')
    valid_hf_y = build_heart_failure_y('428', valid_codes_y, code_map)
    print('building test heart failure labels ...')
    test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)


    print('building train diabetes labels ...')
    train_diabetes_y = build_heart_failure_y('250', train_codes_y, code_map)
    print('building valid diabetes labels ...')
    valid_diabetes_y = build_heart_failure_y('250', valid_codes_y, code_map)
    print('building test diabetes labels ...')
    test_diabetes_y = build_heart_failure_y('250', test_codes_y, code_map)


    print('building pneumonia labels ...')
    train_diabetes_y = build_heart_failure_y('48', train_codes_y, code_map)
    print('building valid diabetes labels ...')
    valid_diabetes_y = build_heart_failure_y('48', valid_codes_y, code_map)
    print('building test diabetes labels ...')
    test_diabetes_y = build_heart_failure_y('250', test_codes_y, code_map)


    encoded_path = os.path.join(dataset_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))

    pickle.dump({
        'train_note_encoded': train_note_encoded,
        'valid_note_encoded': valid_note_encoded,
        'test_note_encoded': test_note_encoded
    }, open(os.path.join(encoded_path,'note_encoded.pkl'), 'wb'))

    pickle.dump({
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))




    print('saving standard data ...')
    standard_path = os.path.join(dataset_path, 'standard')
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    test_path = os.path.join(standard_path, 'test')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(valid_path)
        os.makedirs(test_path)

    pickle.dump(dictionary, open(os.path.join(standard_path, 'note_dictionary.pkl'), 'wb'))

    print('\tsaving training data')
    save_data_notes(train_path, train_code_x, train_visit_lens, train_codes_y, train_hf_y,train_diabetes_y, train_divided, train_neighbors,train_note_x, train_note_lens)
    print('\tsaving valid data')
    save_data_notes(valid_path, valid_code_x, valid_visit_lens, valid_codes_y, valid_hf_y,valid_diabetes_y, valid_divided, valid_neighbors,valid_note_x, valid_note_lens)
    print('\tsaving test data')
    save_data_notes(test_path, test_code_x, test_visit_lens, test_codes_y, test_hf_y,test_diabetes_y, test_divided, test_neighbors, test_note_x, test_note_lens)

    code_adj = normalize_adj(code_adj)
    save_sparse(os.path.join(standard_path, 'code_adj'), code_adj)
    save_sparse(os.path.join(standard_path, 'diabetes_prior'), diabetes_prior)
    save_sparse(os.path.join(standard_path, 'hf_prior'), hf_prior)
