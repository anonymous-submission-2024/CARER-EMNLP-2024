import os

import numpy as np


def save_sparse(path, x):
    idx = np.where(x > 0)
    values = x[idx]
    np.savez(path, idx=idx, values=values, shape=x.shape)


def load_sparse(path):
    data = np.load(path)
    idx, values = data['idx'], data['values']
    mat = np.zeros(data['shape'], dtype=values.dtype)
    mat[tuple(idx)] = values
    return mat

def save_data(path, code_x, visit_lens, codes_y, hf_y,diabetes_y, divided, neighbors):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'code_y'), codes_y)
    np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    np.savez(os.path.join(path, 'diabetes_y'), diabetes_y=diabetes_y)
    # np.savez(os.path.join(path,'note_x'),note_x = note_x)
    # np.savez(os.path.join(path,'note_lens'),note_lens = note_lens)

    save_sparse(os.path.join(path, 'divided'), divided)
    save_sparse(os.path.join(path, 'neighbors'), neighbors)



def save_data_notes(path, code_x, visit_lens, codes_y, hf_y,diabetes_y, divided, neighbors,note_x, note_lens):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'code_y'), codes_y)
    np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    np.savez(os.path.join(path, 'diabetes_y'), diabetes_y=diabetes_y)
    np.savez(os.path.join(path,'note_x'),note_x = note_x)
    np.savez(os.path.join(path,'note_lens'),note_lens = note_lens)

    save_sparse(os.path.join(path, 'divided'), divided)
    save_sparse(os.path.join(path, 'neighbors'), neighbors)



def save_data_notes_bert(path, code_x, visit_lens, codes_y, hf_y,diabetes_y, divided, neighbors,note_x, note_attention_mask):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'code_y'), codes_y)
    np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    np.savez(os.path.join(path, 'diabetes_y'), diabetes_y=diabetes_y)
    np.savez(os.path.join(path,'note_x_bert'),note_x = note_x)
    np.savez(os.path.join(path,'note_attention_mask'),note_attention_mask = note_attention_mask)

    save_sparse(os.path.join(path, 'divided'), divided)
    save_sparse(os.path.join(path, 'neighbors'), neighbors)



