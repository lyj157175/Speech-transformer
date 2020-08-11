import os
import pickle
from tqdm import tqdm
from config import wav_folder, transcript_file, pickle_file
from utils import ensure_folder


def get_data(split):
    print('getting {} data...'.format(split))

    global VOCAB
    with open(transcript_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    tran_dict = dict()
    for line in lines:
        tokens = line.split()
        key = tokens[0]
        trn = ''.join(tokens[1:])
        tran_dict[key] = trn   # tran_dict: {'BAC0009123': wav1.wav, ...}

    samples = []

    folder = os.path.join(wav_folder, split)    # data/data_aishell/wav/train
    ensure_folder(folder)    # 确保floder是一个目录
    dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]  # data/data_aishell/wav/train/S0003
    for dir in tqdm(dirs):
        files = [f for f in os.listdir(dir) if f.endswith('.wav')]    # [wav1, wav2, .....]

        for f in files:
            wave = os.path.join(dir, f)  # data/data_aishell/wav/train/S0003/wav1.wav

            key = f.split('.')[0]
            if key in tran_dict:
                trn = tran_dict[key]
                trn = list(trn.strip()) + ['<eos>']

                for token in trn:
                    build_vocab(token)

                trn = [VOCAB[token] for token in trn]

                samples.append({'trn': trn, 'wave': wave})

    print('split: {}, num_files: {}'.format(split, len(samples)))
    return samples


def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token


if __name__ == "__main__":
    VOCAB = {'<sos>': 0, '<eos>': 1}
    IVOCAB = {0: '<sos>', 1: '<eos>'}

    data = dict()
    data['VOCAB'] = VOCAB
    data['IVOCAB'] = IVOCAB
    data['train'] = get_data('train')
    data['dev'] = get_data('dev')
    data['test'] = get_data('test')

    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_dev: ' + str(len(data['dev'])))
    print('num_test: ' + str(len(data['test'])))
    print('vocab_size: ' + str(len(data['VOCAB'])))


