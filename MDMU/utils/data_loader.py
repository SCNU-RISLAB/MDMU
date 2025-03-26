import torch
from torch import nn
import transformers
# import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor, AutoModelForCausalLM, AutoTokenizer,BertModel, BertTokenizer
from torch.utils.data import DataLoader
import random
import pandas as pd
import numpy as np
import string
import pickle


class Dataset_mosi(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    #  text_context_length
    #  audio_context_length

    def __init__(self, csv_path, audio_features_directory, video_features_directory, mode, text_context_length,
                 audio_context_length, video_context_length):
        self.audio_features_directory = audio_features_directory
        self.video_features_directory = video_features_directory
        self.mode = mode

        self.df = pd.read_csv(csv_path)
        invalid_files = ['3aIQUQgawaI/12.wav', '94ULum9MYX0/2.wav', 'mRnEJOLkhp8/24.wav', 'aE-X_QdDaqQ/3.wav',
                         '94ULum9MYX0/11.wav', 'mRnEJOLkhp8/26.wav']
        for f in invalid_files:
            video_id = f.split('/')[0]
            clip_id = f.split('/')[1].split('.')[0]
            self.df = self.df[~((self.df['video_id'] == video_id) & (self.df['clip_id'] == int(clip_id)))]
        self.df = self.df[self.df['mode'] == mode].sort_values(by=['video_id', 'clip_id']).reset_index()

        # store labels
        self.targets_M = self.df['label']

        # store texts
        self.df['text'] = self.df['text'].str[0] + self.df['text'].str[1::].apply(lambda x: x.lower())
        self.texts = self.df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large")

        # store context
        self.video_id = self.df['video_id']
        self.text_context_length = text_context_length
        self.audio_context_length = audio_context_length

    def __getitem__(self, index):
        # load text
        text = str(self.texts[index])

        # load text context
        text_context = ''
        for i in range(1, self.text_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context = str(self.texts[index - i])
                text_context = context + '</s>' + text_context

        # tokenize text
        tokenized_text = self.tokenizer(
            text,
            max_length=96,
            padding="max_length",  # Pad to the specified max_length.
            truncation=True,  # Truncate to the specified max_length.
            add_special_tokens=True,  # Whether to insert [CLS], [SEP], <s>, etc.
            return_attention_mask=True
        )

        # tokenize text context
        text_context = text_context[:-4]
        tokenized_context = self.tokenizer(
            text_context,
            max_length=96,
            padding="max_length",  # Pad to the specified max_length.
            truncation=True,  # Truncate to the specified max_length.
            add_special_tokens=True,  # Whether to insert [CLS], [SEP], <s>, etc.
            return_attention_mask=True
        )

        # Load the audio features of the current index
        audio_feature_path = f"{self.audio_features_directory}/{self.df['video_id'][index]}/{self.df['clip_id'][index]}.pt"
        audio_feature = torch.load(audio_feature_path)


        audio_context_feature = torch.zeros(96, 1024)
        for i in range(1, self.audio_context_length + 1):
            if index - 1 < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                audio_context_feature_path = f"{self.audio_features_directory}/{self.df['video_id'][index - 1]}/{self.df['clip_id'][index - 1]}.pt"
                audio_context_feature = torch.load(audio_context_feature_path)

        # Load visual features by file name
        video_feature_path = f"{self.video_features_directory}/{self.df['video_id'][index]}/{self.df['clip_id'][index]}.pt"
        self.video_feature = torch.load(video_feature_path).squeeze()

        # Current index contextual video features
        self.video_context_feature = torch.zeros(96, 768)
        for i in range(1, self.audio_context_length + 1):
            if index - 1 < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                video_context_feature_path = f"{self.video_features_directory}/{self.df['video_id'][index - 1]}/{self.df['clip_id'][index - 1]}.pt"
                self.video_context_feature = torch.load(video_context_feature_path).squeeze()

        return {
            # visual
            "video_features": torch.tensor(self.video_feature, dtype=torch.float32),
            "video_context_features": torch.tensor(self.video_context_feature, dtype=torch.float32),
            # text
            "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
            "text_context_tokens": torch.tensor(tokenized_context["input_ids"], dtype=torch.long),
            "text_context_masks": torch.tensor(tokenized_context["attention_mask"], dtype=torch.long),
            # audio
            "audio_inputs": torch.tensor(audio_feature, dtype=torch.float32),
            "audio_context_inputs": torch.tensor(audio_context_feature, dtype=torch.float32),
            # labels
            "targets": torch.tensor(self.targets_M[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.targets_M)

class Dataset_mosei(torch.utils.data.Dataset):

    def __init__(self, csv_path, feature_path, mode, text_context_length, audio_context_length, video_context_length):
        self.feature_path = feature_path
        self.mode = mode
        with open(feature_path, 'rb') as feature_file:
            self.feature_data = pickle.load(feature_file)

        self.df = pd.read_csv(csv_path)
        invalid_files = ['3aIQUQgawaI/12.wav', '94ULum9MYX0/2.wav', 'mRnEJOLkhp8/24.wav', 'aE-X_QdDaqQ/3.wav',
                         '94ULum9MYX0/11.wav', 'mRnEJOLkhp8/26.wav']
        for f in invalid_files:
            video_id = f.split('/')[0]
            clip_id = f.split('/')[1].split('.')[0]
            self.df = self.df[~((self.df['video_id'] == video_id) & (self.df['clip_id'] == int(clip_id)))]

        self.df = self.df[self.df['mode'] == mode].sort_values(by=['video_id', 'clip_id']).reset_index(drop=True)

        # store labels
        self.targets_M = self.df['label'].values

        # store texts
        self.df['text'] = self.df['text'].str[0] + self.df['text'].str[1::].apply(lambda x: x.lower())
        self.texts = self.df['text'].values
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        # store context
        self.video_id = self.df['video_id'].values
        self.clip_id = self.df['clip_id'].values
        self.text_context_length = text_context_length
        self.audio_context_length = audio_context_length
        self.video_context_length = video_context_length


        self.audio_features = self.feature_data[self.mode]['audio']
        self.video_features = self.feature_data[self.mode]['vision']


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # load text
        text = str(self.texts[index])

        # load text context
        text_context = ''
        for i in range(1, self.text_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context = str(self.texts[index - i])
                text_context = context + '</s>' + text_context

        # tokenize text
        tokenized_text = self.tokenizer(
            text,
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )

        # tokenize text context
        if len(text_context) > 0 and text_context.endswith('</s>'):
            text_context = text_context[:-4]
        tokenized_context = self.tokenizer(
            text_context,
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )

        audio_feature = self.audio_features[index]
        video_feature = self.video_features[index]


        audio_context_feature = torch.zeros_like(torch.tensor(audio_feature, dtype=torch.float32))
        for i in range(1, self.audio_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                audio_context_feature = self.audio_features[index - i]

        video_context_feature = torch.zeros_like(torch.tensor(video_feature, dtype=torch.float32))
        for i in range(1, self.video_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                video_context_feature = self.video_features[index - i]

        return {
            # visual
            "video_features": torch.tensor(video_feature, dtype=torch.float32),
            "video_context_features": torch.tensor(video_context_feature, dtype=torch.float32),
            # text
            "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
            "text_context_tokens": torch.tensor(tokenized_context["input_ids"], dtype=torch.long),
            "text_context_masks": torch.tensor(tokenized_context["attention_mask"], dtype=torch.long),
            # audio
            "audio_inputs": torch.tensor(audio_feature, dtype=torch.float32),
            "audio_context_inputs": torch.tensor(audio_context_feature, dtype=torch.float32),
            # labels
            "targets": torch.tensor(self.targets_M[index], dtype=torch.float32),
        }


    def __len__(self):
        return len(self.targets_M)

class Dataset_sims(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid

    def __init__(self, csv_path, audio_features_directory, video_features_directory, mode, text_context_length,
                 audio_context_length, video_context_length):
        self.audio_features_directory = audio_features_directory
        self.video_features_directory = video_features_directory
        self.mode = mode

        import chardet

        with open(csv_path, 'rb') as f:
            result = chardet.detect(f.read())

        print(result['encoding'])

        self.df = pd.read_csv(csv_path, encoding=result['encoding'])
        self.df = self.df[self.df['mode'] == mode].sort_values(by=['video_id', 'clip_id']).reset_index()

        # store labels
        self.targets_M = self.df['label']


        # store texts
        self.texts = self.df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-ch")

        # store context
        self.video_id = self.df['video_id']
        self.text_context_length = text_context_length
        self.audio_context_length = audio_context_length

    def __getitem__(self, index):

        # extract text features
        text = str(self.texts[index])

        # load text context
        text_context = ''
        for i in range(1, self.text_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context = str(self.texts[index - i])
                text_context = context + '</s>' + text_context

        tokenized_text = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",  # Pad to the specified max_length.
            truncation=True,  # Truncate to the specified max_length.
            add_special_tokens=True,  # Whether to insert [CLS], [SEP], <s>, etc.
            return_attention_mask=True
        )

        # tokenize text context
        text_context = text_context[:-4]
        tokenized_context = self.tokenizer(
            text_context,
            max_length=96,
            padding="max_length",  # Pad to the specified max_length.
            truncation=True,  # Truncate to the specified max_length.
            add_special_tokens=True,  # Whether to insert [CLS], [SEP], <s>, etc.
            return_attention_mask=True
        )

        audio_feature_path = f"{self.audio_features_directory}/{self.df['video_id'][index]}/{self.df['clip_id'][index]}.pt"
        audio_feature = torch.load(audio_feature_path)

        audio_context_feature = torch.zeros(96, 1024)
        for i in range(1, self.audio_context_length + 1):
            if index - 1 < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                audio_context_feature_path = f"{self.audio_features_directory}/{self.df['video_id'][index - 1]}/{self.df['clip_id'][index - 1]}.pt"
                audio_context_feature = torch.load(audio_context_feature_path)

        video_feature_path = f"{self.video_features_directory}/{self.df['video_id'][index]}/{self.df['clip_id'][index]}.pt"
        self.video_feature = torch.load(video_feature_path).squeeze()

        self.video_context_feature = torch.zeros(96, 768)
        for i in range(1, self.audio_context_length + 1):
            if index - 1 < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                video_context_feature_path = f"{self.video_features_directory}/{self.df['video_id'][index - 1]}/{self.df['clip_id'][index - 1]}.pt"
                self.video_context_feature = torch.load(video_context_feature_path).squeeze()

        return {
            # visual
            "video_features": torch.tensor(self.video_feature, dtype=torch.float32),
            "video_context_features": torch.tensor(self.video_context_feature, dtype=torch.float32),
            # text
            "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
            "text_context_tokens": torch.tensor(tokenized_context["input_ids"], dtype=torch.long),
            "text_context_masks": torch.tensor(tokenized_context["attention_mask"], dtype=torch.long),
            # audio
            "audio_inputs": torch.tensor(audio_feature, dtype=torch.float32),
            "audio_context_inputs": torch.tensor(audio_context_feature, dtype=torch.float32),
            # labels
            "targets": torch.tensor(self.targets_M[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.targets_M)


def collate_fn_mosi_audiopkl(batch):
    """
    Collate function for padding and creating masks
    """
    # Sort batch by length of text features (descending order)
    batch = sorted(batch, key=lambda x: len(x['text_tokens']), reverse=True)

    # Pad text features and create masks
    text_features = pad_sequence([item['text_tokens'] for item in batch], batch_first=True, padding_value=0)
    text_masks = pad_sequence([item['text_masks'] for item in batch], batch_first=True, padding_value=0)
    text_context_features = pad_sequence([item['text_context_tokens'] for item in batch], batch_first=True,
                                         padding_value=0)
    text_context_masks = pad_sequence([item['text_context_masks'] for item in batch], batch_first=True, padding_value=0)

    # Pad audio features and create masks
    audio_features = pad_sequence([item['audio_inputs'] for item in batch], batch_first=True, padding_value=0)
    audio_masks = pad_sequence([item['audio_masks'] for item in batch], batch_first=True, padding_value=0)
    audio_context_features = pad_sequence([item['audio_context_inputs'] for item in batch], batch_first=True,
                                          padding_value=0)
    audio_context_masks = pad_sequence([item['audio_context_masks'] for item in batch], batch_first=True,
                                       padding_value=0)

    # Stack targets
    targets = torch.stack([item['targets'] for item in batch])

    return {
        # Text
        "text_tokens": text_features,
        "text_masks": text_masks,
        "text_context_tokens": text_context_features,
        "text_context_masks": text_context_masks,
        # Audio
        "audio_inputs": audio_features,
        "audio_masks": audio_masks,
        "audio_context_inputs": audio_context_features,
        "audio_context_masks": audio_context_masks,
        # Labels
        "targets": targets
    }


def collate_fn_sims(batch):
    text_tokens = []
    text_masks = []
    audio_inputs = []
    audio_masks = []

    targets_M = []
    targets_T = []
    targets_A = []

    # organize batch
    for i in range(len(batch)):
        # text
        text_tokens.append(batch[i]['text_tokens'])
        text_masks.append(batch[i]['text_masks'])
        # audio
        audio_inputs.append(batch[i]['audio_inputs'])
        audio_masks.append(batch[i]['audio_masks'])

        # labels
        targets_M.append(batch[i]['target']['M'])
        targets_T.append(batch[i]['target']['T'])
        targets_A.append(batch[i]['target']['A'])

    return {
        # text
        "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
        "text_masks": torch.tensor(text_masks, dtype=torch.long),
        # audio
        "audio_inputs": torch.stack(audio_inputs),
        "audio_masks": torch.stack(audio_masks),
        # labels
        "targets": {
            "M": torch.tensor(targets_M, dtype=torch.float32),
            "T": torch.tensor(targets_T, dtype=torch.float32),
            "A": torch.tensor(targets_A, dtype=torch.float32)
        }
    }


def data_loader(batch_size, dataset, text_context_length, audio_context_length, video_context_length, seed):
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    if dataset == 'MOSI':
        csv_path = 'data/MOSI/label.csv'
        audio_feature_path = "data/MOSI/Audio_Features_data2vec"
        video_feature_path = "data/MOSI/Video_Features"

        train_data = Dataset_mosi(csv_path, audio_feature_path, video_feature_path, 'train',
                                    text_context_length=text_context_length,
                                    audio_context_length=audio_context_length,
                                    video_context_length=video_context_length)
        test_data = Dataset_mosi(csv_path, audio_feature_path, video_feature_path, 'test',
                                   text_context_length=text_context_length,
                                   audio_context_length=audio_context_length,
                                   video_context_length=video_context_length)
        val_data = Dataset_mosi(csv_path, audio_feature_path, video_feature_path, 'valid',
                                  text_context_length=text_context_length,
                                  audio_context_length=audio_context_length,
                                  video_context_length=video_context_length)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                  worker_init_fn=seed_worker, generator=g, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True,
                                 worker_init_fn=seed_worker, generator=g, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True,
                                worker_init_fn=seed_worker, generator=g, num_workers=4)
        return train_loader, test_loader, val_loader

    elif dataset == 'MOSEI':
        csv_path = 'data/MOSEI/label.csv'
        audio_feature_path = "data/MOSEI/Audio_Features_data2vec"
        video_feature_path = "data/MOSEI/Video_Features"

        train_data = Dataset_mosei(csv_path, audio_feature_path, video_feature_path, 'train',
                                     text_context_length=text_context_length,
                                     audio_context_length=audio_context_length,
                                     video_context_length=video_context_length)
        test_data = Dataset_mosei(csv_path, audio_feature_path, video_feature_path, 'test',
                                    text_context_length=text_context_length,
                                    audio_context_length=audio_context_length,
                                    video_context_length=video_context_length)
        val_data = Dataset_mosei(csv_path, audio_feature_path, video_feature_path, 'valid',
                                   text_context_length=text_context_length,
                                   audio_context_length=audio_context_length,
                                   video_context_length=video_context_length)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                  worker_init_fn=seed_worker, generator=g, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True,
                                 worker_init_fn=seed_worker, generator=g, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True,
                                worker_init_fn=seed_worker, generator=g, num_workers=4)
        return train_loader, test_loader, val_loader

    else:  #  dataset == 'SIMS'
        csv_path = 'data/SIMS/labelutf-8.csv'
        audio_feature_path = "data/SIMS/Audio_Features_hubert"
        video_feature_path = "data/SIMS/Video_Features"
        train_data = MyDataset_sims(csv_path, audio_feature_path, video_feature_path, 'train',
                                    text_context_length=text_context_length,
                                    audio_context_length=audio_context_length,
                                    video_context_length=video_context_length)
        test_data = MyDataset_sims(csv_path, audio_feature_path, video_feature_path, 'test',
                                   text_context_length=text_context_length,
                                   audio_context_length=audio_context_length,
                                   video_context_length=video_context_length)
        val_data = MyDataset_sims(csv_path, audio_feature_path, video_feature_path, 'valid',
                                  text_context_length=text_context_length,
                                  audio_context_length=audio_context_length,
                                  video_context_length=video_context_length)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                  worker_init_fn=seed_worker, generator=g, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True,
                                 worker_init_fn=seed_worker, generator=g, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True,
                                worker_init_fn=seed_worker, generator=g, num_workers=4)
        return train_loader, test_loader, val_loader
