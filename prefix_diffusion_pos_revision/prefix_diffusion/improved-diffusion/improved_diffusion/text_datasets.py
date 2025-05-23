# from PIL import Image
# import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, \
    PreTrainedTokenizerFast, \
    PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sys, os
import torch
import pdb
import pickle
# sys.path.insert(0, os.path.join(sys.path[0], '../../transformers/examples/pytorch/language-modeling'))
# from custom_trainer import GPT2LMHeadModelCompress, BERTModelCompress, AutoEncoderWithNoise
from collections import Counter, defaultdict
from functools import partial
from itertools import chain


def load_data_text(
        *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, data_args=None,
        task_mode='roc', model=None, padding_mode='block', split='train', load_vocab=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    print('hello loading text data. ')

    if data_args.experiment.startswith('random') and model is None:
        model = None
    elif data_args.experiment.startswith('random') and model is not None:
        print('loading initialized random embeddings. ')

    if task_mode == 'e2e-tgt':  # true
        print('hello loading e2e-tgt. ')
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                                   padding_mode=padding_mode, split=split,
                                                   load_vocab=load_vocab)

    # data_args.modality=e2e-tgt
    if data_args.modality in ['roc-aug', 'roc', 'book', 'yelp', 'commonGen',
                              'commonGen-aug'] and data_args.cache_mode == 'no':
        dataset = TextDataset_NoCache(
            training_data,
            image_size,
            data_args,
            model_arch=data_args.model_arch,
            model_emb=model
        )
    else:
        dataset = TextDataset(
            training_data,
            image_size,
            data_args,
            model_arch=data_args.model_arch,
        )
    # false
    if deterministic:

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            shuffle=False,
            num_workers=0,
        )

    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            shuffle=True,
            num_workers=0,
            # collate_fn=dataset.collate_fn
        )
    while True:
        yield from data_loader


def helper_tokenize_encode_cond(sentence_lst, vocab_dict, model, seqlen, data_args):
    result_train_lst = []
    group_lst = defaultdict(list)
    with torch.no_grad():
        for (src_ids, input_ids) in sentence_lst:
            tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
            tokenized_src = [vocab_dict.get(x, vocab_dict['UNK']) for x in src_ids]
            input_ids = [0] + tokenized_ + [1]
            group_lst['word_ids'].append(input_ids)
            group_lst['src_ids'].append(tokenized_src)

        print(group_lst['word_ids'][:2])
        print('padding mode is pad')
        max_length = seqlen
        group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
        max_src_length = max([len(xx) for xx in group_lst['src_ids']])
        print(max_src_length, seqlen)
        max_src_length = min(seqlen, max_src_length)
        group_lst['src_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['src_ids'],
                                                                            vocab_dict['PAD'],
                                                                            max_src_length,
                                                                            return_mask=True)

        for input_ids, src_ids, src_mask in zip(group_lst['word_ids'], group_lst['src_ids'],
                                                group_lst['src_mask']):
            if data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            elif data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor
            result_train_lst.append({'input_ids': input_ids,
                                     'hidden_states': hidden_state.cpu().tolist(),
                                     'src_ids': src_ids,
                                     'src_mask': src_mask
                                     })
    return result_train_lst


def helper_tokenize_stream(sentence_lst, vocab_dict, model, seqlen, data_args, padding_mode, ):
    import psutil
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    from datasets import Dataset as Dataset2
    raw_datasets = Dataset2.from_dict({'text': sentence_lst})
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        if isinstance(vocab_dict, dict):
            input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
        elif isinstance(vocab_dict, PreTrainedTokenizerFast):
            examples['text'] = [" ".join(seq) for seq in examples['text']]
            input_ids = vocab_dict(examples['text'], add_special_tokens=True)['input_ids']
        result_dict = {'input_ids': input_ids}
        # clm input could be much much longer than block_size
        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['text'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print(tokenized_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    if padding_mode == 'block':
        block_size = seqlen

        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        def pad_function(group_lst):
            max_length = seqlen
            if isinstance(vocab_dict, dict):
                group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
            else:
                group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id,
                                                               max_length)
            return group_lst

        # Process.memory_info is expressed in bytes, so convert to megabytes
        print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        lm_datasets = tokenized_datasets.map(
            pad_function,
            batched=True,
            num_proc=1,
            desc=f"padding",
        )

    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    import datasets
    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def helper_tokenize_encode(sentence_lst, vocab_dict, model, seqlen, data_args, padding_mode, img_lst, label_lst=None, pos=None, imgbu = None):
    result_train_lst = []
    group_lst = defaultdict(list)
    with torch.no_grad():
        for input_ids in sentence_lst:
            tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
            # input_ids = [0] + tokenized_ + [1]
            input_ids = tokenized_
            group_lst['word_ids'].append(input_ids)
        print(group_lst['word_ids'][:2])
        if padding_mode == 'block':  # true
            print('padding mode is block')
            concatenated_examples = {k: sum(group_lst[k], []) for k in group_lst.keys()}  # 将所有word_id 连接一起
            total_length = len(concatenated_examples[list(group_lst.keys())[0]])  # 句子总长度
            block_size = seqlen  # 64
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            group_lst = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        elif padding_mode == 'pad':
            print('padding mode is pad')
            max_length = seqlen
            group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
        assert len(group_lst['word_ids']) == len(img_lst)
        # print(len(img_lst))
        # print(len(pos))
        i = 0
        for input_ids, posid in tqdm(zip(group_lst['word_ids'], pos)):
            # print(input_ids)
            # print(posid)
            img = img_lst[i]

            if data_args.experiment.startswith('random'):  # true
                hidden_state = model(torch.tensor(input_ids))
            elif data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor
            elif data_args.experiment == 'glove':
                hidden_state = model(torch.tensor(input_ids))
            # input_ids 是64个整数，hidden_state是64*16的编码向量  result_train_lst.append({'input_ids': input_ids, 'img':img})
            # print(label_lst)
            if label_lst == None:
                result_train_lst.append(
                    {'input_ids': input_ids, 'hidden_states': hidden_state.cpu().tolist(), 'img': img, 'pos': posid})
            else:
                result_train_lst.append(
                    {'input_ids': input_ids, 'hidden_states': hidden_state.cpu().tolist(), 'img': img,
                     'label': label_lst[i]})
            i = i + 1
    return result_train_lst


def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = torch.tensor(np.array(split_line[1:], dtype=np.float64))
            # embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def load_glove(vocab):
    model = torch.nn.Embedding(len(vocab), 50)
    glove_model = load_glove_model('glove/glove.6B.50d.txt')
    array_lst = []
    count_ = 0
    for word, idx in vocab.items():
        if word in glove_model:
            array_lst.append(glove_model[word])
        else:
            count_ += 1
            array_lst.append(torch.randn(50))
    print(f'{count_} out of {len(vocab)} is initialized. ')
    array_lst = torch.stack(array_lst)
    print(torch.norm(array_lst, dim=-1).mean())
    model.weight.data = array_lst
    return model


def get_corpus_rocstory(data_args, model, image_size, padding_mode='block',
                        split='train', load_vocab=None):
    import csv, torch, json
    from spacy.lang.en import English

    if data_args.experiment_mode == 'lm':
        if data_args.modality == 'e2e-tgt':
            print('loading dataset from simple e2e dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.e2e_train}/knn_vit_coco_train_text_img_txt.pkl'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.e2e_train}/knn_vit_coco_val_text_img_txt.pkl'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.e2e_train}/knn_vit_pos13_coco_test_text_img_512.pkl'
            if split in ['train', 'valid', 'test']:
                with open(path, 'rb') as ff:
                    all_data = pickle.load(ff)
                print("data size is%0d" % len(all_data['captions']))
                captions_raw = all_data["captions"][0:len(all_data['captions'])]
                # caption_lst = [caption['raw'] for caption in captions_raw]
                img_lst = [caption['img'] for caption in captions_raw]
                sentence_lst = [caption['tokens'] for caption in captions_raw]
                txt_emb_lst = [caption['text_emb'] for caption in captions_raw]
                # pos_lst = [caption['pos'] for caption in captions_raw]
                # imgbu_lst = [caption['imgbu'] for caption in captions_raw]
                # print(pos_lst)
                # label_lst = [caption['label'] for caption in captions_raw]
            print(sentence_lst[:2])
            # print(pos_lst[:2])
        # print(load_vocab)
        # get tokenizer.
        if load_vocab is None:
            counter = Counter()
            for input_ids in sentence_lst:
                counter.update(input_ids)

    if load_vocab is None:
        # vocab_dict = {'START': 0, 'ENDS': 1, 'UNK':2, 'PAD':3}
        vocab_dict = {'START': 0, 'ENDS': 1, 'UNK': 2, 'PAD': 3, }
        for k, v in counter.items():
            if v > 1:
                vocab_dict[k] = len(vocab_dict)
            # vocab_dict[k] = len(vocab_dict)
        print(len(counter), len(vocab_dict))
        path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
        print(f'save the vocab to {path_save_vocab}')
        with open(path_save_vocab, 'w') as f:
            json.dump(vocab_dict, f)
    else:
        vocab_dict = load_vocab
        path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
        if not os.path.exists(path_save_vocab):
            print(f'save the vocab to {path_save_vocab}')
            if isinstance(vocab_dict, dict):
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                assert vocab_dict['START'] == 0
            elif isinstance(vocab_dict, PreTrainedTokenizerFast):
                vocab_dict.save_pretrained(data_args.checkpoint_path)
            else:
                assert False, "invalid type of vocab_dict"

    if model is None and data_args.experiment == 'random':  # # model=None,
        # *
        model = torch.nn.Embedding(len(vocab_dict), data_args.in_channel)  # 编码器
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        path_save = f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)
    elif data_args.experiment == 'glove':
        assert data_args.in_channel == 50
        model = load_glove(vocab_dict)
        path_save = f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)

    path_save = f'{data_args.checkpoint_path}/random_emb.torch'
    if not os.path.exists(path_save) and data_args.experiment == 'random':
        torch.save(model.state_dict(), path_save)

    if data_args.experiment_mode == 'lm':
        # result_train_lst是整数的token和16维的词向量
        result_train_lst = helper_tokenize_encode(sentence_lst, vocab_dict, model, image_size * 3, data_args,
                                                  padding_mode, img_lst, pos=pos_lst)
        # print(result_train_lst[0])
    return {'train': result_train_lst}, model  # model 是Embedding(821,16)


def write_e2e_corr(prompt_lst, file_dict, corr_path):
    print(len(prompt_lst))
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            for line in file_dict[x]:
                print(" ".join(line), file=f)
            print('', file=f)


def write_e2e_src(prompt_lst, corr_path):
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            print(" ".join(x), file=f)
    return


def read_e2e_files(path, args, tokenizer):
    file_dict = {}
    with open(path, 'r') as f:
        for line in f:
            src_lst, word_lst = line.strip().split('||')
            tgt = tuple([x.text for x in tokenizer(word_lst)])
            src = tuple([x.text for x in tokenizer(src_lst)])
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    temp = '1'
    prompt_text_dict = file_dict
    prompt_text_lst = list(prompt_text_dict.keys())
    gold_dir = os.path.join(args.out_dir, '{}_{}_{}'.format(temp, args.split, 'gold'))
    print("gold dir", gold_dir)
    write_e2e_corr(prompt_text_lst, prompt_text_dict, gold_dir)
    src_dir = os.path.join(args.out_dir, '{}_{}_{}'.format(temp, args.split, 'src'))
    write_e2e_src(prompt_text_lst, src_dir)
    final_lst = [(xx, prompt_text_dict[xx][0]) for xx in prompt_text_lst]
    return final_lst


def get_corpus_book(data_args, tokenizer, model, image_size, padding_mode='block', split='train', ):
    max_length = image_size ** 2
    import os
    assert padding_mode == 'block'
    raw_datasets = load_dataset('bookcorpus')
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            'bookcorpus',
            split=f"train[:1%]",
        )
        raw_datasets["train"] = load_dataset(
            'bookcorpus',
            split=f"train[1%:]",
        )
    print(raw_datasets)
    column_names = raw_datasets["train"].column_names

    def tokenize_function(examples):
        output = tokenizer(examples['text'], add_special_tokens=False)
        return output

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    print(tokenized_datasets)

    block_size = max_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    print(lm_datasets)

    if model is None:
        if data_args.training_mode.startswith('e2e'):
            print('since its e2e, initialize a dummy embedding')
            model = torch.nn.Embedding(len(tokenizer), 1)
        else:
            model = torch.nn.Embedding(len(tokenizer), data_args.in_channel)
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        path_save = f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)

    if split == 'train':
        return lm_datasets, model
    else:
        lm_datasets['train'] = lm_datasets['validation']
        return lm_datasets, model


class TextDataset(Dataset):
    def __init__(self, text_datasets, resolution, data_args, model_arch='conv-unet',
                 classes=None, shard=0, num_shards=1, eigen_transform=None,
                 mapping_func=None, model_emb=None):
        super().__init__()
        self.resolution = resolution
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.model_arch = model_arch
        self.data_args = data_args
        print(self.resolution)
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb
        # self.local_images = image_paths[shard:][::num_shards]
        # self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.featbupath = '/home/cumt/wjworkspace/data/cocobu_att'
        self.att_loader = HybridLoader(self.featbupath, '.npz', in_memory=False)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        if self.model_arch == 'conv-unet':
            arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                           dtype=np.float32).reshape(self.resolution, self.resolution, -1)
            # print(self.eigen_transform.shape)
            if self.eigen_transform is not None:
                old_shape = arr.shape
                arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                arr = arr @ self.eigen_transform['map']
                arr = arr.reshape(old_shape)
            if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)

            out_dict = {}
            out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            # print(out_dict.keys())
            return np.transpose(arr, [2, 0, 1]), out_dict
        elif self.model_arch == '1d-unet':
            arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                           dtype=np.float32)  # seqlen, dim
            if self.eigen_transform is not None:
                old_shape = arr.shape
                arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                arr = arr @ self.eigen_transform['map']
                arr = arr.reshape(old_shape)
            if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
            arr = np.transpose(arr, [1, 0])
            out_dict = {}
            out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            # out_dict['mapping_func'] = self.mapping_func
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            # print(arr.shape)
            return arr, out_dict
        else:
            arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                           dtype=np.float32)
            if self.eigen_transform is not None:
                old_shape = arr.shape
                # arr = arr.reshape(1, -1) @ self.eigen_transform
                arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                arr = arr @ self.eigen_transform['map']
                arr = arr.reshape(old_shape)

            if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                # print(arr.dtype)
                # print(self.data_args.noise_level, 'using the noise level.')
                arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
                # print(arr.dtype)
            
            out_dict = {}
            out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_dict['img'] = np.array(self.text_datasets['train'][idx]['img'])
            out_dict['txt_emb'] = np.array(self.text_datasets['train'][idx]['txt_emb'])
            # bu attention and mask    
            # att_feat = self.att_loader.get(str(self.text_datasets['train'][idx]['imgbu']))
            # Reshape to K x C
            # att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            # out_dict['bu_att'] = att_feat
            # out_dict['bu_mask'] = np.array(self.text_datasets['train'][idx]['bu_mask'])
            # out_dict['mapping_func'] = self.mapping_func
            if self.data_args.experiment_mode == 'conditional_gen':
                out_dict['src_ids'] = np.array(self.text_datasets['train'][idx]['src_ids'])
                out_dict['src_mask'] = np.array(self.text_datasets['train'][idx]['src_mask'])
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            return arr, out_dict  #
        # print(arr.dtype)
        # arr = arr.float()
        # print(arr.shape)
    # @staticmethod
    # def collate_fn(batch):
    #     # print(batch)
    #     bu_att_batch = []
    #     input_ids_batch = []
    #     img_batch = []
    #     pos_batch = []
    #     word_emb_batch = []
    #     # wrapped = False
    #
    #     for sample in batch:
    #         word_emb, dictionary = sample
    #         input_ids = dictionary['input_ids']
    #         img = dictionary['img']
    #         pos = dictionary['pos']
    #         bu_att = dictionary['bu_att']
    #         bu_att_batch.append(bu_att)
    #         input_ids_batch.append(input_ids)
    #         pos_batch.append(pos)
    #         img_batch.append(img)
    #         word_emb_batch.append(word_emb)
    #
    #     data = {}
    #     # data['bu_att'] = np.stack(bu_att_batch)
    #     max_bu_att_len = max([_.shape[0] for _ in bu_att_batch])
    #     data['att_feats'] = np.zeros([len(bu_att_batch), max_bu_att_len, bu_att_batch[0].shape[1]], dtype='float32')
    #     for i in range(len(bu_att_batch)):
    #         data['att_feats'][i, :bu_att_batch[i].shape[0]] = bu_att_batch[i]
    #     data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
    #     for i in range(len(bu_att_batch)):
    #         data['att_masks'][i, :bu_att_batch[i].shape[0]] = 1
    #     # set att_masks to None if attention features have same length
    #     if data['att_masks'].sum() == data['att_masks'].size:
    #         data['att_masks'] = None
    #     data['input_ids'] = np.vstack(input_ids_batch)
    #     data['pos'] = np.vstack(pos_batch)
    #     # generate mask
    #     # nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
    #     # mask_batch = np.zeros([data['labels'].shape[0], len(input_ids_batch[0])], dtype='float32')
    #     # for ix, row in enumerate(mask_batch):
    #     #     row[:nonzeros[ix]] = 1
    #     # data['masks'] = mask_batch
    #     # data['labels'] = data['labels']
    #     # data['pos'] = data['pos']
    #     # data['bounds'] = {'it_pos_now': it_pos_now,  # the it_pos_now of the last sample
    #     #                   'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
    #     # data['infos'] = infos
    #     data_wordEmb_np = np.stack(word_emb_batch)
    #     data_wordEmb_th = torch.from_numpy(data_wordEmb_np)
    #     data['img'] = np.stack(img_batch)
    #     data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
    #             data.items()}  # Turn all ndarray to torch tensor
    #     return data_wordEmb_th, data
            
import six, h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """

    def __init__(self, db_path, ext, in_memory=False):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['feat'] if 'feat' in x else x[
                    'z']  # normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.

            self.loader = load_npz
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.lmdb = lmdbdict(db_path, unsafe=True)
            self.lmdb._key_dumps = DUMPS_FUNC['ascii']
            self.lmdb._value_loads = LOADS_FUNC['identity']
        elif db_path.endswith('.pth'):  # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}

    def get(self, key):

        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            f_input = self.lmdb[key]
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input)

        return feat


class TextDataset_NoCache(Dataset):
    def __init__(self, text_datasets, resolution, data_args, model_arch='conv-unet',
                 classes=None, shard=0, num_shards=1, eigen_transform=None,
                 mapping_func=None, model_emb=None):
        super().__init__()
        self.resolution = resolution
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.model_arch = model_arch
        self.data_args = data_args
        print(self.resolution)
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb
        # self.local_images = image_paths[shard:][::num_shards]
        # self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        with torch.no_grad():
            input_ids = self.text_datasets['train'][idx]['input_ids']
            model = self.model_emb
            if self.data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            elif self.data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor

            if self.model_arch == 'conv-unet':
                arr = np.array(hidden_state,
                               dtype=np.float32).reshape(self.resolution, self.resolution, -1)
                # print(self.eigen_transform.shape)
                if self.eigen_transform is not None:
                    old_shape = arr.shape
                    arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                    arr = arr @ self.eigen_transform['map']
                    arr = arr.reshape(old_shape)
                if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                    arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)

                out_dict = {}
                out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
                # if self.local_classes is not None:
                #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
                # print(out_dict.keys())
                return np.transpose(arr, [2, 0, 1]), out_dict
            elif self.model_arch == '1d-unet':
                arr = np.array(hidden_state,
                               dtype=np.float32)  # seqlen, dim
                if self.eigen_transform is not None:
                    old_shape = arr.shape
                    arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                    arr = arr @ self.eigen_transform['map']
                    arr = arr.reshape(old_shape)
                if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                    arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
                arr = np.transpose(arr, [1, 0])
                out_dict = {}
                out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
                # out_dict['mapping_func'] = self.mapping_func
                # if self.local_classes is not None:
                #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
                # print(arr.shape)
                return arr, out_dict
            else:
                arr = np.array(hidden_state,
                               dtype=np.float32)
                if self.eigen_transform is not None:
                    old_shape = arr.shape
                    # arr = arr.reshape(1, -1) @ self.eigen_transform
                    arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                    arr = arr @ self.eigen_transform['map']
                    arr = arr.reshape(old_shape)

                if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                    # print(arr.dtype)
                    # print(self.data_args.noise_level, 'using the noise level.')
                    arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
                    # print(arr.dtype)

                out_dict = {}
                out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
                # out_dict['mapping_func'] = self.mapping_func
                if self.data_args.experiment_mode == 'conditional_gen':
                    out_dict['src_ids'] = np.array(self.text_datasets['train'][idx]['src_ids'])
                    out_dict['src_mask'] = np.array(self.text_datasets['train'][idx]['src_mask'])
                # if self.local_classes is not None:
                #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
                return arr, out_dict


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


def _torch_collate_batch(examples, pad_token_id, max_length):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # length_of_first = examples[0].size(0)
    # Check if padding is necessary.
    # are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    # if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
    #     return torch.stack(examples, dim=0)
    # Creating the full tensor and filling it with our data.
    # max_length = max(x.size(0) for x in examples)
    # if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
    #     max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], pad_token_id)
    for i, example in enumerate(examples):
        if True:
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result
