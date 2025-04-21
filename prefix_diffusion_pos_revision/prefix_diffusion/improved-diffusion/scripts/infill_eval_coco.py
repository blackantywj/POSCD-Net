"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json, sys
import numpy as np
import torch as th
from torch import nn
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.rounding import rounding_func, rounding_func_pos, load_models, load_tokenizer
from improved_diffusion.test_util import get_weights, denoised_fn_round, denoised_fn_round_pos
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
sys.path.insert(0, '../transformers/examples/pytorch/language-modeling')
# from custom_trainer import Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree,Classifier_image
# from infill_util import langevin_fn3, get_score, langevin_fn3_compose, langevin_fn1, langevin_fn4, langevin_fn_tree, langevin_fn_length
from spacy.lang.en import English
# sys.path.append("clip/")
# from model_creation import create_clip_model
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.nn import functional as F
import clip
import pdb
import pickle
import json
from PIL import Image
from tqdm import tqdm
import skimage.io as io
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from eval_metrics import sentence_pro
from eval_metrics import ref_pro
from eval_metrics import calculate_distinct
from eval_metrics import clip_choose


def main():
    set_seed(1234)
    args = create_argparser().parse_args()
    assert os.path.exists(args.out_dir)==True
    # pdb.set_trace()
    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    args.noise_level = 0.0
    args.sigma_small = True

    if args.eval_task_.startswith('control_'):
        args.diffusion_steps = 1300  # 500  # DEBUG
    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')  # args.clip_denoised=False
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()
    args.batch_size = 20
    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])
    # args.modality=e2e-tgt,experiment=random,model_name_or_path=predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None
    # in_channel=16 os.path.split(model_path)[0]=diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e
    pos_embs, model_embs, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                   os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):    # args.training_mode=e2e
        print('e2e, load the right model embeddings', '*'*80)
        model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
        pos_embs.weight = th.nn.Parameter(model.posEmbedding.weight.clone().cpu())
    model_embs = model_embs.cuda()  # model_embs=Embedding(821,16)
    pos_embs = pos_embs.cuda()
    model3 = get_weights(model_embs, args)  # model3 = Embedding(821,16)
    model4 = get_weights(pos_embs, args)
    
    logger.log("sampling...")
    sample_dict = {}
    img_path= "/home/cumt/workspace/prefix_diffusion_clip/prefix_diffusion/improved-diffusion/datasets/coco_vit_L14/prefix_diffusion_pos+textemb_test.pkl"
    with open(img_path, 'rb') as ff:
        all_data = pickle.load(ff)
    print("data size is%0d" % len(all_data['captions']))
    captions_raw = all_data['captions'][0:len(all_data['captions'])]
    img_lst = [caption['img'] for caption in captions_raw]
    id_lst = [caption['img_id'] for caption in captions_raw]
    imgbu_lst = [caption['imgbu'] for caption in captions_raw]
    
    with open("/home/cumt/wjworkspace/prefix_diffusion_pos_revision/prefix_diffusion/improved-diffusion/scripts/captions_val2014.json", 'r') as f:
        alldata = json.load(f)
    
    ref_data = []
    # for imgbu, key, valuelist in zip(imgbu_lst, id_lst, captions_raw):
    #     for value in valuelist['raw']:
    #         out_dict = dict()
    #         out_dict['id'] = int(key)
    #         out_dict['image_id'] = int(imgbu)
    #         out_dict['caption'] = "".join(value).strip()
    #         ref_data.append(out_dict)
    
    # with open("/home/cumt/wjworkspace/prefix_diffusion_pos_revision/prefix_diffusion/improved-diffusion/datasets/coco/test_ref_coco.json", 'r+') as ref:
    #     json.dump({'annotations':ref_data, 'info':alldata['info'], 'images':alldata['images'], 'licenses':alldata['licenses']}, ref)    
    
    seqlen = 30
    #device="cpu"
    text_lst = []
    pos_lst = []
    for image_feature in tqdm(img_lst[:1]):
        all_texts = []
        all_texts_pos = []
        while len(all_texts) * args.batch_size < args.num_samples: # batch_size=64
            model_kwargs = {}

            sample_shape = (args.batch_size, seqlen, args.in_channel, )     # sample_shape=64*64*16
            if args.use_ddim:   # true
                loop_func_ = diffusion.ddim_sample_loop_progressive
            else:
                loop_func_ = diffusion.p_sample_loop_progressive
            for sample in loop_func_(
                    model,
                    sample_shape,
                    denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                    pos_denoised_fn=partial(denoised_fn_round_pos, args, model4.cuda()),
                    clip_denoised=args.clip_denoised,   # clip_denoised=false
                    model_kwargs=model_kwargs,  # model_kwargs={},w为空字典
                    device=device,   #
                    eta=args.eta,   # eta=1.0
                    image_feature = image_feature.to(th.float32).to(device)
            ):
                final = sample["sample"]    # sample["sample"].shape=64*64*16,sample["pre_xstart"]=64*64*16
                final_pos = sample["sample_pos"]    # sample["sample"].shape=64*64*16,sample["pre_xstart"]=64*64*16
            sample = final  # sample.shape=64*64*16
            sample_pos = final_pos
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_texts.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log(f"created {len(all_texts) * args.batch_size} samples")
            # pos decoder
            gathered_samples_pos = [th.zeros_like(sample_pos) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples_pos, sample_pos)  # gather not supported with NCCL
            all_texts_pos.extend([sample_pos.cpu().numpy() for sample_pos in gathered_samples_pos])
            logger.log(f"created {len(all_texts_pos) * args.batch_size} samples")
        arr = np.concatenate(all_texts, axis=0)
        arr_pos = np.concatenate(all_texts_pos, axis=0)
        arr = arr[: args.num_samples]
        arr_pos = arr_pos[: args.num_samples]
        text_lst.append(arr)
        pos_lst.append(arr_pos)
        # break
    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'_{os.path.split(args.model_path)[1]}'
    dist.barrier()
    logger.log("sampling complete")
    def decode_helper(args, text_lst, img_lst, diff_model=None):
        if not diffusion.training_mode.startswith('e2e'):
            logger.log('decode by rounding. ')
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])
        text_all=[]
        for arr in text_lst:
            if diffusion.training_mode.startswith('e2e'):
                word_lst_e2e = []
                print('decoding for e2e', )
                x_t = th.tensor(arr).cuda()     # 50*64*16
                print(x_t.shape)
                if args.model_arch == 'conv-unet':
                    reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
                else:
                    reshaped_x_t = x_t
                logits = diff_model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab 50*64*821
                cands = th.topk(logits, k=1, dim=-1)
                tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
                for seq in cands.indices:
                    tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                    word_lst_e2e.append(tokens)
                word_lst = word_lst_e2e     # len(word_lst=50,每个条件生成50个sample
            else:
                word_lst = rounding_func(args.experiment, arr, model, tokenizer)
            word_lst = sentence_pro(word_lst)
            text_all.append(word_lst)
            # pdb.set_trace()
        # text_all = clip_choose(text_all,img_lst)
        return text_all
    
    def decode_helper_pos(args, text_lst, img_lst, diff_model=None):
        if not diffusion.training_mode.startswith('e2e'):
            logger.log('decode by rounding. ')
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])
        text_all=[]
        for arr in text_lst:
            if diffusion.training_mode.startswith('e2e'):
                word_lst_e2e = []
                print('decoding for e2e', )
                x_t = th.tensor(arr).cuda()     # 50*64*16
                print(x_t.shape)
                if args.model_arch == 'conv-unet':
                    reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
                else:
                    reshaped_x_t = x_t
                logits = diff_model.get_pos_logits(reshaped_x_t)  # bsz, seqlen, vocab 50*64*821
                cands = th.topk(logits, k=1, dim=-1)
                # tokenizer = {0: " UNK ", 1: " 形 ", 2:" 介 ", 3:" 副 ", 4:" 连 ", 5:" 冠 ", 6:" 名 ", 7:" 数 ", 8:" PRT ", 9:" 代 ", 10:" 动 ", 11:" 符 ", 12:" 叹 ", 13: "PAD"}
                tokenizer = {0: "0", 1: "1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"10", 11:"11", 12:"12", 13: "13"}
                # tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
                for seq in cands.indices:
                    tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                    word_lst_e2e.append(tokens)
                word_lst = word_lst_e2e     # len(word_lst=50,每个条件生成50个sample
            else:
                word_lst = rounding_func_pos(args.experiment, arr, model, tokenizer)
            word_lst = sentence_pro(word_lst)
            text_all.append(word_lst)
            # pdb.set_trace()
        # text_all = clip_choose(text_all,img_lst)
        return text_all
    
    if args.verbose == 'pipe':
        print(f'sampled for {len(sample_dict)} control tasks')
        out_path_pipe = os.path.join(args.out_dir, f"{model_base_name}.json")
        # out_path_pipe = os.path.join(args.out_dir, f"{model_base_name}.json")
        result_dict = decode_helper(args, text_lst,img_lst, diff_model=model)
        result_dict_pos = decode_helper_pos(args, pos_lst, img_lst, diff_model=model)
        result_lst = []
        for key, value in zip(id_lst, result_dict):
            out_dict = dict()
            out_dict['image_id'] = int(key)
            out_dict['caption'] = "".join(value).strip()
            result_lst.append(out_dict)
        with open(out_path_pipe, 'w', encoding='utf-8') as f:
            json.dump(result_lst, f)
        print(f'written the decoded output to {out_path_pipe}')
        annFile = "/home/cumt/wjworkspace/prefix_diffusion_pos_revision/prefix_diffusion/improved-diffusion/datasets/coco/test_ref_coco.json"
        coco = COCO(annFile)
        cocoRes = coco.loadRes(out_path_pipe)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()
        result_file = args.out_dir + f"{model_base_name}_coco_result.txt"
        for metric, score in cocoEval.eval.items():
            with open(result_file, 'a') as fo:
                fo.write(f'{metric} = {round(score, 4)}\n')
        calculate_distinct(out_path_pipe, result_file, 14145)    #
    return args

def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=5, batch_size=1, model_path="",
        out_dir="/home/cumt/wjworkspace/prefix_diffusion_pos_revision/prefix_diffusion/improved-diffusion/generation_out/coco/1230/20ttt/",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
        start_idx=0, end_idx=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = main()


