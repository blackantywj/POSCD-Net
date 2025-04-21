import json

with open("/home/cumt/wjworkspace/prefix_diffusion_pos_revision/prefix_diffusion/improved-diffusion/scripts/captions_val2014.json", 'r') as f:
    alldata = json.load(f)
    annodict = alldata['annotations']
    annodict = dict()
    pass

with open("/home/cumt/wjworkspace/prefix_diffusion_pos_revision/prefix_diffusion/improved-diffusion/scripts/karpathy_test_images.txt", "r", encoding='utf-8') as f:  #打开文本
    karpdata = f.read()   #读取文本
    karpdata = karpdata.split('\n')
    karpdatalist = karpdata[:5000]
    # print()
    
imgidlist = []
# for i in karpdatalist:
#     imgid = i.split(' ')
#     # assert annodict['image_id'][int(imgid[1])]
#     for annitem in annodict:
#         annitem = 
#     imgidlist.append(imgid[1])
    
pass