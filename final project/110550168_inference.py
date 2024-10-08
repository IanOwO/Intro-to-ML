import torch
import warnings
torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")
import os
import argparse
import tqdm
from training.utils.config_utils import load_yaml
from training.vis_utils import ImgLoader
import csv

def build_model(pretrainewd_path: str,
                img_size: int, 
                fpn_size: int, 
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True, 
                use_selection: bool = True,
                use_combiner: bool = True, # marker, ori : true 
                comb_proj_size: int = None):
    from training.models.pim_module.pim_module_eval import PluginMoodel

    model = \
        PluginMoodel(img_size = img_size,
                     use_fpn = use_fpn,
                     fpn_size = fpn_size,
                     proj_type = "Linear",
                     upsample_type = "Conv",
                     use_selection = use_selection,
                     num_classes = num_classes,
                     num_selects = num_selects, 
                     use_combiner = use_combiner,
                     comb_proj_size = comb_proj_size)

    if pretrainewd_path != "":
        ckpt = torch.load(pretrainewd_path)
        model.load_state_dict(ckpt['model'])
    
    model.eval()

    return model
@torch.no_grad()
def sum_all_out(out, sum_type="softmax"):
    target_layer_names = \
    ['layer1', 'layer2', 'layer3', 'layer4',
    'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4',
    'comb_outs'] # marker, unmarked

    sum_out = None
    for name in target_layer_names:
        if name != "comb_outs":
            tmp_out = out[name].mean(1)
        else:
            tmp_out = out[name]
        
        if sum_type == "softmax":
            tmp_out = torch.softmax(tmp_out, dim=-1)
        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out # note that use '+=' would cause inplace error
    return sum_out

if __name__ == "__main__":
    # ===== 0. get setting =====
    parser = argparse.ArgumentParser("Visualize SwinT Large")
    args = parser.parse_args()

     
    load_yaml(args, "./training/records/FGVC-HERBS/basline/config.yaml")

    # ===== 1. build model =====
    # pretrainewd_path = args.pretrained_root + "/backup/last.pt",    
    model = build_model( pretrainewd_path = "./best.pt",
                        img_size = args.data_size, 
                        fpn_size = args.fpn_size, 
                        num_classes = args.num_classes,
                        num_selects = args.num_selects)
    model.cuda()

    img_loader = ImgLoader(img_size = args.data_size)

    # files = os.listdir(args.image_root)
    files = os.listdir("./training/final_training_data/test/")
    files.sort()
    total = 0
    pbar = tqdm.tqdm(total=len(files), ascii=True)
    # wrongs = {}


    with open("result.csv","w",newline="") as result:
        result_writer = csv.writer(result)
        result_writer.writerow(("id","label"))

        imgs = []
        img_paths = []
        update_n = 0

        # types = os.listdir('final_training_data/train') 
        types = os.listdir('./training/final_training_data/train') 
        types.sort()
        for fi, f in enumerate(files):
            # img_path = args.image_root + "/" + f    
            img_path = "./training/final_training_data/test/" + f
            img_paths.append(img_path)
            img, ori_img = img_loader.load(img_path)
            img = img.unsqueeze(0) # add batch size dimension
            imgs.append(img)
            update_n += 1
            if (fi+1) % 32 == 0 or fi == len(files) - 1:    
                imgs = torch.cat(imgs, dim=0)
            else:
                continue
            with torch.no_grad():
                imgs = imgs.cuda()
                outs = model(imgs)
                sum_outs = sum_all_out(outs, sum_type="softmax") # softmax
                preds = torch.sort(sum_outs, dim=-1, descending=True)[1]
                for bi in range(preds.size(0)):
                    name = img_paths[bi].split("/")[-1][:-4]
                    label = types[preds[bi, 0].item()]
                    result_writer.writerow([name, label])

            imgs = []
            img_paths = []
            
            pbar.update(update_n)
            update_n = 0
    pbar.close()

