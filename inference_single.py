import os
import shutil
from inference import Inferencer
from types import SimpleNamespace
from PIL import Image
import torch

# ---- USER CONFIG ----
MODEL_PATH = "./FSG-Net-DRIVE.pt"
MODEL_NAME = "FSGNet"
INPUT_IMAGE = "/Users/apple/home/freelance/retina_fundus_images/data11/ AMDX-2011P-001_103-305_images/OD/1_AMDX-2011P-001_103-305_1945_OD_20250604_11hr31min09sec_before_02hr15min55sec.tif"
INPUT_SIZE = [1024, 1024]  
# ---------------------

def prepare_single_image_dir(img_path, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(img_path))[0] + '.png'  
    img = Image.open(img_path).convert("RGB")
    img = img.resize(INPUT_SIZE, Image.BILINEAR)
    dst = os.path.join(tmp_dir, img_name)
    img.save(dst)
    mask = Image.new("L", INPUT_SIZE)
    mask.save(os.path.join(tmp_dir, "dummy_mask.png"))
    return dst, os.path.join(tmp_dir, "dummy_mask.png")

def main():
    tmp_x = "tmp_x"
    tmp_y = "tmp_y"
    os.makedirs(tmp_x, exist_ok=True)
    os.makedirs(tmp_y, exist_ok=True)
    img_dst, mask_dst = prepare_single_image_dir(INPUT_IMAGE, tmp_x), prepare_single_image_dir(INPUT_IMAGE, tmp_y)
    args = SimpleNamespace(
        val_x_path=tmp_x,
        val_y_path=tmp_y,
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        cuda=torch.cuda.is_available(),
        n_classes=1,
        input_channel=3,
        input_size=INPUT_SIZE,
        dataloader='Image2Image_resize',
        worker=0,
        pin_memory=False,
        task='segmentation',
        debug=False,
        inference_mode='segmentation',
        mode='inference',
        transform_cutmix=False,
        transform_rand_resize=False,
        transform_hflip=False,
        transform_jitter=False,
        transform_blur=False,
        transform_perspective=False,
        input_space='RGB',
        transform_rand_crop=None,
        wandb=False,
        CUDA_VISIBLE_DEVICES='0'
    )
    inferencer = Inferencer(args)
    inferencer.start_inference_segmentation()
    print(f"Done. Check output in the model directory or as per Inferencer's logic.")

if __name__ == "__main__":
    main()