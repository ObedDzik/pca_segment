#%%
import numpy as np
import SimpleITK as sitk
import os
from monai.transforms import SaveImage
from monai.data import MetaTensor
from monai.transforms import LoadImaged, Compose
from monai.data import Dataset, DataLoader
#%%
output_dir = "/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/test_predictions/segresnet_preds/L1ghdl_ce/ensemble_results"
network = "segresnet"
fold = [0,1,2,3,4]
inputsize = 128
experiment_code = []
preds_folder = []
preds_list = []

for i in fold:
    experiment_code.append(f"{network}_fold{fold[i]}_randcrop{inputsize}")
    preds_folder.append(f"/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/test_predictions/segresnet_preds/L1ghdl_ce/predictions/{'fold'+str(fold[i])}/{network}/{experiment_code[i]}")
    preds_list.append(sorted(os.listdir(preds_folder[i])))

pred_fold0 = []
pred_fold1 = []
pred_fold2 = []
pred_fold3 = []
pred_fold4 = []

transforms = Compose([LoadImaged(keys="image")])

def load_images(preds_folder, preds_list):
    images = []
    for i in range(len(preds_list)):
        data_dict = {"image": os.path.join(preds_folder, preds_list[i])}
        dataset = Dataset(data=[data_dict], transform=transforms)
        dataloader = DataLoader(dataset, batch_size=1)
        for batch in dataloader:
            image = batch["image"]
            images.append(image)
    return images

pred_fold0 = load_images(preds_folder[0], preds_list[0])
pred_fold1 = load_images(preds_folder[1], preds_list[1])
pred_fold2 = load_images(preds_folder[2], preds_list[2])
pred_fold3 = load_images(preds_folder[3], preds_list[3])
pred_fold4 = load_images(preds_folder[4], preds_list[4])

#%%
preds = [pred_fold0, pred_fold1, pred_fold2, pred_fold3, pred_fold4]
#%%
save_image = SaveImage(output_dir=output_dir, 
                       output_postfix="seg",
                       resample=False,
                       separate_folder=False)
for i in range(len(pred_fold0)):
    scan_preds = [pred_fold[i] for pred_fold in preds]
    stacked_scan_preds = np.stack(scan_preds, axis=-1)
    majority_vote = np.sum(stacked_scan_preds, axis=-1) > 2
    final_segmentation = majority_vote.astype(np.uint8)
    filename = preds_list[0][i]
    
    original_pred = pred_fold0[i]
    image_meta = original_pred.meta
    image_meta['affine'] = image_meta['affine'].squeeze()

    original_filename = os.path.basename(image_meta["filename_or_obj"][0])
    if original_filename.endswith(".nii.gz"):
        stripped_filename = original_filename[:-7]  # Remove .nii.gz
    elif original_filename.endswith(".nii"):
        stripped_filename = original_filename[:-4]  # Remove .nii
    else:
        stripped_filename = original_filename  # No extension to strip

    image_meta["filename_or_obj"] = stripped_filename
    save_image(final_segmentation, meta_data=image_meta)

    print("done")

# %%
