import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes, # only for testing
    iou_width_height as iou,
    non_max_suppression as nms, # only for testing
    plot_image #only for testing
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(self,
                 csv_file,
                 img_dir, label_dir,
                 anchors,
                 image_size = 416,
                 S = [13,26,52],
                 C = 20,
                 transform = None,
                 ):
        #print(csv_file)
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[idx,1])
        # make the class lable at the last idx
        bboxes = np.roll(np.loadtxt(fname=label_path,delimiter=" ",ndmin = 2,),4,axis=1).tolist()# roll to reposition the output
        img_path = os.path.join(self.img_dir,self.annotations.iloc[idx,0])
        image = np.array(Image.open(img_path).convert("RGB"))
        #print("Image shape: ", np.array(Image.open(img_path)).shape)
        if self.transform:
            augmentations = self.transform(image=image,bboxes = bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        #the label
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        #print("target shape: ", targets[0].shape)
        #save all boxes and put the bounding box labels on targets
        #print("anchors:",self.anchors)
        for box in bboxes: #loop through each bounding boxex in label
            #calculate iou for a particular box and all anchors, return iou for all anchors
            iou_anchors = iou(torch.tensor(box[2:4]),self.anchors) # 2:4 means width and height
            #first one is the best anchor idx with highest iou
            anchor_indices = iou_anchors.argsort(descending=True,dim = 0)
            #print("sorted_anchors",anchor_indices)
            x, y, width, height, class_label = box
            has_anchor = False

            for anchor_idx in anchor_indices:
                # find the anchor on while scale, and which one on the scale
                scale_idx = anchor_idx // self.num_anchors_per_scale #0,1,2
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale #0,1,2
                S = self.S[scale_idx]
                #i, j are idx of cells in a specific scale4
                i ,j = int(S*y), int (S*x) #e.g: S= 13, x = 0.5, y = 0.6, x & y are relative to the image
                anchor_taken = targets[scale_idx][anchor_on_scale, i ,j , 0]

                if not anchor_taken: #if anchor hasn't been used
                    if not has_anchor: #if one scale can only have one anchor for this bbox
                        targets[scale_idx][anchor_on_scale, i ,j, 0] = 1
                        #print(targets[scale_idx].shape)
                        #following are the center and size of the box, but relative to
                        # the cell (before relative to 1 to 1 image
                        x_cell, y_cell = S*x - j, S*y - i # between 0,1, position in a cell
                        width_cell,height_cell = ( #width and height compared with size of the cell
                            width * S,
                            height * S
                        )

                        box_coordinates = torch.tensor(
                            [x_cell, y_cell, width_cell, height_cell]
                        )
                        targets[scale_idx][anchor_on_scale,i,j,1:5] = box_coordinates
                        targets[scale_idx][anchor_on_scale,i,j,5] = int(class_label)
                        #print(targets[scale_idx][anchor_on_scale,i,j])
                        #print("sample target bbox: ",targets[scale_idx][anchor_on_scale,i,j])

                        has_anchor = True
                        #print(targets[scale_idx].shape)

                    # if not the first anchor in the same scale, but also have high iou, then ignore it
                    elif iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                            #mark it so that it's not in the loss
                            targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        return image,tuple(targets)

def test():
    anchors = config.ANCHORS

    transform = config.train_transforms
    transform = None #to simplify testing

    dataset = YOLODataset(
        #config.DATASET+'/train.csv',
        "PASCAL_VOC/100examples.csv",
        config.IMG_DIR,
        config.LABEL_DIR,
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for x, y in loader:
        boxes = []
        #print("x,y from loader: ",x.shape,y[0].shape)

        for i in range(y[0].shape[1]): #for each of three scales
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]


        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print("boxes after nms: ",boxes)
        plot_image(x[0].to("cpu"), boxes)


if __name__ == "__main__":
    test()
