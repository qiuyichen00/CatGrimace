import torch
import torch.nn as nn

import config
from model import YOLOv3
from utils import intersection_over_union, load_checkpoint, get_loaders
import torch.optim as optim
from tqdm import tqdm #smart loop progress meter
from utils import (
    iou_width_height as iou,
    non_max_suppression as nms, # only for testing
)

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss() #contains sigmoid and bce
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        #Constants
        self.lambda_class = 1
        self.lambda_noobj = 10 #value no objects more
        self.lambda_obj = 10
        self.lambda_box = 1

    def forward(self, predictions, target, anchors): #for each scale
        obj = target[...,0] == 1 # if one anchor per labeled box, then obj can be empty
        noobj = target[...,0] == 0
        #ignore all spots with -1
        #print(predictions[...,0:1][noobj])
        #No Object Loss, if target i has no object, then compare target i and predictions i with bce
        no_object_loss = self.bce( #binary class entropy
            #predictions[...,0:1][noobj],
            predictions[..., 0:1][noobj],
            target[...,0:1][noobj]
        )



        #Object Loss, if target i has object, then compare target i and (predictions i) * iou with bce
        #calculate iou between target boxes and predictions,
        #sometimes NaN because not every target (one from three targets) contains object, so zero obj length
        anchors = anchors.reshape(1,3,1,1,2) #reshape for broadcasting
        # x,y (0 to 1 from sigmoid)
        box_preds = torch.cat([self.sigmoid(predictions[...,1:3]),
                               torch.exp(predictions[...,3:5]) * anchors],dim=-1)# b_w = p_w * exp(t_w), p is anchor box dimension
        ious = intersection_over_union(box_preds[obj], target[...,1:5][obj]).detach()

        object_loss = self.bce(predictions[...,0:1][obj],ious * target[...,0:1][obj])


        #box coordinate loss
        predictions[...,1:3] = self.sigmoid(predictions[...,1:3]) #x,y between 0 and 1
        #invert bounding boxes to model prediction
        target[...,3:5] = torch.log(1e-16 + target[...,3:5] / anchors)
        box_loss = self.mse(predictions[...,1:5][obj],target[...,1:5][obj])



        #class loss
        class_loss = self.entropy(predictions[...,5:][obj], target[...,5][obj].long())


        if no_object_loss != no_object_loss: #if nan because some targets don't have bboxes
            no_object_loss = 0
        if object_loss != object_loss:
            object_loss = 0
        if box_loss != box_loss:
            box_loss = 0
        if class_loss != class_loss:
            class_loss = 0





        output = self.lambda_box * box_loss\
                + self.lambda_obj * object_loss\
                + self.lambda_noobj * no_object_loss\
                + self.lambda_class * class_loss
        temp = [no_object_loss,object_loss,box_loss,class_loss]
        #print(temp[0].item(),temp[1].item(),temp[2].item(),temp[3].item())
        return output
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr = config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY,
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path = config.train_csv_path, test_csv_path=config.test_csv_path,
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1,3,2)).to(config.DEVICE)

    batch = next(iter(train_loader))
    (x,y) = batch
    x = x.to(config.DEVICE)
    y0, y1, y2 = (
        y[0].to(config.DEVICE),
        y[1].to(config.DEVICE),
        y[2].to(config.DEVICE)
    )
    with torch.cuda.amp.autocast():  # automatically casting data type
        out = model(x)  # have three out
        loss = (  # scale anchor to map the anchor size based on cell size
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
        )
        print(loss)

