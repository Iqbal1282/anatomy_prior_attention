import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
        

class FinalClassificationLoss(nn.Module):
    def __init__(self, cfg):
        super(FinalClassificationLoss, self).__init__()
        self.classification_loss_fcn = F.binary_cross_entropy_with_logits
        self.cfg = cfg

    def forward(self, class_label_preds, class_label_gts, redundent=False):
        '''
        class_labels_pred => (batch, D)
        class_labels_gt => (batch, D)
        '''
        num_classes = class_label_gts.shape[1]
        device = class_label_gts.device
        pos_weights = torch.from_numpy(np.array(self.cfg.pos_weight)).to(device).type_as(class_label_gts)
        losses = torch.zeros(num_classes, device = device)

        for index in range(num_classes):
            label_pred = class_label_preds[:, index]
            label_gt = class_label_gts[:, index]
   
            if self.cfg.use_batch_weight:
                if label_gt.sum() != 0:
                    weight = (label_gt.shape[0] - label_gt.sum()) / label_gt.sum()
                    losses[index] = self.classification_loss_fcn(label_pred, label_gt, pos_weight=weight)
                else:
                    losses[index] = torch.tensor(0., requires_grad=True).to(device)        
            else:
                weight = pos_weights[index]
                losses[index] = self.classification_loss_fcn(label_pred, label_gt, pos_weight=weight)

        if redundent:
            return losses.detach().cpu().numpy()

        return losses.sum()

    

class FinalClassificationWithHMLoss(nn.Module):
    '''
    Heatmap - abnormality loss
    '''
    def __init__(self, cfg, gamma = 10):
        super(FinalClassificationWithHMLoss, self).__init__()
        self.classification_loss_fcn = nn.BCEWithLogitsLoss()
        self.gamma = gamma

    def forward(self, class_label_preds, class_label_gts, heatmaps, mask):
        classification_loss = self.classification_loss_fcn(class_label_preds, class_label_gts)
        
        _, _, H, W = heatmaps.shape
        small_mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
        heatmap_loss = (torch.sigmoid(heatmaps) * (small_mask == 0)).mean()

        return classification_loss + self.gamma * heatmap_loss