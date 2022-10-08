import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_heatmap(raw_heatmap, size=(224, 224)):
    heatmap =  torch.sigmoid(raw_heatmap).permute(1, 2, 0).detach().cpu().numpy() * 255
    heatmap = cv2.resize(heatmap.astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
    return heatmap.transpose(2, 0, 1)


def process_heatmap1(raw_heatmap, size=(224, 224)):
    heatmap =  (1 - torch.sigmoid(raw_heatmap)).permute(1, 2, 0).detach().cpu().numpy() * 255
    heatmap = cv2.resize(heatmap.astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
    return heatmap.transpose(2, 0, 1)
    

def process_image(image):
    image =  image.permute(1, 2, 0).detach().cpu().numpy()
    image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    return image.astype(np.uint8)


def process_bbox_gts(bbox_gts):
    bbox_gts = bbox_gts.detach().cpu().numpy()
    return bbox_gts.astype(np.uint8)


def heatmap_to_bboxes(heatmap):
    '''
    heatmap -> (num_diseases, H, W)
    bbox_list -> (num_diseases, K, 4)
    '''
    N = heatmap.shape[0]
    bbox_list = [[]  for i in range(N)]

    for i in range(N):
        # thresh = cv2.threshold(heatmap[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.threshold(heatmap[i], 127, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            bbox_list[i].append(cv2.boundingRect(c))
        bbox_list[i] = np.array(bbox_list[i])
    
    # bbox_list = np.array(bbox_list)
    for i in range(N):
        if bbox_list[i].any():
            widths = bbox_list[i][:, 2]
            heights = bbox_list[i][:, 3]
            bbox_list[i][:, 2] = bbox_list[i][:, 0] + widths
            bbox_list[i][:, 3] = bbox_list[i][:, 1] + heights
        else:
            bbox_list[i] = np.zeros((1, 4), dtype=np.uint8)
            
    return bbox_list


pathology_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax']

def show_overlayed_heatmap(image, heatmaps, mode=3):
    '''
    image => (H, W, C)
    heatmaps => (N, H, W)
    mode:
    0 -> show image with heatmap
    1 -> show image with bbox
    3 -> show image with heatmap and bbox
    4 -> show only image
    '''
    bbox_list = heatmap_to_bboxes(heatmaps)
    if mode == 4:
        assert len(image.shape == 3) and image.shape[2] == 3
        plt.imshow(image)
        plt.axis('off')
    else:
        fig, ax = plt.subplots(2, 4, figsize=(18, 9))
        for i, pathology in enumerate(pathology_list):
            current_axis = ax[i//4, i%4]
            current_axis.axis('off')
            current_axis.set_title(pathology)

            modified_image = image.copy()
            if mode == 1 or mode == 3:
                for c in bbox_list[i]:
                    x1, y1, x2, y2 = c
                    cv2.rectangle(modified_image, (x1, y1), (x2, y2), (0,0,0), 2)
            if mode == 0 or mode == 3:
                heatmap = cv2.applyColorMap(heatmaps[i], cv2.COLORMAP_JET)
                modified_image = cv2.addWeighted(modified_image, 0.7, heatmap, 0.3, 0)
            
            current_axis.imshow(modified_image)
        plt.show()
        


def show_overlayed_bboxes(image, bbox_list):
    '''
    image => (H, W, C)
    bbox_list => (D, 4) : x1, y1, x2, y2
    '''
    fig, ax = plt.subplots(2, 4, figsize=(18, 9))
    for i, pathology in enumerate(pathology_list):
        current_axis = ax[i//4, i%4]
        current_axis.axis('off')
        current_axis.set_title(pathology)

        modified_image = image.copy()
        x1, y1, x2, y2 = bbox_list[i]
        cv2.rectangle(modified_image, (x1, y1), (x2, y2), (36,255,12), 2)
            
        current_axis.imshow(modified_image)
    plt.show()

    
def bbox_area(bboxes):
    '''
    (x1, y1, x2, y2)
    '''
    x1, y1, x2, y2 = bboxes
    return np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)


def bbox_intersection(box1, box2):
    '''
    (x1, y1, x2, y2)
    '''
    inter_x1 = np.maximum(box1[0], box2[0])
    inter_y1 = np.maximum(box1[1], box2[1])
    inter_x2 = np.minimum(box1[2], box2[2])
    inter_y2 = np.minimum(box1[3], box2[3])
    return bbox_area([inter_x1, inter_y1, inter_x2, inter_y2])


def _iou(pred_bboxes, gt_bbox):
    '''
    pred_bboxes -> (K, 4)
    gt_bbox -> (4)
    '''
    eps = 1e-8
    intersection_list = []
    area_list = []
    for pred_bbox in pred_bboxes:
        area_list.append(bbox_area(pred_bbox))
        intersection_list.append(bbox_intersection(pred_bbox, gt_bbox))

    area_list = np.array(area_list)
    intersection_list = np.array(intersection_list)
    gt_bbox_area = bbox_area(gt_bbox)
    intersection = intersection_list.sum()
    union = area_list.sum() + gt_bbox_area - intersection

    if intersection == 0 and union == 0:
        return 1

    return intersection / (union + eps)


def _iobb(pred_bboxes, gt_bbox):
    '''
    pred_bboxes -> (K, 4)
    gt_bbox -> (4))
    '''
    eps = 1e-8
    intersection_list = []
    area_list = []
    for pred_bbox in pred_bboxes:
        area_list.append(bbox_area(pred_bbox))
        intersection_list.append(bbox_intersection(pred_bbox, gt_bbox))

    area_list = np.array(area_list)
    intersection_list = np.array(intersection_list)
    intersection = intersection_list.sum()
    pred_area = area_list.sum()

    return intersection / (pred_area + eps)


def iou(pred_bboxes, gt_bboxes):
    '''
    pred_bboxes -> (D, K, 4)
    gt_bboxes -> (D, 4)
    '''
    D = len(gt_bboxes)
    iou_list = []
    for i in range(D):
        pred_bbox = pred_bboxes[i]
        gt_bbox = gt_bboxes[i]
        iou_list.append(_iou(pred_bbox, gt_bbox))
    
    iou_list = np.array(iou_list)
    return iou_list


def iobb(pred_bboxes, gt_bboxes):
    '''
    pred_bboxes -> (D, K, 4)
    gt_bboxes -> (D, 4)
    '''
    D = len(gt_bboxes)
    iobb_list = []
    for i in range(D):
        pred_bbox = pred_bboxes[i]
        gt_bbox = gt_bboxes[i]
        iobb_list.append(_iobb(pred_bbox, gt_bbox))
    
    iobb_list = np.array(iobb_list)
    return iobb_list


