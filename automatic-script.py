import json
import os
from PIL import Image
import cv2
import numpy as np
import torch
from model import ScribblePromptUNet
import torch.nn.functional as F
from skimage.segmentation import find_boundaries
from skimage.io import imread 
from scipy.ndimage import binary_erosion, binary_dilation
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("D:\\Anaconda3\\Lib\\site-packages")
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence



class Evaluator:
    def __init__(self, pos_points, neg_points, input_img_path, gt_path, pos_margin, neg_margin, threshold):
          self.pos_points = pos_points
          self.neg_points = neg_points
          self.pos_points_for_prediction = []
          self.neg_points_for_prediction = []
          self.img = Image.open(input_img_path)
          self.threshold = threshold

          first_image = imread(gt_path)
          image_height, image_width = first_image.shape[:2]

          self.gt = np.zeros((1, image_height, image_width), dtype=first_image.dtype)

          image = imread(gt_path)

          self.gt = image

          self.pos_margin = pos_margin
          self.neg_margin = neg_margin

          self.map_result = {}
          self.result_data = []

    def find_boundary_with_margin(self, intensity_profile, pos_margin, neg_margin):
        diffs = np.abs(np.diff(intensity_profile))
        
        max_diff_idx = np.argmax(diffs)
        
        margin_right = int(len(intensity_profile) * pos_margin)
        margin_left = int(len(intensity_profile) * neg_margin)
        
        left_bound = max(0, max_diff_idx - margin_right)
        right_bound = min(len(intensity_profile)-1, max_diff_idx + margin_left)
        
        return max_diff_idx, left_bound, right_bound

    def bresenham_line(self, img, x0, y0, x1, y1):
            x0 = int(x0)
            x1 = int(x1)
            y0 = int(y0)
            y1 = int(y1)
            points = []
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            
            while True:
                points.append((x0, y0))
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy

            brightness_values = []
            for point in points:
                brightness = img.getpixel((point[1], point[0]))  # napari использует (row, col) = (y, x)
                brightness_values.append(brightness)
            return brightness_values, points

    def get_points(self, img, pos_point, neg_point, pos_margin, neg_margin):
        brightness, points = self.bresenham_line(img, *pos_point, *neg_point)
        max_diff_idx, beg, end = self.find_boundary_with_margin(brightness, self.pos_margin, self.neg_margin)
        pos_idx = [idx for idx in range(0, beg+1)]
        neg_idx = [idx for idx in range(end, len(brightness))]
        pos_points = [[points[idx][0], points[idx][1]] for idx in pos_idx]
        neg_points = [[points[idx][0], points[idx][1]] for idx in neg_idx]
        return pos_points, neg_points

    def boundary_iou(self, pred_mask, gt_mask, threshold, dilation_radius=3):
        assert pred_mask.shape == gt_mask.shape, "Маски должны иметь одинаковый размер"
        
        pred_mask = (pred_mask > threshold).numpy().astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)

        pred_boundary = find_boundaries(pred_mask, mode='inner')
        gt_boundary = find_boundaries(gt_mask, mode='inner')
        
        if dilation_radius > 0:
            struct = np.ones((2*dilation_radius+1, 2*dilation_radius+1))
            pred_boundary = binary_dilation(pred_boundary, structure=struct)
            gt_boundary = binary_dilation(gt_boundary, structure=struct)
        
        intersection = np.logical_and(pred_boundary, gt_boundary).sum()
        union = np.logical_or(pred_boundary, gt_boundary).sum()
        
        boundary_iou = intersection / union if union > 0 else 0.0
        
        return boundary_iou
    
    def dice_coefficient(self, pred_mask, gt_mask, threshold, epsilon=1e-6):
        
        assert pred_mask.shape == gt_mask.shape, "Маски должны иметь одинаковый размер"
        
        pred_mask = (pred_mask > threshold).numpy().astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        sum_masks = pred_mask.sum() + gt_mask.sum()
        
        dice = (2. * intersection + epsilon) / (sum_masks + epsilon)
        
        return dice
    
    def calculate_iou(self, pred_mask, gt_mask, threshold):
        pred_mask = (pred_mask > threshold).numpy().astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)


        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()

        if union > 0:
            iou = intersection / union

        return iou
    
    def calc_quality(self, img, gt, pos_points, neg_points, threshold):
        original_shape = (img.height, img.width)
        img = torch.tensor(np.asarray(img.resize((128,128)).convert('L')))/255
        img = img[None,None,...].float()
        point_coords = torch.tensor([[ [66, 26], *neg_points, *pos_points]])
        point_labels = torch.tensor(np.array([[0] * (point_coords.shape[1])], dtype=np.int32))
        point_labels[:, 1 + (len(neg_points)):(point_coords.shape[1])+1] = 1
            
        scribbles = None

        sp_unet = ScribblePromptUNet(version="v1")
        mask_unet = sp_unet.predict(img=img, scribbles=scribbles, point_coords=point_coords, point_labels=point_labels)
        mask = F.interpolate(mask_unet, size=original_shape, mode='bilinear').squeeze()

        iou = self.calculate_iou(mask, gt, threshold)
        biou = self.boundary_iou(mask, gt, threshold)
        dice = self.dice_coefficient(mask, gt, threshold)
        return iou, biou, dice
    
    def dilation_around_point(self, center, radius, shape):

        x_center, y_center = center
        height, width = shape

        points = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = x_center + dx
                y = y_center + dy

                if 0 <= x < width and 0 <= y < height and (dx**2 + dy**2) <= radius**2:
                    points.append((x, y))

        return points
    
    def evaluate(self):
        for i in range(len(self.pos_points)): 
            pos_points, neg_points = self.get_points(self.img, self.pos_points[i], self.neg_points[i], self.pos_margin, self.neg_margin)
            self.pos_points_for_prediction.extend(pos_points)
            self.neg_points_for_prediction.extend(neg_points)
        iou, biou, dice = self.calc_quality(self.img, self.gt, self.pos_points_for_prediction, self.neg_points_for_prediction, self.threshold)
        return dice

def objective_function(params):
    quality_scores = []
    instance = Evaluator([[35, 73], [94, 52], [80, 23]],
                    [[35, 107], [97, 77], [76, 33]],
                    "INPUT_PATH",
                    "OUTPUT_PATH",
                    params[0],
                    params[1],
                    params[2]
                    )
    
    score = instance.evaluate()
    
    return -score

space = [Real(0.0, 1.0, name='x1'),  
         Real(0.0, 0.4, name='x2'),
         Real(0.1, 0.9, name='x3')] 

result = gp_minimize(objective_function,      
                     space,                   
                     n_calls=100,              
                     random_state=42)         

print("Best parameters: x1 = {:.4f}, x2 = {:.4f} x3 = {:.4f}".format(result.x[0], result.x[1], result.x[2]))
print("Minimum value: {:.4f}".format(result.fun))

plot_convergence(result)