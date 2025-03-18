import numpy as np
import torch
import torch.nn.functional as F
import os
import cv2
import pathlib
from typing import Dict, Optional

from unet import rescale_inputs, prepare_inputs, UNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RES = 256

file_dir = pathlib.Path(os.path.dirname(__file__))
example_dir = file_dir / "examples"

#test_examples = [str(example_dir / x) for x in sorted(os.listdir(example_dir)) if not x.endswith('.npy')]
#default_example = test_examples[0]

exp_dir = file_dir / "../checkpoints" 
default_model = 'ScribblePrompt-Unet'

model_dict = {
    'ScribblePrompt-Unet': 'ScribblePrompt_unet_v1_nf192_res128.pt'
}

# -----------------------------------------------------------------------------
# Model Predictor Class
# -----------------------------------------------------------------------------

class Predictor:
    """
    Wrapper for ScribblePrompt-UNet model
    """
    def __init__(self, path: str, verbose: bool = True):
        
        assert path.exists(), f"Checkpoint {path} does not exist"
        
        self.path = path
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.load()
        self.model.eval()
        self.to_device()

    def build_model(self):
        """
        Build the model
        """
        self.model = UNet(
            in_channels = 5,
            out_channels = 1,
            features = [192, 192, 192, 192],
        )

    def load(self):
        """
        Load the state of the model from a checkpoint file.
        """
        with (self.path).open("rb") as f:
            state = torch.load(f, map_location=self.device)
            self.model.load_state_dict(state, strict=True)
            if self.verbose:
                print(
                    f"Loaded checkpoint from {self.path} to {self.device}"
                )
        
    def to_device(self):
        """
        Move the model to cpu or gpu
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def predict(self, prompts: Dict[str,any], img_features: Optional[torch.Tensor] = None, multimask_mode: bool = False):
        """
        Make predictions!

        Returns:
            mask (torch.Tensor): H x W
            img_features (torch.Tensor): B x 1 x H x W (for SAM models)
            low_res_mask (torch.Tensor): B x 1 x H x W logits
        """
        if self.verbose:
            print("point_coords", prompts.get("point_coords", None))
            print("point_labels", prompts.get("point_labels", None))
            print("box", prompts.get("box", None))
            print("img", prompts.get("img").shape, prompts.get("img").min(), prompts.get("img").max())
            if prompts.get("scribbles") is not None:
                print("scribbles", prompts.get("scribbles", None).shape, prompts.get("scribbles").min(), prompts.get("scribbles").max())

        original_shape = prompts.get('img').shape[-2:]

        # Rescale to 128 x 128
        prompts = rescale_inputs(prompts)

        # Prepare inputs for ScribblePrompt unet (1 x 5 x 128 x 128)
        x = prepare_inputs(prompts).float()

        with torch.no_grad():
            yhat = self.model(x.to(self.device)).cpu()

        mask = torch.sigmoid(yhat)

        # Resize for app resolution
        mask = F.interpolate(mask, size=original_shape, mode='bilinear').squeeze()

        # mask: H x W, yhat: 1 x 1 x H x W
        return mask, None, yhat

# -----------------------------------------------------------------------------
# Model initialization functions
# -----------------------------------------------------------------------------

def load_model(exp_key: str = default_model):
    fpath = exp_dir / model_dict.get(exp_key)
    exp = Predictor(fpath)
    return exp, None

# -----------------------------------------------------------------------------
# Vizualization functions
# -----------------------------------------------------------------------------

def _get_overlay(img, lay, const_color="l_blue"):
    """
    Helper function for preparing overlay
    """
    assert lay.ndim==2, "Overlay must be 2D, got shape: " + str(lay.shape)

    if img.ndim == 2:
        img = np.repeat(img[...,None], 3, axis=-1)

    assert img.ndim==3, "Image must be 3D, got shape: " + str(img.shape)

    if const_color == "blue":
        const_color = 255*np.array([0, 0, 1])
    elif const_color == "green":
        const_color = 255*np.array([0, 1, 0])
    elif const_color == "red":
        const_color = 255*np.array([1, 0, 0])
    elif const_color == "l_blue":
        const_color = np.array([31, 119, 180]) 
    elif const_color == "orange":
        const_color = np.array([255, 127, 14]) 
    else:
        raise NotImplementedError
       
    x,y = np.nonzero(lay)
    for i in range(img.shape[-1]):
        img[x,y,i] = const_color[i]

    return img

def image_overlay(img, mask=None, scribbles=None, contour=False, alpha=0.5):
    """
    Overlay the ground truth mask and scribbles on the image if provided
    """
    assert img.ndim == 2, "Image must be 2D, got shape: " + str(img.shape)
    output = np.repeat(img[...,None], 3, axis=-1)

    if mask is not None:
        
        assert mask.ndim == 2, "Mask must be 2D, got shape: " + str(mask.shape)
        
        if contour:
            contours = cv2.findContours((mask[...,None]>0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours[0], -1, (0, 255, 0), 1)
        else:
            mask_overlay = _get_overlay(img, mask)
            mask2 = 0.5*np.repeat(mask[...,None], 3, axis=-1)
            output = cv2.convertScaleAbs(mask_overlay * mask2 + output * (1 - mask2))
    
    if scribbles is not None:
        pos_scribble_overlay = _get_overlay(output, scribbles[0,...], const_color="green")
        cv2.addWeighted(pos_scribble_overlay, alpha, output, 1 - alpha, 0, output)
        neg_scribble_overlay = _get_overlay(output, scribbles[1,...], const_color="red")
        cv2.addWeighted(neg_scribble_overlay, alpha, output, 1 - alpha, 0, output)

    return output
    

def viz_pred_mask(img, mask=None, point_coords=None, point_labels=None, bbox_coords=None, seperate_scribble_masks=None, binary=True):
    """
    Visualize image with clicks, scribbles, predicted mask overlaid
    """
    assert isinstance(img, np.ndarray), "Image must be numpy array, got type: " + str(type(img))
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

    if binary and mask is not None:
        mask = 1*(mask > 0.5)

    out = image_overlay(img, mask=mask, scribbles=seperate_scribble_masks)

    if point_coords is not None:
        for i,(col,row) in enumerate(point_coords):
            if point_labels[i] == 1:
                cv2.circle(out,(col, row), 2, (0,255,0), -1)
            else:
                cv2.circle(out,(col, row), 2, (255,0,0), -1)

    if bbox_coords is not None:
        for i in range(len(bbox_coords)//2):
            cv2.rectangle(out, bbox_coords[2*i], bbox_coords[2*i+1], (255,165,0), 1)
        if len(bbox_coords) % 2 == 1:
            cv2.circle(out, tuple(bbox_coords[-1]), 2, (255,165,0), -1)

    return out

# -----------------------------------------------------------------------------
# Collect scribbles
# -----------------------------------------------------------------------------

def get_scribbles(seperate_scribble_masks, last_scribble_mask, scribble_img, label: int):
    """
    Record scribbles
    """
    assert isinstance(seperate_scribble_masks, np.ndarray),\
         "seperate_scribble_masks must be numpy array, got type: " + str(type(seperate_scribble_masks))

    if scribble_img is not None:
        
        color_mask = scribble_img.get('mask')
        scribble_mask = color_mask[...,0]/255
        
        not_same = (scribble_mask != last_scribble_mask)
        if not isinstance(not_same, bool):
            not_same = not_same.any()

        if not_same:
            # In case any scribbles were removed
            corrected_scribble_masks = np.stack(2*[(scribble_mask > 0)], axis=0)*seperate_scribble_masks
            corrected_last_scribble_mask = last_scribble_mask*(scribble_mask > 0)

            delta = (scribble_mask - corrected_last_scribble_mask) > 0
            new_scribbles = scribble_mask * delta
            corrected_scribble_masks[label,...] = np.clip(corrected_scribble_masks[label,...] + new_scribbles, a_min=0, a_max=1)

            last_scribble_mask = scribble_mask
            seperate_scribble_masks = corrected_scribble_masks

        return seperate_scribble_masks, last_scribble_mask

def get_predictions(predictor, input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks, low_res_mask, img_features, multimask_mode):
    """
    Make predictions
    """
    box = None
    if len(bbox_coords) == 1:
        #gr.Error("Please click a second time to define the bounding box")
        box = None
    elif len(bbox_coords) == 2:
        box = torch.Tensor(bbox_coords).flatten()[None,None,...].int().to(device) # B x n x 4

    if seperate_scribble_masks is not None:
        scribble = torch.from_numpy(seperate_scribble_masks)[None,...].to(device)
    else:
        scribble = None  
    
    prompts = dict(
        img=torch.from_numpy(input_img)[None,None,...].to(device)/255, 
        point_coords=torch.Tensor([click_coords]).int().to(device) if len(click_coords)>0 else None, 
        point_labels=torch.Tensor([click_labels]).int().to(device) if len(click_labels)>0 else None,
        scribbles=scribble,
        mask_input=low_res_mask.to(device) if low_res_mask is not None else None, 
        box=box,
        )
    
    mask, img_features, low_res_mask = predictor.predict(prompts, img_features, multimask_mode=multimask_mode)

    return mask, img_features, low_res_mask

def refresh_predictions(predictor, input_img, output_img, click_coords, click_labels, bbox_coords, brush_label,
                        scribble_img, seperate_scribble_masks, last_scribble_mask, 
                        best_mask, low_res_mask, img_features, binary_checkbox, multimask_mode):
    
    # Record any new scribbles
    seperate_scribble_masks, last_scribble_mask = get_scribbles(
        seperate_scribble_masks, last_scribble_mask, scribble_img, 
        label=(0 if brush_label == "Positive (green)" else 1) # current color of the brush
    )

    # Make prediction
    best_mask, img_features, low_res_mask = get_predictions(
        predictor, input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks, low_res_mask, img_features, multimask_mode
    )

    # Update input visualizations
    mask_to_viz = best_mask.numpy()
    click_input_viz = viz_pred_mask(input_img, mask_to_viz, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox)
    scribble_input_viz = viz_pred_mask(input_img, mask_to_viz, click_coords, click_labels, bbox_coords, None, binary_checkbox)
    
    out_viz = [
        viz_pred_mask(input_img, mask_to_viz, point_coords=None, point_labels=None, bbox_coords=None, seperate_scribble_masks=None, binary=binary_checkbox),
        255*(mask_to_viz[...,None].repeat(axis=2, repeats=3)>0.5) if binary_checkbox else mask_to_viz[...,None].repeat(axis=2, repeats=3),
    ]
    
    return click_input_viz, scribble_input_viz, out_viz, best_mask, low_res_mask, img_features, seperate_scribble_masks, last_scribble_mask


