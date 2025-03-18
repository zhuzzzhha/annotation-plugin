import os
from typing import Literal, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
import pathlib

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------

class Conv2d(nn.Module):
    """ Perform a 2D convolution

    inputs are [b, c, h, w] where 
        b is the batch size
        c is the number of channels 
        h is the height
        w is the width
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 padding: int,
                 do_activation: bool = True, 
                 ):
        super(Conv2d, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]

        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)

    def forward(self, x):
        # x is [B, C, H, W]
        return self.conv(x)
    
# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

class _UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: List[int] = [64, 64, 64, 64, 64],
                 conv_kernel_size: int = 3,
                 conv: Optional[nn.Module] = None,
                 conv_kwargs: Dict[str,Any] = {}
                 ):
        """
        UNet (but can switch out the Conv)
        """
        super(_UNet, self).__init__()

        self.in_channels = in_channels

        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feat in features:
            self.downs.append(
                conv(
                    in_channels, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                )
            )
            in_channels = feat

        # Up part of U-Net
        for feat in reversed(features):
            self.ups.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.ups.append(
                conv(
                    # Factor of 2 is for the skip connections
                    feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                )
            )

        self.bottleneck = conv(
            features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
            )
        self.final_conv = conv(
            features[0], out_channels, kernel_size=1, padding=0, do_activation=False, **conv_kwargs
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
    

class UNet(_UNet):
    """
    Unet with normal conv blocks

    input shape: B x C x H x W
    output shape: B x C x H x W 
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(conv=Conv2d, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
        
checkpoint_dir = pathlib.Path(__file__).resolve().parent / "checkpoints"

class ScribblePromptUNet:

    weights = {
        "v1": checkpoint_dir / "ScribblePrompt_unet_v1_nf192_res128.pt"
    }
    
    def __init__(self, version: Literal["v1"] = "v1", device = None) -> None:
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.version = version
        self.device = device
        self.build_model(pretrained=True)
        self.input_size = (128,128)
    
    def build_model(self, pretrained: bool = True):
        """
        Build model
        """
        self.model = UNet(
            in_channels = 5,
            out_channels = 1,
            features = [192, 192, 192, 192],
        ).to(self.device)
        if pretrained:
            checkpoint_file = pathlib.Path("D:\\annotation-plugin\\checkpoints\\ScribblePrompt_unet_v1_nf192_res128.pt")
            assert os.path.exists(checkpoint_file), f"Checkpoint file not found: {checkpoint_file}. Please download from Dropbox"
            
            with open(checkpoint_file, "rb") as f:
                state = torch.load(f, map_location=self.device)
            
            self.model.load_state_dict(state)

    def forward(self, x):
        return self.model(x)
    
    def parameters(self):
        return self.model.parameters()
    
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device

    @torch.no_grad()
    def predict(self,
                img: torch.Tensor, # B x 1 x H x W
                point_coords: Optional[torch.Tensor] = None, # B x n x 2
                point_labels: Optional[torch.Tensor] = None, # B x n 
                scribbles: Optional[torch.Tensor] = None, # B x 2 x H x W
                box: Optional[torch.Tensor] = None, # B x 1 x 4
                mask_input: Optional[torch.Tensor] = None, # B x 1 x H x W
                return_logits: bool = False,
                ):
        """
        Make predictions from pytorch tensor inputs with grayscale images
        If the tensors are on the GPU it will prepare the inputs on GPU and retun the mask on the GPU

        Note: if batch size > 1, the number of clicks/boxes must be the same for each image in the batch

        Args:
            img: torch.Tensor (B x 1 x H x W) image to segment on [0,1] scale
            point_coords: torch.Tensor (B x n x 2) coordinates of pos/neg clicks in [x,y] format
            point_labels: torch.Tensor (B x n) labels of clicks (0 or 1)
            scribbles: torch.Tensor (B x 2 x H x W) pos/neg scribble inputs
            box: torch.Tensor (B x 1 x 4) bounding box inputs in [x1, y1, x2, y2] format
            mask_input: torch.Tensor (B x 1 x 128 x 128) logits of previous prediction
            return_logits: bool, if True return logits instead of mask on [0,1] scale
        
        Returns:
            mask (torch.Tensor): B x 1 x H x W prediction for each image in batch
        
        """
        assert (len(img.shape)==4) and (img.shape[1]==1), f"img shape should be B x 1 x H x W. current shape: {img.shape}"
        assert img.min() >= 0 and img.max() <= 1, f"img should be on [0,1] scale. current range: {img.min()} {img.max()}"
        
        prompts = {
            'img': img,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'scribbles': scribbles,
            'box': box,
            'mask_input': mask_input,
        }

        # Prepare inputs for ScribblePrompt unet (B x 5 x H x W)
        x = prepare_inputs(prompts).float().to(self.device)

        yhat = self.model(x)

        # B x 1 x H x W
        if return_logits:
            return yhat
        else:
            return torch.sigmoid(yhat)

    
# -----------------------------------------------------------------------------
# Prepare inputs
# -----------------------------------------------------------------------------

def rescale_inputs(inputs: Dict[str,any], input_size: Tuple[int] = (128,128)):
    """
    Rescale the inputs 
    """ 
    h,w = inputs['img'].shape[-2:]
    if [h,w] != input_size:
        
        inputs.update(dict(
            img = F.interpolate(inputs['img'], size=input_size, mode='bilinear')
        ))

        if inputs.get('scribbles') is not None:
            inputs.update({
                'scribbles': F.interpolate(inputs['scribbles'], size=input_size, mode='bilinear') 
            })
        
        if inputs.get("box") is not None:
            boxes = inputs.get("box").clone()
            coords = boxes.reshape(-1, 2, 2)
            coords[..., 0] = coords[..., 0] * (input_size[1] / w)
            coords[..., 1] = coords[..., 1] * (input_size[0] / h)
            inputs.update({'box': coords.reshape(1, -1, 4).int()})
        
        if inputs.get("point_coords") is not None:
            coords = inputs.get("point_coords").clone()
            coords[..., 0] = coords[..., 0] * (input_size[1] / w)
            coords[..., 1] = coords[..., 1] * (input_size[0] / h)
            inputs.update({'point_coords': coords.int()})

    return inputs

def prepare_inputs(inputs: Dict[str,torch.Tensor], device = None) -> torch.Tensor:
    """
    Prepare inputs for network

    Returns: 
        x (torch.Tensor): B x 5 x H x W
    """
    img = inputs['img']
    if device is None:
        device = img.device

    img = img.to(device)
    shape = tuple(img.shape[-2:])
    
    if inputs.get("box") is not None:
        # Embed bounding box
        # Input: B x 1 x 4 
        # Output: B x 1 x H x W
        box_embed = bbox_shaded(inputs['box'], shape=shape, device=device)
    else:
        box_embed = torch.zeros(img.shape, device=device)

    if inputs.get("point_coords") is not None:
        # Embed points
        # B x 2 x H x W
        scribble_click_embed = click_onehot(inputs['point_coords'], inputs['point_labels'], shape=shape)
    else:
        scribble_click_embed = torch.zeros((img.shape[0], 2) + shape, device=device)

    if inputs.get("scribbles") is not None:
        # Combine scribbles with click embedding
        # B x 2 x H x W
        scribble_click_embed = torch.clamp(scribble_click_embed + inputs.get('scribbles'), min=0.0, max=1.0)

    if inputs.get('mask_input') is not None:
        # Previous prediction
        mask_input = inputs['mask_input']
    else:
        # Initialize empty channel for mask input
        mask_input = torch.zeros(img.shape, device=img.device)

    print(f'img {img.shape} box_embed {box_embed.shape} scribble_click_embed {scribble_click_embed.shape} mask_input {mask_input.shape}')
    x = torch.cat((img, box_embed, scribble_click_embed, mask_input), dim=-3)
    # B x 5 x H x W

    return x
    
# -----------------------------------------------------------------------------
# Encode clicks and bounding boxes
# -----------------------------------------------------------------------------

def click_onehot(point_coords, point_labels, shape: Tuple[int,int] = (128,128), indexing: Literal['xy','uv'] ='xy'):
    """
    Represent clicks as two HxW binary masks (one for positive clicks and one for negative) 
    with 1 at the click locations and 0 otherwise

    Args:
        point_coords (torch.Tensor): BxNx2 tensor of xy coordinates
        point_labels (torch.Tensor): BxN tensor of labels (0 or 1)
        shape (tuple): output shape     
    Returns:
        embed (torch.Tensor): Bx2xHxW tensor 
    """
    assert len(point_coords.shape) == 3, "point_coords must be BxNx2"
    assert point_coords.shape[-1] == 2, "point_coords must be BxNx2"
    assert point_labels.shape[-1] == point_coords.shape[1], "point_labels must be BxN"
    assert len(shape)==2, f"shape must be 2D: {shape}"

    device = point_coords.device
    batch_size = point_coords.shape[0]
    n_points = point_coords.shape[1]

    embed = torch.zeros((batch_size,2)+shape, device=device)
    labels = point_labels.flatten().float()

    idx_coords = torch.cat((
        torch.arange(batch_size, device=device).reshape(-1,1).repeat(1,n_points)[...,None], 
        point_coords
    ), axis=2).reshape(-1,3)

    if indexing=='xy':
        embed[ idx_coords[:,0], 0, idx_coords[:,2], idx_coords[:,1] ] = labels
        embed[ idx_coords[:,0], 1, idx_coords[:,2], idx_coords[:,1] ] = 1.0-labels
    else:
        embed[ idx_coords[:,0], 0, idx_coords[:,1], idx_coords[:,2] ] = labels
        embed[ idx_coords[:,0], 1, idx_coords[:,1], idx_coords[:,2] ] = 1.0-labels

    return embed


def bbox_shaded(boxes, shape: Tuple[int,int] = (128,128), device='cpu'):
    """
    Represent a bounding box as a binary mask with 1 inside the box and 0 outside

    Args:
        boxes (torch.Tensor): Bx1x4 [x1, y1, x2, y2]
    Returns:
        bbox_embed (torch.Tesor): Bx1xHxW according to shape
    """
    assert len(shape)==2, "shape must be 2D"
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.int().cpu().numpy()

    batch_size = boxes.shape[0]
    n_boxes = boxes.shape[1]
    bbox_embed = torch.zeros((batch_size,1)+tuple(shape), device=device, dtype=torch.float32)

    if boxes is not None:
        for i in range(batch_size):
            for j in range(n_boxes):
                x1, y1, x2, y2 = boxes[i,j,:]
                x_min = min(x1,x2)
                x_max = max(x1,x2)
                y_min = min(y1,y2)
                y_max = max(y1,y2)
                bbox_embed[ i, 0, y_min:y_max, x_min:x_max ] = 1.0

    return bbox_embed

def show_scribbles(mask, ax, alpha=0.5):
    """
    Overlay positive scribbles (green) and negative (red) scribbles 
    Args:
        mask: 1 x (C) x H x W or 2 x (C) x H x W
        ax: matplotlib axis
        alpha: transparency of the overlay
    """
    mask = mask.squeeze() # 2 x H x W
    if len(mask.shape)==2:
        # If there's only channel of scribbles, overlay the scribbles in blue
        h, w = mask.shape
        color = np.array([30/255, 144/255, 255/255, alpha])
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    elif len(mask.shape)==3:
        # If there are 2 channels, take the first channel as positive scribbles (green) and the second channel as negaitve scribbles (red)
        c, h, w = mask.shape
        color = np.array([0, 1, 0, alpha])
        mask_image = mask[0,...].reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        color = np.array([1, 0, 0, alpha])
        mask_image = mask[1,...].reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    else:
        raise ValueError("mask must be 2 or 3 dimensional")
    
def show_mask(mask, ax, random_color=False, alpha=0.5):
    """
    Overlay mask
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)