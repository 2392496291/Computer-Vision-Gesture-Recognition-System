"""Shared checkpoint loading and threshold-based decision fusion."""
import numpy as np
import torch
from common import CLASSES,image_tensor
from models import GestureModel

def load_model(checkpoint,device):
    saved=torch.load(checkpoint,map_location=device,weights_only=True)
    config=saved['config']
    if config['classes']!=CLASSES:
        raise ValueError('Checkpoint class order mismatch')
    model=GestureModel(config['model'],len(CLASSES),pretrained=False).to(device)
    model.load_state_dict(saved['state_dict']); model.eval()
    return model,config

def fuse(scores,rule,detected,threshold):
    scores=np.asarray(scores,dtype=float)
    if not detected or scores.max()>=threshold or rule not in CLASSES:
        return scores.copy(),int(scores.argmax()),'CNN' if detected else 'CNN full-image fallback'
    output=np.zeros(len(CLASSES)); target=CLASSES.index(rule); output[target]=1.
    return output,target,'3D-normalised landmark rule'

def predict_image(model,config,image,device):
    tensor=image_tensor(image,config['size'],config['gray']).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).float().softmax(1)[0].cpu().numpy()
