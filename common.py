"""Shared image processing and manifest handling. All variants use identical transforms."""
import csv
import hashlib
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / 'data' / 'dataset_split'
CLASSES = ['0','1','2','3','4','5','6','7','8','OK']
EXTENSIONS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_bgr(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode image: '+str(path))
    return image

def save_bgr(path, image):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('Cannot encode '+str(path))
    buffer.tofile(str(path))

def image_tensor(image, size=64, gray=True, augment=False):
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size,size), interpolation=cv2.INTER_AREA)
    if augment:
        if random.random() < 0.3:
            image = cv2.flip(image, 1)
        matrix = cv2.getRotationMatrix2D(((size-1)/2, (size-1)/2), random.uniform(-10,10), 1.0)
        image = cv2.warpAffine(image, matrix, (size,size), borderMode=cv2.BORDER_REFLECT_101)
    array = image.astype(np.float32) / 255.0
    array = (array - np.array([0.485,0.456,0.406], np.float32)) / np.array([0.229,0.224,0.225],np.float32)
    return torch.from_numpy(array.transpose(2,0,1).copy())

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed); random.seed(seed)

def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_records(manifest, split):
    records = [json.loads(line) for line in Path(manifest).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    records = [r for r in records if r['split'] == split]
    if not records:
        raise ValueError('No records for split '+split)
    for row in records:
        if row['label'] != CLASSES[row['target']]:
            raise ValueError('Inconsistent label mapping')
    return records

class GestureDataset(Dataset):
    def __init__(self, manifest, split, view='raw', size=64, gray=True, augment=False):
        self.records = load_records(manifest, split)
        self.view, self.size, self.gray, self.augment = view, size, gray, augment
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]
        # A missed detection uses the full image, never silently drops the sample.
        path = row['crop_path'] if self.view == 'mp' and row['detected'] else row['path']
        return image_tensor(read_bgr(path), self.size, self.gray, self.augment), row['target']
