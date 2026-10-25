"""Check dependencies, dataset paths and optional CUDA support."""
import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from importlib.metadata import version
from common import ROOT,DEFAULT_DATA,write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda',action='store_true')
    args=parser.parse_args()
    modules={'torch':'torch','torchvision':'torchvision','opencv-contrib-python':'cv2',
             'numpy':'numpy','mediapipe':'mediapipe','scikit-learn':'sklearn','tqdm':'tqdm','PyQt5':'PyQt5.QtCore'}
    installed={}
    for distribution,module in modules.items():
        importlib.import_module(module)
        installed[distribution]=version(distribution)
    import torch
    import mediapipe as mp
    if not hasattr(mp,'solutions'):
        raise RuntimeError('This project requires mediapipe.solutions.hands')
    for split in ('train','val','test'):
        if not (DEFAULT_DATA/split).is_dir():
            raise RuntimeError('Dataset is missing: '+str(DEFAULT_DATA/split))
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('GPU run requires a CUDA-enabled interpreter')
    report={'status':'passed','python':platform.python_version(),'executable':sys.executable,
        'prefix':sys.prefix,'project':str(ROOT),'working_directory':str(Path.cwd()),
        'dataset':str(DEFAULT_DATA),'packages':installed,'cuda_available':torch.cuda.is_available(),
        'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    write_json(ROOT/'reports'/'environment_check.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('\nEnvironment check PASSED.')
    if not torch.cuda.is_available():
        print('Current PyTorch uses CPU. Full training can be slow.')

if __name__=='__main__':
    main()
