"""Meaningful development checks; these are not accuracy benchmarks."""
import ast
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import CLASSES,image_tensor
from inference import fuse
from metrics import classification_metrics
from models import GestureModel,DenseBlock,DenseLayer

def main():
    torch.set_num_threads(2)
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    checks=[]
    for name in ('plain_cnn','resnext50','resnext50_dense'):
        model=GestureModel(name,10)
        images=torch.randn(2,3,64,64); targets=torch.tensor([0,9])
        logits=model(images); loss=nn.functional.cross_entropy(logits,targets); loss.backward()
        assert logits.shape==(2,10) and torch.isfinite(loss)
        assert model.head[-1].weight.grad is not None
        groups=[m.groups for m in model.modules() if isinstance(m,nn.Conv2d) and m.groups>1]
        if name!='plain_cnn':
            assert len(groups)==16 and set(groups)=={32}
        if name=='resnext50_dense':
            blocks=[m for m in model.modules() if isinstance(m,DenseBlock)]
            assert len(blocks)==2
            for block in blocks:
                assert block(torch.randn(2,256,2,2)).shape==(2,384,2,2)
            assert all(m.layers[-1].weight.grad is not None for m in model.modules() if isinstance(m,DenseLayer))
        checks.append({'model':name,'parameters':sum(p.numel() for p in model.parameters()),
            'grouped_convolutions':len(groups),'forward_backward':'passed'})
        del model,logits,loss
    truth=np.array([0,1,9]); perfect=np.eye(10)[truth]
    metrics=classification_metrics(truth,perfect,CLASSES)
    assert metrics['accuracy']==1 and metrics['classification_mAP']==1
    rejected=classification_metrics(truth,perfect,CLASSES,[-1,1,9])
    assert rejected['rejected']==1 and np.isclose(rejected['accuracy'],2/3)
    cnn=np.array([.1,.6,.1,.05,.05,.02,.02,.02,.02,.02])
    assert fuse(cnn,'OK',True,.75)[1]==9
    assert fuse(cnn,'OK',True,.5)[1]==1
    assert fuse(cnn,'OK',False,.75)[1]==1
    assert fuse(cnn,None,True,.75)[1]==1
    tensor=image_tensor(np.zeros((80,100,3),np.uint8))
    assert tensor.shape==(3,64,64) and torch.isfinite(tensor).all()
    print(checks)
    print('Syntax, gradients, grouped convolutions, dense concatenation, rejections, fusion and preprocessing: PASSED')

if __name__=='__main__':
    main()
