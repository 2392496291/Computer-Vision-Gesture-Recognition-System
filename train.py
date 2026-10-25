"""Train one architecture and restore the best validation Macro-F1 checkpoint."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from common import (ROOT,CLASSES,GestureDataset,seed_everything,seed_worker,
                    manifest_digest,write_json)
from models import GestureModel,MODEL_NAMES
from metrics import classification_metrics

def choose_device(request):
    if request=='auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device(request)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install a matching GPU PyTorch build or use --device cpu.')
    return device

def run_epoch(model,loader,criterion,device,optimizer=None,scaler=None):
    training=optimizer is not None; model.train(training)
    total_loss=0.; targets=[]; scores=[]
    for images,labels in tqdm(loader,desc='train' if training else 'val',leave=False):
        images,labels=images.to(device),labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type,enabled=bool(scaler and scaler.is_enabled())):
                logits=model(images); loss=criterion(logits,labels).mean()
            if training:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        total_loss+=loss.item()*len(labels)
        targets.extend(labels.detach().cpu().tolist())
        scores.extend(logits.detach().float().softmax(1).cpu().tolist())
    return total_loss/len(targets),classification_metrics(targets,scores,CLASSES)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODEL_NAMES,required=True)
    parser.add_argument('--view',choices=['raw','mp'],default='raw')
    parser.add_argument('--manifest',type=Path,default=ROOT/'prepared'/'manifest.jsonl')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--weight-decay',type=float,default=0.)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--rgb',action='store_true',help='Default is OpenCV grayscale repeated into three channels')
    parser.add_argument('--pretrained',action='store_true',help='Optional ImageNet weights; requires download/cache')
    parser.add_argument('--class-weighted',action='store_true')
    parser.add_argument('--amp',action='store_true',help='Mixed precision on CUDA only')
    parser.add_argument('--allow-leakage',action='store_true',help='Diagnostic only; results are not valid independent-test evidence')
    parser.add_argument('--smoke',action='store_true',help='Permit a limited development manifest')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<2 or args.size<64:
        parser.error('epochs >= 1, batch-size >= 2, size >= 64 required')
    metadata=json.loads((args.manifest.parent/'metadata.json').read_text(encoding='utf-8'))
    audit=json.loads((args.manifest.parent/'audit.json').read_text(encoding='utf-8'))
    if metadata['smoke'] and not args.smoke:
        raise RuntimeError('Smoke manifest cannot be used for reported experiments. Prepare the full dataset.')
    if args.view=='mp' and not metadata['mediapipe']:
        raise RuntimeError('Prepare real MediaPipe crops first.')
    leakage=bool(audit.get('unresolved_cross_split_groups',len(audit['cross_split_duplicates'])) or audit.get('unresolved_label_conflicts',len(audit['conflicting_labels'])))
    if leakage and not args.allow_leakage:
        raise RuntimeError('Audit found split leakage/label conflicts. Inspect audit.json before training.')
    torch.set_num_threads(args.threads)
    device=choose_device(args.device); seed_everything(args.seed)
    gray=not args.rgb
    train=GestureDataset(args.manifest,'train',args.view,args.size,gray,True)
    val=GestureDataset(args.manifest,'val',args.view,args.size,gray)
    test=GestureDataset(args.manifest,'test',args.view,args.size,gray)
    generator=torch.Generator().manual_seed(args.seed)
    common={'num_workers':args.workers,'worker_init_fn':seed_worker,'pin_memory':device.type=='cuda'}
    # Backbone BatchNorm cannot train on a singleton with a 1x1 feature map.
    drop_last=len(train)%args.batch_size==1
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=generator,drop_last=drop_last,**common)
    val_loader=DataLoader(val,batch_size=args.batch_size,**common)
    test_loader=DataLoader(test,batch_size=args.batch_size,**common)
    model=GestureModel(args.model,len(CLASSES),args.pretrained).to(device)
    weights=None
    if args.class_weighted:
        counts=np.bincount([r['target'] for r in train.records],minlength=len(CLASSES))
        weights=torch.tensor(len(train)/(len(CLASSES)*counts),dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights,reduction='none')
    eval_criterion=nn.CrossEntropyLoss(reduction='none')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=3)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp and device.type=='cuda')
    out=args.output or ROOT/'runs'/('%s_%s_seed%d'%(args.model,args.view,args.seed))
    out=out.resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():
        raise RuntimeError('Run already exists. Choose a new --output to preserve previous evidence.')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(classes=CLASSES,gray=gray,device_used=str(device),manifest_sha256=manifest_digest(args.manifest),
        parameters=sum(p.numel() for p in model.parameters()),leakage_detected=leakage,
        torch_version=str(torch.__version__),selection_metric='validation f1_macro',
        pretrained_backbone=bool(args.pretrained),dropped_singleton_per_epoch=drop_last,
        score_definition='softmax CNN logits; classification mAP, no IoU threshold')
    write_json(out/'config.json',config)
    print('Device:',device,'parameters:',config['parameters'],'train/val/test:',len(train),len(val),len(test),flush=True)
    history=[]; best=-1.; started=time.perf_counter()
    for epoch in range(args.epochs):
        loss,tm=run_epoch(model,train_loader,criterion,device,optimizer,scaler)
        vl,vm=run_epoch(model,val_loader,eval_criterion,device)
        scheduler.step(vm['f1_macro'])
        row={'epoch':epoch+1,'train_loss':loss,'val_loss':vl,'train_accuracy':tm['accuracy'],
            'val_accuracy':vm['accuracy'],'val_f1_macro':vm['f1_macro'],'lr':optimizer.param_groups[0]['lr']}
        history.append(row)
        print(json.dumps(row),flush=True)
        write_json(out/'history.json',history)
        if vm['f1_macro']>best:
            best=vm['f1_macro']
            checkpoint={'state_dict':model.state_dict(),'config':config,'epoch':epoch+1,'val_metrics':vm}
            torch.save(checkpoint,out/'best.pt')
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    tl,metrics=run_epoch(model,test_loader,eval_criterion,device)
    metrics.update(test_loss=tl,best_epoch=checkpoint['epoch'],elapsed_training_and_test_seconds=time.perf_counter()-started,
        smoke=metadata['smoke'],leakage_detected=leakage,checkpoint='best.pt')
    write_json(out/'test_metrics.json',metrics)
    with (out/'history.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    print('Saved:',out,flush=True)
    print(json.dumps({k:metrics[k] for k in ('accuracy','precision_macro','recall_macro','f1_macro','classification_mAP')},indent=2))

if __name__=='__main__':
    main()
