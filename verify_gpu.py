"""Validate CUDA kernels and real model backward passes without starting full training."""
import json
import time
import torch
from common import ROOT,write_json
from models import GestureModel

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this interpreter')
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    a=torch.randn(256,256,device=device)
    b=a@a; torch.cuda.synchronize()
    if not torch.isfinite(b).all():
        raise RuntimeError('Non-finite matrix product')
    result={'status':'passed','torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,
        'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
        'compiled_architectures':torch.cuda.get_arch_list(),'matrix_product':'passed','models':[]}
    for name in ('plain_cnn','resnext50','resnext50_dense'):
        torch.cuda.reset_peak_memory_stats()
        model=GestureModel(name,10).to(device).train()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        images=torch.randn(2,3,64,64,device=device)
        labels=torch.tensor([0,9],device=device)
        started=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs=model(images); loss=torch.nn.functional.cross_entropy(outputs,labels)
        loss.backward(); optimizer.step(); torch.cuda.synchronize()
        assert outputs.shape==(2,10) and torch.isfinite(loss)
        result['models'].append({'model':name,'forward_backward_adam':'passed',
            'batch_size':2,'seconds_including_first_kernel_initialisation':time.perf_counter()-started,
            'peak_allocated_MB':torch.cuda.max_memory_allocated()/1024**2})
        del model,optimizer,images,labels,outputs,loss
        torch.cuda.empty_cache()
    write_json(ROOT/'reports'/'gpu_verification.json',result)
    print(json.dumps(result,indent=2))
    print('CUDA matrix multiply and all three model backward/Adam checks PASSED. No full training started.')

if __name__=='__main__':
    main()
