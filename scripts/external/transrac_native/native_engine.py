"""Four DataParallel-native shards on one device; no model substitutions."""
from __future__ import annotations
import copy
import random
import numpy as np
import torch
from torch import nn
from contextlib import nullcontext

def bn_state(model):
    return {name: value.detach().clone()
            for prefix, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            for suffix, value in module.named_buffers(recurse=False)
            if value is not None
            for name in [(prefix + '.' if prefix else '') + suffix]}

def restore_bn(model, state):
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, value in state.items():
            buffers[name].copy_(value)

def rng_get(device):
    return torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else torch.get_rng_state().clone()

def rng_set(device, state):
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state.cpu(), device)
    else:
        torch.set_rng_state(state.cpu())

def virtual_train_step(model, optimizer, scaler, inputs, targets, streams):
    """One B32/B20 optimizer opportunity, preserving primary-replica BN."""
    device = next(model.parameters()).device
    batch = len(inputs)
    assert batch in (32, 20) and len(streams) == 4
    model.train()
    optimizer.zero_grad()
    initial_bn = bn_state(model)
    primary_bn = None
    loss_value = 0.0
    predictions = []
    for shard, (x, y) in enumerate(zip(torch.chunk(inputs, 4), torch.chunk(targets, 4))):
        restore_bn(model, initial_bn)
        rng_set(device, streams[shard])
        # CUDA AMP is exactly the native training loop's outer context. The
        # unchanged TransferModel also owns its internal autocast in every mode.
        with torch.cuda.amp.autocast() if device.type == 'cuda' else nullcontext():
            output, _ = model(x.to(device, non_blocking=False))
            density = y.to(device, non_blocking=False)
            loss = nn.functional.mse_loss(output, density) * (len(x) / batch)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f'nonfinite density MSE at virtual shard {shard}')
        streams[shard] = rng_get(device)
        if shard == 0:
            primary_bn = bn_state(model)
        loss_value += float(loss.detach())
        predictions.append(output.detach().sum(dim=1).float().cpu())
        scaler.scale(loss).backward()  # graph freed before any BN buffer restore
        del output, density, loss
    restore_bn(model, primary_bn)
    scaler.unscale_(optimizer)
    finite = all(bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None)
    called = [False]
    handle = optimizer.register_step_post_hook(lambda *_: called.__setitem__(0, True))
    old_scale = float(scaler.get_scale())
    try:
        scaler.step(optimizer)
        scaler.update()
    finally:
        handle.remove()
    if called[0] != finite:
        raise RuntimeError('GradScaler update/finite-gradient inconsistency')
    return {'loss':loss_value, 'gradient_finite':finite, 'optimizer_updated':called[0],
            'overflow_skip':not called[0], 'scale_before':old_scale,
            'scale_after':float(scaler.get_scale()), 'prediction':torch.cat(predictions)}

def capture_rng(train_generator, valid_generator, streams):
    return {'python':random.getstate(), 'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'virtual_streams':[s.clone() for s in streams],
            'train_generator':train_generator.get_state(), 'valid_generator':valid_generator.get_state()}

def restore_rng(state, train_generator, valid_generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    train_generator.set_state(state['train_generator'].cpu())
    valid_generator.set_state(state['valid_generator'].cpu())
    return [s.cpu().clone() for s in state['virtual_streams']]

def optimizer_and_scheduler(model):
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),
                                 lr=8e-6, betas=(.9,.999), eps=1e-8, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0,40,80,120,160], gamma=.8)
    return optimizer, scheduler

class Primitive(nn.Module):
    """CPU-only arithmetic fixture; never a research model or dataset result."""
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.fc = nn.Linear(12, 64)
    def forward(self, x):
        x = sum(torch.tanh(self.bn1(x + shift)) for shift in (0., .1, -.2))
        x = self.bn2(x.mean(dim=2))
        return self.fc(x.flatten(1)), None

def primitive_checks():
    torch.set_num_threads(2)
    errors = []
    for batch in (32,20):
        torch.manual_seed(17)
        serial = Primitive().double().train()
        reference = copy.deepcopy(serial)
        replicas = [copy.deepcopy(reference) for _ in range(4)]
        x = torch.randn(batch,3,2,2,2,dtype=torch.float64)
        y = torch.randn(batch,64,dtype=torch.float64)
        opt, _ = optimizer_and_scheduler(serial)
        refopt, _ = optimizer_and_scheduler(reference)
        loss = 0.
        for model, xx, yy in zip(replicas, torch.chunk(x,4), torch.chunk(y,4)):
            out,_ = model(xx)
            part = nn.functional.mse_loss(out,yy)*(len(xx)/batch)
            loss += float(part.detach())
            part.backward()
        for name,p in reference.named_parameters():
            p.grad = sum(dict(m.named_parameters())[name].grad for m in replicas)
        restore_bn(reference,bn_state(replicas[0]))
        expected_grads = {k:p.grad.clone() for k,p in reference.named_parameters()}
        scaler = torch.amp.GradScaler('cuda',enabled=False)
        result = virtual_train_step(serial,opt,scaler,x,y,[torch.get_rng_state().clone() for _ in range(4)])
        grad_error = max(float((p.grad-expected_grads[k]).abs().max()) for k,p in serial.named_parameters())
        refopt.step()
        param_error = max(float((p-dict(reference.named_parameters())[k]).abs().max()) for k,p in serial.named_parameters())
        buffer_error = max(float((b-bn_state(reference)[k]).abs().max()) for k,b in bn_state(serial).items())
        entry={'batch':batch,'shards':[len(c) for c in torch.chunk(x,4)],
               'loss_error':abs(result['loss']-loss),'gradient_max_abs':grad_error,
               'adam_parameter_max_abs':param_error,'primary_bn_max_abs':buffer_error}
        assert max(entry[k] for k in ('loss_error','gradient_max_abs','adam_parameter_max_abs','primary_bn_max_abs')) < 2e-12, entry
        errors.append(entry)
    # Independent persistent RNG streams: interleave and checkpoint/restore.
    device=torch.device('cpu')
    original=torch.get_rng_state()
    g=torch.Generator().manual_seed(1)
    states=[g.get_state().clone() for _ in range(4)]
    expected=[]
    for i in range(4):
        rng_set(device,states[i]); first=torch.rand(7); states[i]=rng_get(device)
        expected.append(first)
    saved=[s.clone() for s in states]
    second=[]
    for i in range(4):
        rng_set(device,states[i]); second.append(torch.rand(11)); states[i]=rng_get(device)
    for i in range(4):
        rng_set(device,saved[i]); assert torch.equal(torch.rand(11),second[i])
    torch.set_rng_state(original)
    assert all(torch.equal(expected[0],v) for v in expected)
    # All 200 native learning rates, including constructor milestone0.
    m=Primitive().double(); opt,sched=optimizer_and_scheduler(m)
    lrs=[]
    for epoch in range(200):
        lrs.append(opt.param_groups[0]['lr'])
        opt.zero_grad(); m.fc.weight.grad=torch.zeros_like(m.fc.weight); opt.step(); sched.step()
    expected_lr=[8e-6*.8**(1+e//40) for e in range(200)]
    assert max(abs(a-b) for a,b in zip(lrs,expected_lr))<1e-19
    return {'status':'PASS','scope':'server CPU FP64 resource-adapter primitives only; no dataset performance/GPU parity',
            'arithmetic':errors,'rng_stream_restore':'PASS','lr_sequence':lrs}
