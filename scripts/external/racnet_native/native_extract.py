"""Official full sequential single-frame Swin path; no labels or 64-RGB cache."""
import importlib.util
import logging
import os
from pathlib import Path
import sys
import time
import types
import uuid
import numpy as np
import torch
from native_common import ROOT, contract, gpu_gate, identity, load, savejson, sha, syncdir, storage_guard, timing_summary


def backbone(work):
    """Import original mathematical source, isolating only registry/log imports."""
    c = contract()['quva_extraction']
    source = Path(c['backbone_source'])
    assert sha(source) == c['backbone_source_sha256']
    assert sha(c['base_config']) == c['base_config_sha256']
    class Registry:
        def register_module(self):
            return lambda cls: cls
    names = ('mmaction', 'mmaction.models', 'mmaction.models.backbones', 'mmaction.models.builder', 'mmaction.utils')
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
        sys.modules['mmaction.models.builder'].BACKBONES = Registry()
        sys.modules['mmaction.utils'].get_root_logger = lambda *a, **kw: logging.getLogger('racnet-official-swin')
        spec = importlib.util.spec_from_file_location('mmaction.models.backbones.swin_transformer', source)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    cfg = dict(c['backbone'])
    cfg.pop('type')
    for key in ('patch_size', 'window_size'):
        cfg[key] = tuple(cfg[key])
    model = mod.SwinTransformer3D(**cfg)
    weights = c['backbone_weights']
    # The existing source acquisition receipt already certifies the full weight SHA.
    ready = load(ROOT / 'baselines/transrac-native-assets20261112/BACKBONE_READY_RECEIPT.json')
    assert ready['status'] == 'complete_exact_expected_sha' and ready['sha256'] == c['backbone_weight_sha256']
    assert ready['path'] == weights and ready['bytes'] == os.stat(weights).st_size == c['backbone_weight_bytes']
    # Exact checkpoint keys/shapes, independently checked below; receipt format is retained in CPU report.
    payload = torch.load(weights, map_location='cpu', weights_only=False)
    original = payload['state_dict']
    state = {k[len('backbone.'):]: v for k, v in original.items() if k.startswith('backbone.')}
    omitted = [k for k in original if not k.startswith('backbone.')]
    assert state and all(k.startswith('cls_head.') for k in omitted), ('unexpected non-backbone checkpoint keys', omitted)
    expected = model.state_dict()
    assert set(state) == set(expected), {'missing': sorted(set(expected)-set(state)), 'extra': sorted(set(state)-set(expected))}
    assert all(state[k].shape == expected[k].shape and state[k].dtype == expected[k].dtype for k in expected)
    model.load_state_dict(state, strict=True)
    return model, {'source_sha256': sha(source), 'base_config_sha256': sha(c['base_config']),
                   'weights_sha256': c['backbone_weight_sha256'], 'weight_bytes': os.stat(weights).st_size,
                   'strict_keys': len(state), 'omitted_classifier_keys': omitted,
                   'existing_weight_receipt': ready, 'no_forward': True}


def extraction_binding(work, strict):
    return {'candidate_identity': identity(work), 'backbone': {k: strict[k] for k in ('source_sha256', 'base_config_sha256', 'weights_sha256')},
            'inputs_sha256': sha(Path(work) / 'inputs/quva_raw_inputs.json'),
            'mode': 'official-single-frame-transpose0321-fp32-minus127.5-div127.5'}


def extract(work, receipt):
    if (Path(work) / 'RUN_CONTRACT_MISSING5_V2.json').exists():
        from native_recovery import extract_required105
        return extract_required105(work, receipt)
    invocation_start = time.monotonic()
    work = Path(work)
    gpu_gate(work, receipt, 'extract-quva')
    import cv2
    from native_common import runtime_flags
    runtime_flags()
    model, strict = backbone(work)
    model.eval().to('cuda')
    binding = extraction_binding(work, strict)
    if (work / 'QUVA_BINDING.json').exists():
        assert load(work / 'QUVA_BINDING.json') == binding
    else:
        savejson(work / 'QUVA_BINDING.json', binding)
    rows = load(work / 'inputs/quva_raw_inputs.json')
    assert len(rows) == 100 and sum(r['input_T'] for r in rows) == 48030
    done, reused_keys = [], set()
    for r in rows:
        storage_guard(work)
        dest = Path(r['path'])
        marker = Path(str(dest) + '.json')
        if marker.exists():
            prev = load(marker)
            assert prev['binding'] == binding and prev['raw_video_sha256'] == r['raw_video_sha256']
            assert prev['T'] == r['input_T'] and sha(dest) == prev['sha256']
            done.append(prev)
            reused_keys.add(r['key'])
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.parent / ('.' + dest.name + '.' + uuid.uuid4().hex)
        features = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(r['input_T'], 37632))
        cap = cv2.VideoCapture(r['raw_video'])
        assert cap.isOpened(), r['key']
        n, start = 0, time.monotonic()
        try:
            with torch.no_grad():
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    assert n < r['input_T'], ('extra decoded frame', r['key'])
                    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_LINEAR)
                    # Source frames.transpose(0,3,2,1): preserve its H/W exchange.
                    x = torch.FloatTensor(np.ascontiguousarray(rgb.transpose(2, 1, 0)))
                    x -= 127.5
                    x /= 127.5
                    out = model(x[None, :, None].to('cuda'))
                    assert tuple(out.shape) == (1, 768, 1, 7, 7)
                    flat = out[0, :, 0].contiguous().view(37632).cpu().numpy()
                    assert np.isfinite(flat).all(), ('nonfinite official feature', r['key'], n)
                    features[n] = flat
                    n += 1
            assert n == r['input_T'], ('decode count mismatch', r['key'], n)
            features.flush()
            del features
            with open(temp, 'rb') as f:
                os.fsync(f.fileno())
            os.replace(temp, dest)
            syncdir(dest.parent)
            row = {'key': r['key'], 'T': n, 'shape': [n, 37632], 'dtype': 'float32', 'finite': True,
                   'sha256': sha(dest), 'bytes': dest.stat().st_size, 'seconds': time.monotonic()-start,
                   'raw_video_sha256': r['raw_video_sha256'], 'binding': binding}
            savejson(marker, row)
            done.append(row)
            savejson(work / 'QUVA_PROGRESS.json', {'videos': len(done), 'frames': sum(x['T'] for x in done), 'binding': binding})
        finally:
            cap.release()
            # A failed partial stays as evidence; never promoted to a reusable output.
    assert len(done) == 100 and sum(r['T'] for r in done) == 48030
    timing = timing_summary(done, time.monotonic()-invocation_start, reused_keys, total_field='seconds')
    savejson(work / 'QUVA_COMPLETE.json', {'status': 'PASS', 'binding': binding, 'videos': done,
             'frames': 48030, 'timing': timing})
