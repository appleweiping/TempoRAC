import pathlib,shutil,json,hashlib,difflib,re
B=pathlib.Path('/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites')
S=pathlib.Path('/root/autodl-tmp/TempoRAC/external-baselines-20261111/code/DeTRC')
C=B/'candidate'
if not C.exists():shutil.copytree(S,C,ignore=shutil.ignore_patterns('.git','__pycache__'))
def change(rel,fn):
 p=C/rel;s=p.read_text();n=fn(s)
 if n!=s:p.write_text(n)
for p in C.rglob('*.py'):
 change(p.relative_to(C),lambda s:s.replace('DetRC.', 'DeTRC.').replace('np.bool)', 'bool)'))
change('DeTRC/repcount_dataset.py',lambda s:s.replace('labels = rows[6:]','labels = rows.iloc[6:]').replace('labels[2 * i + 1]','labels.iloc[2 * i + 1]').replace('labels[2 * i]','labels.iloc[2 * i]').replace('gt = np.array(gt)','gt = np.array(gt).reshape(-1, 3)').replace('np.array(gt)\n','np.array(gt).reshape(-1, 3)\n'))
old='''                pos_embeds = self.pos_embed.unsqueeze(0).repeat(
                    input_feature.shape[0], -1, -1
                )'''
new='''                # Fresh TSN default retraining only: train always passes None.
                # Keep the author transformer permute; all-zero table makes it inert.
                if torch.count_nonzero(self.pos_embed.weight).item() != 0:
                    raise RuntimeError("TSN default route requires an exactly zero position table")
                pos_embeds = self.pos_embed.weight.unsqueeze(0).repeat(
                    input_feature.shape[0], 1, 1
                )'''
change('DeTRC/model/DetTRC.py',lambda s:s.replace(old,new))
for p in [C/'DeTRC/model/grid_sample1d/grid_sample1d_cuda.cpp',C/'DeTRC/model/grid_sample1d/grid_sample1d_cuda_kernel.cu',C/'DeTRC/model/roi_align/src/roi_align_kernel.cu']:
 change(p.relative_to(C),lambda s:s.replace('.type().is_cuda()', '.is_cuda()').replace('AT_DISPATCH_FLOATING_TYPES(input.type(),','AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),').replace('AT_DISPATCH_FLOATING_TYPES(grad.type(),','AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(),').replace('.data<scalar_t>()','.data_ptr<scalar_t>()'))
change('DeTRC/model/roi_align/roi_align.py',lambda s:s.replace('return grad_input, None, None, None, None','return grad_input, None, None, None'))
for rel in ['tools/train.py','tools/test.py']:
 change(rel,lambda s:s.replace('import mmcv','import DeTRC.model.DetTRC\nimport DeTRC.repcount_dataset\nimport DeTRC.durable_state\n\nimport mmcv',1))
# Real hook is supplied separately; no runtime stub.
shutil.copy2(B/'durable_state.py',C/'DeTRC/durable_state.py')
change('DeTRC/config/repcount_tsn_feature_enc_contrastive.py',lambda s:s+'\n# Epoch-boundary recovery; native model/optimizer/selection remain unchanged.\ncustom_hooks = [dict(type="DeTRCDurableStateHook", priority="LOWEST")]\n')
rows=[];patch=[]
for p in sorted(C.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts:continue
 rel=p.relative_to(C).as_posix();raw=p.read_bytes();orig=S/rel
 if not orig.exists() or raw!=orig.read_bytes():
  rows.append({'path':rel,'sha256':hashlib.sha256(raw).hexdigest(),'before_sha256':hashlib.sha256(orig.read_bytes()).hexdigest() if orig.exists() else None})
  if p.suffix in ['.py','.cu','.cpp']:
   a=orig.read_text().splitlines(True) if orig.exists() else []
   patch.extend(difflib.unified_diff(a,p.read_text().splitlines(True),fromfile='author/'+rel,tofile='candidate/'+rel))
(B/'CANDIDATE_DIFF.patch').write_text(''.join(patch))
(B/'CHANGED_FILES.json').write_text(json.dumps(rows,indent=2))
print('Changed files',len(rows))
