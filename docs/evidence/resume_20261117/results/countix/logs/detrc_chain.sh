#!/bin/bash
# DeTRC (official code, RepCount-trained) on Countix test: CPU frames now; GPU stages on GPU0 after the RepNet/TransRAC chain.
Q=/root/autodl-tmp/TempoRAC/resume20261117; cd $Q
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/autodl-tmp/TempoRAC/env/bin/python -u detrc_countix.py frames --workers 12 || exit 1
while pgrep -f "bash countix_chain.sh" > /dev/null; do sleep 60; done
free0() { nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits; }
until [ $(free0) -ge 5000 ]; do sleep 120; done
/root/autodl-tmp/TempoRAC/env/bin/python -u detrc_countix.py extract --gpu 0 || exit 2
/root/autodl-tmp/TempoRAC/env/bin/python -u detrc_countix.py predict --gpu 0 || exit 3
/root/autodl-tmp/TempoRAC/env/bin/python -u detrc_countix.py score || exit 4
echo DETRC_CHAIN_DONE
