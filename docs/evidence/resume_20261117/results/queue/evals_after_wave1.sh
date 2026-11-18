#!/bin/bash
# After the six first-wave trainings (wo_depth_s*, wo_balance_s*) exit, run their missing evaluations with the same queue.
Q=/root/autodl-tmp/TempoRAC/resume20261117
while pgrep -f "temporac_cvpr.(train|eval_saved) .*ablation20261112/wo_(depth|balance)_s[012] " > /dev/null; do sleep 60; done
cd $Q && /root/autodl-tmp/TempoRAC/env/bin/python -u ks_queue.py --only wo_depth_s0,wo_depth_s1,wo_depth_s2,wo_balance_s0,wo_balance_s1,wo_balance_s2 --per-gpu 3 --min-free 6000
