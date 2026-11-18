#!/bin/bash
# GPU0 of the first server (shared with another project): official RepNet then official-code TransRAC on Countix test,
# one after the other, each started only when GPU0 has enough free memory.
R=/root/autodl-tmp/TempoRAC; Q=$R/resume20261117
free0() { nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits; }
until [ $(free0) -ge 5000 ]; do sleep 120; done
cd $Q && $R/envs/repnet/bin/python repnet_countix.py --gpu 0 --mem 3072 --min-free 5000 --out $Q/repnet_countix > $Q/repnet_countix.out 2>&1
echo "REPNET_COUNTIX_EXIT $?" >> $Q/repnet_countix.out
until [ $(free0) -ge 4000 ]; do sleep 120; done
cd $Q && $R/envs/transrac-native310/bin/python -u transrac_countix.py --gpu 0 --out $Q/transrac_countix > $Q/transrac_countix.out 2>&1
echo "TRANSRAC_COUNTIX_EXIT $?" >> $Q/transrac_countix.out
