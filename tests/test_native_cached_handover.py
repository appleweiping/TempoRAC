"""Artificial process identities only; never signal a real process."""
import importlib.util
from pathlib import Path
import signal
import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace
import uuid
import os

module_path=Path(__file__).with_name('cached_sequence.py')
if not module_path.exists():module_path=Path(__file__).resolve().parents[1]/'scripts/run_temporac_cached_campaign.py'
spec=importlib.util.spec_from_file_location('handover',module_path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class HandoverTests(unittest.TestCase):
    def setup_case(self):
        root=Path(__file__).parent/'test-artifacts'/uuid.uuid4().hex
        root.mkdir(parents=True)
        cfg={'run_root':str(root),'old_worker':{'output_root':str(root/'old'),
             'processes':[{'pid':10},{'pid':11}]}}
        return root,cfg

    def test_existing_gpu_evidence_forbids_stop(self):
        root,cfg=self.setup_case()
        with patch.object(m,'old_has_training',return_value='steps.jsonl'),patch.object(m,'send') as send:
            with self.assertRaises(ValueError):m.handover(cfg,root/'config.json')
            send.assert_not_called()

    def test_gpu_race_after_stop_rolls_back_without_terminate(self):
        root,cfg=self.setup_case();states={10:'R',11:'R'};calls=[]
        linux_signal=SimpleNamespace(SIGSTOP=19,SIGCONT=18,SIGTERM=15)
        def same(row):return dict(row,state=states[row['pid']])
        def send(row,sig):
            calls.append((row['pid'],sig));states[row['pid']]='T' if sig==linux_signal.SIGSTOP else 'R'
        with patch.object(m,'old_has_training',side_effect=[None,'old worker GPU race']),\
             patch.object(m,'same',side_effect=same),patch.object(m,'send',side_effect=send),\
             patch.object(m,'identity',return_value={'pid':99}),\
             patch.object(m.subprocess,'Popen',return_value=Mock(poll=lambda:None,pid=999)),\
             patch.object(m,'signal',linux_signal),\
             patch.object(m.time,'sleep',return_value=None):
            with self.assertRaises(ValueError):m.handover(cfg,root/'config.json')
        self.assertTrue(all(sig!=linux_signal.SIGTERM for _,sig in calls))
        self.assertEqual(states,{10:'R',11:'R'})
        self.assertEqual(m.load(root/'handover_state.json')['phase'],'handover_failed')

    def test_pid_identity_mismatch_prevents_signal(self):
        expected=dict(pid=10,start_ticks=1,uid=123,cwd='/old',command='python old')
        current=dict(expected,start_ticks=2,state='R')
        with patch.object(m,'identity',return_value=current),patch.object(m.os,'kill') as kill:
            with self.assertRaises(ValueError):m.send(expected,signal.SIGTERM)
            kill.assert_not_called()

    def test_dev_ready_is_late_and_records_exact_manifest(self):
        root,cfg=self.setup_case();cfg['dev_ready_path']=str(root/'late-dev-ready.json')
        # Missing dev data is allowed while preparing the sequence, but blocks prediction.
        with patch.object(m,'BASE',root.parent):
            with self.assertRaises(FileNotFoundError):m.dev_ready(cfg,'canonical')
            manifest=Path(cfg['dev_ready_path'])
            m.save(manifest,dict(schema='temporac.prepared-cache.manifest.v1',status='complete',phase='dev',natural={'records':[{}]*486}))
            ref=dict(path=str(manifest),sha256=m.sha(manifest))
            self.assertEqual(m.dev_ready(cfg,'canonical'),(manifest.resolve(),ref['sha256']))
            self.assertEqual(m.load(root/'logs/dev-ready-canonical.json')['dev_cache_manifest'],ref)

    def test_incomplete_or_changed_dev_never_predicts(self):
        root,cfg=self.setup_case();cfg['dev_ready_path']=str(root/'dev-ready.json')
        manifest=Path(cfg['dev_ready_path'])
        with patch.object(m,'BASE',root.parent):
            m.save(manifest,dict(schema='temporac.prepared-cache.manifest.v1',status='complete',phase='dev',natural={'records':[{}]*485}))
            ref=dict(path=str(manifest),sha256=m.sha(manifest))
            with self.assertRaises(ValueError):m.dev_ready(cfg,'canonical')
            m.save(manifest,{})
            with self.assertRaises(ValueError):m.dev_ready(cfg,'canonical')

    @unittest.skipUnless(os.name=='posix','actual Linux flock semantics')
    def test_project_lock_blocks_duplicate_and_releases_on_exit(self):
        root,cfg=self.setup_case()
        with patch.object(m,'CAMPAIGN_LOCK',root/'fixed-project.lock'):
            with m.campaign_lock(cfg):
                with self.assertRaisesRegex(RuntimeError,'another NTR1 actor'):
                    with m.campaign_lock(cfg):self.fail('duplicate actor acquired lock')
            _,second=self.setup_case()
            with m.campaign_lock(second):pass

    @unittest.skipUnless(os.name=='posix','Linux affinity API')
    def test_cpu_affinity_contains_cache_and_caps_combined_set(self):
        root,cfg=self.setup_case();receipt=root/'cache-affinity.json'
        m.save(receipt,dict(schema='temporac.ntr1.cpu-affinity.v1',allowed_cpus=[2,3,4,5,6],
            campaign_cpus=[2,3,4,5],cache_cpus=[2,3,4]))
        cfg['cache_affinity_receipt']=dict(path=str(receipt),sha256=m.sha(receipt))
        with patch.object(m,'BASE',root.parent),patch.object(m.os,'sched_getaffinity',side_effect=[{2,3,4,5,6},{2,3,4,5}]),\
             patch.object(m.os,'sched_setaffinity') as apply:
            m.pin_cpu(cfg);apply.assert_called_once_with(0,[2,3,4,5])
        self.assertEqual(m.load(root/'cpu_affinity.json')['cache_cpus'],[2,3,4])


if __name__=='__main__':unittest.main(verbosity=2)
