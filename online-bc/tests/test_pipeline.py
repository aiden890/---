import json,tempfile,unittest,tarfile,hashlib,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coffee_metrics import score
from replay import Replay
from learner_service import unpack

class Contracts(unittest.TestCase):
    def row(self,step,**kw):return dict(step=step,mug_under_dispenser=True,mug_contact=False,mug_grasped=False,coffee_machine_on=False,**kw)
    def test_held_cup_is_not_placed(self):
        rows=[dict(self.row(i),mug_contact=True,mug_grasped=True) for i in range(1,15)];self.assertFalse(score(rows)['cup_placed'])
    def test_placement_requires_five_steps(self):
        self.assertFalse(score([self.row(i) for i in range(1,5)])['cup_placed']);self.assertEqual(score([self.row(i) for i in range(1,6)])['cup_placement_step'],5)
    def test_interruption_resets_streak(self):
        rows=[self.row(i) for i in range(1,8)];rows[3]['mug_contact']=True;self.assertFalse(score(rows)['cup_placed'])
    def test_earlier_button_not_success(self):
        rows=[dict(self.row(i),coffee_machine_on=True) for i in range(1,10)];self.assertFalse(score(rows)['button_after_placement'])
    def test_later_button_success(self):
        rows=[dict(self.row(i),coffee_machine_on=i>=7) for i in range(1,10)];self.assertEqual(score(rows)['button_press_step'],7)
    def test_cup_moved_before_button(self):
        rows=[dict(self.row(i),coffee_machine_on=i>=7,mug_under_dispenser=i<7) for i in range(1,10)];self.assertFalse(score(rows)['button_after_placement'])
    def test_terminal_placement_short_episode(self):
        self.assertTrue(score([self.row(1,official_success=True)])['cup_placed'])
    def test_replay_fifo_and_growth(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=Replay([root],max_episodes=2)
            for e,count in enumerate([1,4,2,7]):
                p=root/str(e)/'bc';p.mkdir(parents=True);rows=[]
                for i in range(count):
                    np.savez(p/f'o{i}.npz',x=np.array([e]));a=np.zeros((50,12),np.float32);a[0,0]=e;valid=np.arange(50)<1
                    np.savez(p/f'a{i}.npz',actions=a,valid=valid);rows.append(dict(observation=f'o{i}.npz',actions=f'a{i}.npz',skill='cup_placement',step=i*16))
                (p/'manifest.json').write_text(json.dumps(dict(samples=rows,prompt='cup',model='pi05',seed=e)))
                r.ingest()
            self.assertEqual(len(r.episodes),2);self.assertEqual(r.ingest(),9)
            for _ in range(50):
                s=r.sample('cup_placement');self.assertIn(s['seed'],[2,3]);self.assertEqual(s['actions'][0,0],s['obs']['x'][0]);self.assertEqual(s['valid'].sum(),1)
            with self.assertRaises(RuntimeError):r.sample('button_press')
    def test_replay_live_exclusion_and_restore(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);control=root/'controls.json';control.write_text(json.dumps(dict(revision=0,excluded={},paused=False)))
            for seed in [1,2]:
                folder=root/str(seed)/'bc';folder.mkdir(parents=True)
                np.savez(folder/'obs.npz',x=np.array([seed]));np.savez(folder/'act.npz',actions=np.zeros((50,12),np.float32),valid=np.arange(50)<1)
                (folder/'manifest.json').write_text(json.dumps(dict(model='pi05',seed=seed,prompt='cup',samples=[dict(skill='cup_placement',step=0,observation='obs.npz',actions='act.npz')])))
            replay=Replay([root],controls=control);replay.ingest()
            control.write_text(json.dumps(dict(revision=1,excluded={'pi05-seed1':{}},paused=False)))
            for _ in range(30):self.assertEqual(replay.sample('cup_placement')['episode_id'],'pi05-seed2')
            control.write_text(json.dumps(dict(revision=2,excluded={'pi05-seed1':{},'pi05-seed2':{}},paused=False)))
            with self.assertRaises(RuntimeError):replay.sample('cup_placement')
            control.write_text(json.dumps(dict(revision=3,excluded={},paused=False)))
            self.assertEqual({replay.sample('cup_placement')['episode_id'] for _ in range(30)},{'pi05-seed1','pi05-seed2'})
    def test_training_loop_records_usage_and_applies_mid_round_exclusion(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import train_online_bc
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);control=root/'controls.json';control.write_text(json.dumps(dict(revision=0,excluded={},paused=False)))
            for seed in [1,2]:
                folder=root/'data'/str(seed)/'bc';folder.mkdir(parents=True)
                np.savez(folder/'obs.npz',x=np.array([seed]));np.savez(folder/'act.npz',actions=np.zeros((50,12),np.float32),valid=np.arange(50)<1)
                (folder/'manifest.json').write_text(json.dumps(dict(model='pi05',seed=seed,prompt='cup',samples=[dict(skill='cup_placement',step=0,observation='obs.npz',actions='act.npz')])))
            calls=[]
            class Backend:
                def __init__(self,*args):pass
                def update(self,sample,seed):
                    calls.append(sample['episode_id'])
                    if len(calls)==1:control.write_text(json.dumps(dict(revision=1,excluded={sample['episode_id']:{}},paused=False)))
                    return dict(loss=.1,grad_norm=.1)
                def save(self,dest,step):(dest/'fake-test-adapter.json').write_text('{}')
            argv=['train_online_bc.py','--model','pi05','--checkpoint','test-only','--data',str(root/'data'),'--out',str(root/'out'),'--steps','3','--save-every','3','--skills','cup_placement','--controls',str(control)]
            with patch.object(sys,'argv',argv),patch.object(train_online_bc.importlib,'import_module',return_value=SimpleNamespace(Backend=Backend)):
                train_online_bc.main()
            self.assertEqual(len(calls),3);self.assertNotEqual(calls[0],calls[1]);self.assertEqual(calls[1],calls[2])
            usage=json.loads((root/'out/data-usage.json').read_text());self.assertEqual(usage['completed_updates'],3);self.assertEqual(usage['used'][calls[0]],1);self.assertEqual(usage['used'][calls[1]],2);self.assertEqual(usage['excluded'],[calls[0]])
            records=json.loads((root/'out/updates.json').read_text());self.assertEqual([x['control_revision'] for x in records],[0,1,1])
    def test_transfer_checksum_and_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=p/'bc-shards.tar.gz';f=p/'x';f.write_text('x')
            with tarfile.open(a,'w:gz') as tar:tar.add(f,arcname='ep/bc/x')
            (p/'manifest.json').write_text(json.dumps(dict(sha256=hashlib.sha256(a.read_bytes()).hexdigest())))
            unpack(p,p/'out');self.assertEqual((p/'out/ep/bc/x').read_text(),'x')
            with tarfile.open(a,'w:gz') as tar:tar.add(f,arcname='../escape')
            (p/'manifest.json').write_text(json.dumps(dict(sha256=hashlib.sha256(a.read_bytes()).hexdigest())))
            with self.assertRaises(ValueError):unpack(p,p/'out2')
            (p/'manifest.json').write_text(json.dumps(dict(sha256='bad')))
            with self.assertRaises(AssertionError):unpack(p,p/'out3')
if __name__=='__main__':unittest.main()
