import pickle,urllib.request,sys,os
import numpy as np
from pan_common import rollout

class Policy:
    def __init__(self,model):
        self.model=model
        self.bc_url=os.environ.get('COFFEE_BC_POLICY_URL')
        if self.bc_url:return
        if model=='xiaomi':
            from grpo_train_loop import TrainerClient
            self.client=TrainerClient('/checkpoint','127.0.0.1',10089,'robocasa365',.95)
            assert self.client.metrics()['policy_version']==0
        elif model=='pi05':
            sys.path.insert(0,'/results/pi-client-libs')
            sys.path.insert(0,'/openpi-client/src')
            from openpi_client import websocket_client_policy,image_tools
            self.client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',18205);self.image_tools=image_tools

    def infer(self,obs,images,states,instruction,seed,chunk):
        if self.bc_url:
            sample={'obs':{k:np.array(v,copy=True) for k,v in obs.items() if k.startswith(('state.','video.'))},'prompt':instruction}
            sample['obs']['xiaomi/state_history']=rollout.sample_history(states,4,2)
            for key,q in images.items():sample['obs']['xiaomi/'+key]=rollout.sample_history(q,4,2)
            req=urllib.request.Request(self.bc_url,data=pickle.dumps(dict(sample=sample,seed=seed*131+chunk),protocol=4),headers={'Content-Type':'application/octet-stream'})
            with urllib.request.urlopen(req,timeout=180) as response:reply=pickle.loads(response.read())
            actions=reply['actions'];self.version=reply['version']
        elif self.model=='xiaomi':
            actions,_=self.client.infer(rollout.sample_history(states,4,2),{key:rollout.sample_history(q,4,2) for key,q in images.items()},instruction,eta=0.,skill=None,seed=seed*131+1,chunk_index=chunk,expected_policy_version=0)
        elif self.model=='pi05':
            state=np.concatenate([obs[key] for key in ['state.end_effector_position_relative','state.end_effector_rotation_relative','state.base_position','state.base_rotation','state.gripper_qpos']])
            el={'observation/state':state,'prompt':instruction};it=self.image_tools
            for key,obskey in [('observation/image','video.robot0_agentview_left'),('observation/right_image','video.robot0_agentview_right'),('observation/wrist_image','video.robot0_eye_in_hand')]:el[key]=it.convert_to_uint8(it.resize_with_pad(np.ascontiguousarray(obs[obskey]),224,224))
            actions=self.client.infer(el)['actions']
        else:
            data=pickle.dumps(dict(obs=obs,instruction=instruction,seed=seed*131+chunk),protocol=4)
            req=urllib.request.Request('http://127.0.0.1:18217',data=data,headers={'Content-Type':'application/octet-stream'})
            with urllib.request.urlopen(req,timeout=180) as response:actions=pickle.loads(response.read())['actions']
        actions=np.asarray(actions,dtype=np.float32)
        assert actions.ndim==2 and actions.shape[1]==12 and len(actions)>=16 and np.isfinite(actions).all(),actions.shape
        return actions[:16]
