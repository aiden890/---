import argparse,collections,hashlib,json,random,time
from pathlib import Path
import numpy as np,imageio.v2 as imageio
from pan_common import rollout,queues,append,overlay,write,state_hash
from policy_adapter import Policy
from bc_capture import Capture
from coffee_metrics import score
import gymnasium as gym,robocasa
from robocasa.utils import object_utils as OU
from robocasa.utils.env_utils import convert_action
from robocasa.utils.dataset_registry_utils import get_task_horizon

def predicates(k,rest_pos):
    obj=k.objects['obj'];g=k.robots[0].gripper['right'];pos=np.array(k.sim.data.body_xpos[k.obj_body_id[obj.name]])
    contact=bool(k.check_contact(g,obj));lift=float(pos[2]-rest_pos[2]);motion=float(np.linalg.norm(pos-rest_pos))
    return dict(mug_grasped=bool(k._check_grasp(g,obj)),mug_contact=contact,mug_lift=lift,mug_lifted=bool(contact and lift>.03),
        mug_moved_while_held=bool(contact and motion>.1),mug_motion=motion,
        mug_under_dispenser=bool(k.coffee_machine.check_receptacle_placement_for_pouring(k,'obj')),
        coffee_machine_on=bool(k.coffee_machine._turned_on),gripper_mug_far=bool(OU.gripper_obj_far(k)),
        gripper_button_far=bool(k.coffee_machine.gripper_button_far(k)),official_success=bool(k._check_success()),mug_world_position=pos.tolist())

ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['xiaomi','pi05','groot'],required=True);ap.add_argument('--out',required=True);ap.add_argument('--seeds',default='981001,981002,981003');ap.add_argument('--skill',choices=['cup_placement']);args=ap.parse_args()
root=Path(args.out);root.mkdir(parents=True,exist_ok=True);policy=Policy(args.model);horizon=int(get_task_horizon('PrepareCoffee'))
for seed in map(int,args.seeds.split(',')):
    out=root/f'{args.model}-seed{seed}'
    if (out/'result.json').exists():continue
    if out.exists():out.rename(root/(out.name+'-incomplete-'+str(int(time.time()))))
    out.mkdir(exist_ok=False);np.random.seed(seed);random.seed(seed)
    env=gym.make('robocasa/PrepareCoffee',split='pretrain',seed=seed)
    try:
        obs,_=rollout.reset_env(env,seed);k=env.unwrapped.env;instruction=str(obs['annotation.human.task_description'])
        rest_pos=np.array(k.sim.data.body_xpos[k.obj_body_id['obj']],copy=True);initial_hash=state_hash(k);xml_hash=hashlib.sha256(k.sim.model.get_xml().encode()).hexdigest()
        write(out/'initial.json',dict(seed=seed,initial_state_hash=initial_hash,scene_xml_sha256=xml_hash,instruction=instruction,ep_meta=k.get_ep_meta(),mug_world_position=rest_pos.tolist()))
        imageio.imwrite(out/'initial.png',rollout.make_video_frame(obs))
        images,states=queues(obs);plan=collections.deque();calls=0;trace=[];actions=[];times=[];milestones={}
        writer=imageio.get_writer(out/'video.partial.mp4',fps=20);writer.append_data(overlay(obs,f'{args.model} | PrepareCoffee | seed {seed} | step 0'))
        started=time.time();success=False;capture=Capture(out);cup_skill_start=None;placement_streak=0
        cup_instruction='Place the mug you are holding upright on the coffee machine tray directly under the dispenser, then release the mug.'
        try:
            for step in range(1,horizon+1):
                if not plan:
                    if args.skill and cup_skill_start is None and trace and trace[-1]['mug_grasped'] and trace[-1]['mug_motion']>.1:
                        cup_skill_start=step-1
                    policy_instruction=cup_instruction if args.skill and cup_skill_start is not None else instruction
                    capture.observation(step-1,obs,images,states,rollout)
                    t=time.time();plan.extend(policy.infer(obs,images,states,policy_instruction,seed,calls));times.append(time.time()-t);calls+=1
                action=np.asarray(plan.popleft(),np.float32);actions.append(action.copy());obs,_,done,trunc,info=env.step(convert_action(action));append(obs,images,states)
                p=predicates(k,rest_pos);success=p['official_success'];assert success==bool(info['success']);trace.append(dict(step=step,**p))
                for key in ['mug_grasped','mug_lifted','mug_moved_while_held','mug_under_dispenser','coffee_machine_on','official_success']:
                    if p[key] and key not in milestones:milestones[key]=step
                if step%2==0 or success:writer.append_data(overlay(obs,f'{args.model} | step {step} | lifted {int(p["mug_lifted"])} | dispenser {int(p["mug_under_dispenser"])} | on {int(p["coffee_machine_on"])}'))
                if step%160==0:print(json.dumps(dict(event='PROGRESS',model=args.model,seed=seed,step=step,milestones=milestones,**p)),flush=True)
                if args.skill:
                    stable=p['mug_under_dispenser'] and not p['mug_grasped'] and not p['mug_contact']
                    placement_streak=placement_streak+1 if stable else 0
                    success=cup_skill_start is not None and placement_streak>=5 and not p['coffee_machine_on']
                if success or done or trunc:break
        finally:writer.close()
        (out/'video.partial.mp4').replace(out/'video.mp4');np.save(out/'actions.npy',np.asarray(actions));write(out/'trace.json',trace)
        if success:failure=None
        elif 'mug_moved_while_held' not in milestones:failure='mug_pickup_or_extraction'
        elif 'mug_under_dispenser' not in milestones:failure='mug_placement_under_dispenser'
        elif 'coffee_machine_on' not in milestones:failure='start_button_press'
        else:failure='final_placement_or_gripper_clearance'
        result=dict(task='PrepareCoffee',model=args.model,seed=seed,status='complete',success=success,steps=step,horizon=horizon,instruction=instruction,milestones=milestones,failure_stage=failure,
            initial_state_hash=initial_hash,scene_xml_sha256=xml_hash,final_predicates=p,seconds=time.time()-started,infer_seconds=sum(times),infer_calls=calls,training_updates=0,video=str(out/'video.mp4'),camera_keys=list(rollout.CAMERA_KEYS),video_stride=2,video_fps=20,split='pretrain',replan_steps=16,skill=args.skill,cup_skill_start=cup_skill_start,policy_version=getattr(policy,'version',0),official_task_success=p['official_success'])
        result.update(score(trace));capture.finalize(np.asarray(actions),trace,args.model,seed,instruction)
        write(out/'result.json',result);print(json.dumps(dict(event='VIDEO_READY',**result)),flush=True)
    finally:env.close()
write(root/'complete.json',dict(model=args.model,finished_at=time.time()))
