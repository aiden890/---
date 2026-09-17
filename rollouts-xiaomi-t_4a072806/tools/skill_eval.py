"""Single-skill instruction rollouts for CloseBlenderLid with the unchanged Xiaomi VLA (t_4a072806).

Reuses /work/rollout.py (EvalClient, history sampling, state/video helpers, argparse
defaults: replan 16, obs history 4/2, crop 0.95, video stride 2 @ 20 fps).  Only the
instruction text and the simulator start state differ per skill.

Start states:
  GRASP        : official CloseBlenderLid reset(seed=S) (deterministic scene + robot pose).
  MOVE_HOLDING : snapshot taken from a *generation* episode (same env, full task
                 instruction, VLA-driven, no oracle) at the first moment the lid is
                 grasped + lifted and still far from the blender.
  PLACE        : snapshot from the same generation episode when the grasped lid is in
                 the pre-place region above the blender and lid_on_blender is False.
Snapshots are full MuJoCo qpos/qvel + sim time + gripper ramp state + blender fixture
state, restored into the same loaded model, then one hold step settles the frame.
Frames/steps of the generation episode are never part of a skill video; each skill
video frame 0 is the settled start state and the VLA's first action affects frame 1.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, "/work")
import rollout  # noqa: E402  (unchanged /work/rollout.py)

import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401
from robocasa.models.fixtures import Counter  # noqa: E402
from robocasa.utils import object_utils as OU  # noqa: E402
from robocasa.utils.env_utils import convert_action  # noqa: E402
from robosuite.utils.binding_utils import MjSimState  # noqa: E402

FULL_INSTRUCTION = "Close the lid blender by securely placing the lid on top."
SKILLS = {
    "grasp": "Pick up the blender lid securely.",
    "move_holding": "Move the grasped blender lid above the blender without colliding with the surroundings.",
    "place": "Place the grasped blender lid securely on top of the blender, release it, and move the gripper away.",
}
# thresholds (metres / steps) -- decided from robocasa source: Blender._BLENDER_LID_POS_THRESH=0.04,
# CloseBlenderLid success = lid_on_blender and gripper_fxtr_far(th=0.15)
LIFT_DZ = 0.05            # lid z above its reset rest height counts as lifted
PREPLACE_XY = 0.06        # xy distance lid -> lid_closed_pos
PREPLACE_DZ = (0.015, 0.25)  # z above lid_closed_pos
GRASP_HOLD_STEPS = 20     # consecutive steps for GRASP success
MOVE_HOLD_STEPS = 3       # consecutive steps in pre-place region for MOVE success
SNAP_MOVE_STEPS = 5       # consecutive grasped+lifted+far steps before MOVE snapshot
SNAP_PLACE_STEPS = 3      # consecutive pre-place steps before PLACE snapshot


def f(x):
    return [float(v) for v in np.asarray(x).reshape(-1)]


class Sim:
    """Read-only predicate access + snapshot/restore on one gym env instance."""

    def __init__(self, genv):
        self.genv = genv                      # gymnasium wrapper chain
        self.g = genv.unwrapped               # RoboCasaGymEnv
        self.k = self.g.env                   # robocasa Kitchen (CloseBlenderLid); hard_reset rebuilds robots/fixtures
        self.rest_lid_pos = None
        self.rest_xy_to_closed = None

    # references are looked up live because every reset() reloads the model
    @property
    def robot(self):
        return self.k.robots[0]

    @property
    def gripper(self):
        return self.robot.gripper["right"]

    @property
    def lid(self):
        return self.k.blender.blender_lid

    @property
    def lid_body(self):
        return f"{self.lid.name}_main"

    # ---- geometry ---------------------------------------------------------
    def lid_pos(self):
        return np.array(self.k.sim.data.get_body_xpos(self.lid_body))

    def eef_pos(self):
        return np.array(self.k.sim.data.site_xpos[self.robot.eef_site_id["right"]])

    def closed_pos(self):
        return np.array(self.k.blender.get_lid_closed_pos(self.k))

    def finger_qpos(self):
        return f(self.k.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes["right"]])

    def lid_on_counter(self):
        return any(
            self.k.check_contact(self.lid, fx) for fx in self.k.fixtures.values() if isinstance(fx, Counter)
        )

    def lid_other_contacts(self):
        gripper_geoms = set(self.gripper.contact_geoms)
        lid_geoms = set(self.lid.contact_geoms)
        others = set()
        for i in range(self.k.sim.data.ncon):
            c = self.k.sim.data.contact[i]
            n1 = self.k.sim.model.geom_id2name(c.geom1)
            n2 = self.k.sim.model.geom_id2name(c.geom2)
            if n1 in lid_geoms and n2 not in lid_geoms and n2 not in gripper_geoms:
                others.add(n2 or f"geom{c.geom2}")
            if n2 in lid_geoms and n1 not in lid_geoms and n1 not in gripper_geoms:
                others.add(n1 or f"geom{c.geom1}")
        return sorted(others)

    def predicates(self):
        lid = self.lid_pos()
        closed = self.closed_pos()
        contact = bool(self.k.check_contact(self.gripper, self.lid))
        on_counter = self.lid_on_counter()
        dz_rest = float(lid[2] - self.rest_lid_pos[2]) if self.rest_lid_pos is not None else None
        lifted = dz_rest is not None and dz_rest > LIFT_DZ and not on_counter
        grasped = contact and lifted
        xy = float(np.linalg.norm(lid[:2] - closed[:2]))
        dz_closed = float(lid[2] - closed[2])
        preplace = xy < PREPLACE_XY and PREPLACE_DZ[0] < dz_closed < PREPLACE_DZ[1]
        st = self.k.blender.get_state()
        return {
            "gripper_lid_contact": contact,
            "lid_on_counter": on_counter,
            "lid_lifted": bool(lifted),
            "lid_grasped": bool(grasped),
            "lid_dz_from_rest": dz_rest,
            "lid_xy_to_closed_pos": xy,
            "lid_dz_to_closed_pos": dz_closed,
            "lid_dist_to_closed_pos": float(np.linalg.norm(lid - closed)),
            "in_preplace_region": bool(preplace),
            "lid_on_blender": bool(st["lid_on_blender"]),
            "lid_upright_7deg": bool(OU.check_fxtr_upright(self.k, self.lid_body, th=7)),
            "gripper_lid_far_0.15": bool(OU.gripper_fxtr_far(self.k, self.lid_body, th=0.15)),
            "eef_lid_dist": float(np.linalg.norm(self.eef_pos() - lid)),
            "official_check_success": bool(self.k._check_success()),
            "finger_qpos": self.finger_qpos(),
            "gripper_ramp_action": f(self.gripper.current_action),
            "lid_other_contacts": self.lid_other_contacts(),
            "lid_pos": f(lid),
            "eef_pos": f(self.eef_pos()),
            "base_pos": f(self.k.sim.data.site_xpos[self.k.sim.model.site_name2id(self.robot.robot_model.base.correct_naming("center"))]),
        }

    def scene_meta(self):
        m = self.k.sim.model
        bid = m.body_name2id(self.lid_body)
        body_ids = {bid}
        for b in range(m.nbody):  # include child bodies of the lid
            if m.body_parentid[b] in body_ids:
                body_ids.add(b)
        geoms = []
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] in body_ids:
                geoms.append({
                    "name": m.geom_id2name(gid),
                    "type": int(m.geom_type[gid]),
                    "size": f(m.geom_size[gid]),
                    "pos": f(m.geom_pos[gid]),
                    "group": int(m.geom_group[gid]),
                    "contype": int(m.geom_contype[gid]),
                })
        ep = json.loads(json.dumps(self.k.get_ep_meta(), default=str))
        return {
            "layout_id": ep.get("layout_id"), "style_id": ep.get("style_id"),
            "lid_fixture_name": self.lid.name,
            "lid_xml": str(getattr(self.lid, "file", None)),
            "lid_size_wdh": f(getattr(self.lid, "size", [])),
            "lid_scale": f(getattr(self.lid, "_scale", getattr(self.lid, "scale", []))),
            "blender_fixture_name": self.k.blender.name,
            "blender_xml": str(getattr(self.k.blender, "file", None)),
            "blender_pos": f(self.k.blender.pos), "blender_rot": float(self.k.blender.rot),
            "lid_closed_pos": f(self.closed_pos()),
            "lid_geoms": geoms,
            "init_robot_base_pos": f(getattr(self.k, "init_robot_base_pos", [])),
            "init_robot_base_ori": f(getattr(self.k, "init_robot_base_ori", [])),
            "ep_meta_fixtures": ep.get("fixtures"),
        }

    # ---- snapshot / restore ----------------------------------------------
    def snapshot(self, tag, step):
        s = self.k.sim.get_state()
        b = self.k.blender
        return {
            "tag": tag, "gen_step": step, "time": float(s.time),
            "nq": int(s.qpos.shape[0]), "lid_qpos_addr": int(self.k.sim.model.get_joint_qpos_addr(self.lid.joints[0])[0]),
            "qpos": s.qpos.copy(), "qvel": s.qvel.copy(),
            "gripper_current_action": np.array(self.gripper.current_action, copy=True),
            "blender": (bool(b._lid_on_blender), bool(b._turned_on), bool(b._button_contact_prev_timestep)),
            "predicates": self.predicates(),
        }

    def restore(self, snap):
        k = self.k
        cur_addr = int(k.sim.model.get_joint_qpos_addr(self.lid.joints[0])[0])
        if k.sim.data.qpos.shape[0] != snap["nq"] or cur_addr != snap["lid_qpos_addr"]:
            raise RuntimeError("loaded model differs from snapshot model; cannot restore state")
        k.sim.set_state(MjSimState(snap["time"], snap["qpos"].copy(), snap["qvel"].copy()))
        k.sim.forward()
        self.gripper.current_action = np.array(snap["gripper_current_action"], copy=True)
        b = k.blender
        b._lid_on_blender, b._turned_on, b._button_contact_prev_timestep = snap["blender"]
        # same sequence robot.reset() uses after setting qpos: controller origin + goals := current
        self.robot.composite_controller.update_state()
        self.robot.composite_controller.reset()
        k.timestep = 0
        k.cur_time = float(snap["time"])
        k.done = False


def hold_action(gripper_closed: bool):
    a = np.zeros(12, dtype=np.float32)
    a[6] = 1.0 if gripper_closed else 0.0   # unmap: <0.5 -> open(-1), else closed(+1)
    return a


def run_episode(sim, client, args, instruction, obs, horizon, success_fn, log_path, video_frames, log_extra,
                post_success_hold=0):
    """VLA loop mirrored from rollout.evaluate_task (same chunk/replan/history handling).

    post_success_hold=0 (default): identical to the original behaviour -- break the moment
    ``success_fn`` first latches (or on done/trunc/horizon).
    post_success_hold=N>0: after the skill first succeeds, DO NOT break; keep issuing the
    same instruction for N more steps, logging predicates/frames, so the "does it stop after
    success or keep overfitting into the next motion?" question can be observed. The success
    flag is latched at ``success_step`` and stays True even if the predicate later drops.
    """
    queue_length = (args.obs_history - 1) * args.obs_interval + 1
    image_queues = {key: collections.deque(maxlen=queue_length) for key in rollout.CAMERA_KEYS}
    state_queue = collections.deque(maxlen=queue_length)
    for key, image in rollout.collect_images(obs).items():
        image_queues[key].append(image)
    state_queue.append(rollout.observation_to_state(obs))
    action_plan = collections.deque()
    steps, success, infer_calls, infer_seconds = 0, False, 0, 0.0
    hold = 0
    success_step = None
    done = trunc = False
    p = sim.predicates()
    with open(log_path, "w") as log:
        log.write(json.dumps({"type": "start", "instruction": instruction, "horizon": horizon,
                              "post_success_hold": int(post_success_hold), **log_extra}) + "\n")
        while steps < horizon:
            if not action_plan:
                states = rollout.sample_history(state_queue, args.obs_history, args.obs_interval)
                images = {key: rollout.sample_history(q, args.obs_history, args.obs_interval) for key, q in image_queues.items()}
                t0 = time.time()
                chunk = client.infer(states, images, instruction)
                infer_seconds += time.time() - t0
                infer_calls += 1
                if len(chunk) < args.replan_steps:
                    raise RuntimeError(f"Model returned {len(chunk)} actions < replan {args.replan_steps}")
                action_plan.extend(chunk[: args.replan_steps])
                log.write(json.dumps({"type": "chunk", "step": steps, "chunk_len": int(len(chunk)), "executed": int(args.replan_steps)}) + "\n")
            a = np.asarray(action_plan.popleft(), dtype=np.float32)
            obs, _, done, trunc, info = sim.genv.step(convert_action(a))
            steps += 1
            for key, image in rollout.collect_images(obs).items():
                image_queues[key].append(image)
            state = rollout.observation_to_state(obs)
            state_queue.append(state)
            p = sim.predicates()
            hold = hold + 1 if success_fn(p) else 0
            step_success = hold >= success_fn.hold
            if step_success and success_step is None:
                success_step = steps
                success = True  # latch: the skill succeeded at least once this episode
                log.write(json.dumps({"type": "success", "step": steps, "success_step": steps}) + "\n")
            phase = "post_success" if success_step is not None else "pre_success"
            rec = {"type": "step", "step": steps, "action": f(a),
                   "proprio": f(state), "predicates": p,
                   "official_success": bool(info.get("success", False)), "success_hold": hold,
                   "success_step": success_step, "phase": phase}
            log.write(json.dumps(rec) + "\n")
            if steps % args.video_stride == 0 or step_success or done or trunc:
                video_frames.append(rollout.make_video_frame(obs))
                rec_frame = len(video_frames) - 1
                log.write(json.dumps({"type": "frame", "step": steps, "frame_index": rec_frame}) + "\n")
            if done or trunc:
                break
            if success_step is not None:
                if post_success_hold <= 0:
                    break  # legacy: stop immediately on first success
                if steps - success_step >= post_success_hold:
                    break  # held the requested number of extra steps after success
    if success_step is not None and post_success_hold > 0:
        terminated_by = "post_success_hold"
    elif success:
        terminated_by = "success"
    elif done:
        terminated_by = "done"
    elif trunc:
        terminated_by = "truncated"
    else:
        terminated_by = "horizon"
    return {"success": bool(success), "steps": steps, "success_step": success_step,
            "post_success_hold": int(post_success_hold),
            "post_success_steps": (steps - success_step) if success_step is not None else 0,
            "infer_calls": infer_calls, "infer_seconds": infer_seconds,
            "final_predicates": p, "terminated_by": terminated_by}


class Cond:
    def __init__(self, fn, hold):
        self.fn, self.hold = fn, hold

    def __call__(self, p):
        return self.fn(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--gen-attempts", type=int, default=6)
    ap.add_argument("--gen-horizon", type=int, default=900)
    ap.add_argument("--horizon-grasp", type=int, default=300)
    ap.add_argument("--horizon-move", type=int, default=300)
    ap.add_argument("--horizon-place", type=int, default=400)
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=10086)
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--skills", default="grasp,move_holding,place")
    ap.add_argument("--instruction-grasp", default=None)
    ap.add_argument("--instruction-move", default=None)
    ap.add_argument("--instruction-place", default=None)
    ap.add_argument("--post-success-hold", type=int, default=0,
                    help="After a skill first succeeds, keep issuing the same instruction for N "
                         "more steps (default 0 = legacy break-on-success). Only affects skill "
                         "episodes; generation snapshot episodes are unchanged.")
    my, rest = ap.parse_known_args()
    args = rollout.parse_args(rest)  # verified defaults: replan 16, obs 4/2, crop 0.95, stride 2, fps 20
    rollout.validate_args(args)
    out = Path(my.out)
    out.mkdir(parents=True, exist_ok=True)

    client = rollout.EvalClient(my.model_path, my.server_addr, my.server_port, args.robot_type, args.crop_ratio)
    genv = gym.make("robocasa/CloseBlenderLid", split=args.split, seed=my.seed)
    sim = Sim(genv)
    try:
        obs, _ = rollout.reset_env(genv, my.seed)
        assert obs["annotation.human.task_description"] == FULL_INSTRUCTION
        sim.rest_lid_pos = sim.lid_pos()
        sim.rest_xy_to_closed = float(np.linalg.norm(sim.rest_lid_pos[:2] - sim.closed_pos()[:2]))
        move_far_xy = min(0.12, 0.5 * sim.rest_xy_to_closed)
        scene = sim.scene_meta()
        scene["rest_lid_pos"] = f(sim.rest_lid_pos)
        scene["rest_xy_to_closed"] = sim.rest_xy_to_closed
        scene["move_start_min_xy_to_closed"] = move_far_xy
        grasp_snap = sim.snapshot("grasp_reset", 0)
        (out / "scene.json").write_text(json.dumps({"seed": my.seed, "split": args.split, **scene,
            "reset_predicates": grasp_snap["predicates"], "thresholds": {"LIFT_DZ": LIFT_DZ, "PREPLACE_XY": PREPLACE_XY,
            "PREPLACE_DZ": PREPLACE_DZ, "GRASP_HOLD_STEPS": GRASP_HOLD_STEPS, "MOVE_HOLD_STEPS": MOVE_HOLD_STEPS}}, indent=2))
        wanted = my.skills.split(",")
        instructions = dict(SKILLS)
        if my.instruction_grasp is not None:
            instructions["grasp"] = my.instruction_grasp
        if my.instruction_move is not None:
            instructions["move_holding"] = my.instruction_move
        if my.instruction_place is not None:
            instructions["place"] = my.instruction_place

        # ---- generation episodes (VLA, full instruction) for MOVE/PLACE snapshots ----
        snaps = {}
        gen_summary = []
        if any(s in wanted for s in ("move_holding", "place")):
            for attempt in range(my.gen_attempts):
                if attempt > 0:
                    obs, _ = rollout.reset_env(genv, my.seed)
                found = {}
                cnt_move = cnt_place = 0

                def gen_cond(p):
                    nonlocal cnt_move, cnt_place
                    cnt_move = cnt_move + 1 if (p["lid_grasped"] and p["lid_xy_to_closed_pos"] > move_far_xy) else 0
                    cnt_place = cnt_place + 1 if (p["lid_grasped"] and p["in_preplace_region"] and not p["lid_on_blender"]) else 0
                    if "move_holding" not in found and cnt_move >= SNAP_MOVE_STEPS:
                        found["move_holding"] = None  # marker; snapshot taken below
                    if "place" not in found and cnt_place >= SNAP_PLACE_STEPS:
                        found["place"] = None
                    return "place" in found

                # wrap genv.step so the snapshot is taken at the exact step the condition first holds
                orig_step = sim.genv.step
                gen_step = [0]

                def snap_step(action, _orig=orig_step):
                    r = _orig(action)
                    gen_step[0] += 1
                    p = sim.predicates()
                    gen_cond(p)
                    for k in list(found):
                        if found[k] is None:
                            found[k] = sim.snapshot(k, gen_step[0])
                    return r
                sim.genv.step = snap_step
                gen_frames = []
                res = run_episode(sim, client, args, FULL_INSTRUCTION, obs, my.gen_horizon,
                                  Cond(lambda p: "place" in found and found["place"] is not None, 1),
                                  out / f"generation_attempt{attempt}.jsonl", gen_frames,
                                  {"kind": "generation", "attempt": attempt, "seed": my.seed})
                sim.genv.step = orig_step
                for k in list(found):
                    if found[k] is not None and k not in snaps:
                        snaps[k] = found[k]
                gen_summary.append({"attempt": attempt, "steps": res["steps"], "found": {k: v["gen_step"] for k, v in found.items() if v is not None},
                                    "terminated_by": res["terminated_by"], "final_predicates": res["final_predicates"]})
                imageio.mimsave(out / f"generation_attempt{attempt}.mp4", gen_frames, fps=args.video_fps)
                if "move_holding" in snaps and "place" in snaps:
                    break
        (out / "generation_summary.json").write_text(json.dumps(gen_summary, indent=2, default=str))

        # ---- skill episodes ----
        conds = {
            "grasp": Cond(lambda p: p["lid_grasped"], GRASP_HOLD_STEPS),
            "move_holding": Cond(lambda p: p["lid_grasped"] and p["in_preplace_region"], MOVE_HOLD_STEPS),
            "place": Cond(lambda p: p["official_check_success"], 1),
        }
        horizons = {"grasp": my.horizon_grasp, "move_holding": my.horizon_move, "place": my.horizon_place}
        results = {}
        # Snapshot-based MOVE/PLACE must run before GRASP's env reset. Some seeds
        # select a different fixture model on reset; restoring a snapshot across
        # model variants is invalid. GRASP is independent and safely runs last.
        episode_order = [s for s in wanted if s != "grasp"] + \
                        [s for s in wanted if s == "grasp"]
        for skill in episode_order:
            rec = {"skill": skill, "instruction": instructions[skill], "seed": my.seed, "horizon": horizons[skill],
                   "replan_steps": args.replan_steps, "obs_history": args.obs_history, "obs_interval": args.obs_interval,
                   "crop_ratio": args.crop_ratio, "video_stride": args.video_stride, "video_fps": args.video_fps,
                   "model_path": my.model_path}
            if skill == "grasp":
                obs, _ = rollout.reset_env(genv, my.seed)
                rec["start_state_source"] = "official reset(seed) (deterministic scene)"
                rec["init_step_kind"] = "reset"
            else:
                if skill not in snaps:
                    rec.update({"status": "skipped", "reason": "generation episodes never produced the start state"})
                    results[skill] = rec
                    (out / f"{skill}.json").write_text(json.dumps(rec, indent=2, default=str))
                    continue
                snap = snaps[skill]
                sim.restore(snap)
                rec["start_state_source"] = f"snapshot from generation attempt at gen step {snap['gen_step']} (VLA-driven, full instruction; excluded from video)"
                rec["snapshot_predicates_before_restore"] = snap["predicates"]
                # one hold step to settle physics (gripper kept closed); recorded as init, not VLA
                obs, _, _, _, _ = sim.genv.step(convert_action(hold_action(True)))
                rec["init_step_kind"] = "restore + 1 hold step (zero motion, gripper closed)"
                np.savez_compressed(out / f"{skill}_start_state.npz", qpos=snap["qpos"], qvel=snap["qvel"], time=snap["time"],
                                    gripper_current_action=snap["gripper_current_action"])
            rec["init_predicates"] = sim.predicates()
            rec["scene"] = {k: scene[k] for k in ("layout_id", "style_id", "lid_fixture_name", "lid_xml", "lid_size_wdh", "lid_closed_pos", "rest_lid_pos")}
            frames = [rollout.make_video_frame(obs)]
            rec["vla_start_frame_index"] = 1
            rec["frame0"] = "settled start state (before any VLA action)"
            res = run_episode(sim, client, args, instructions[skill], obs, horizons[skill], conds[skill],
                              out / f"{skill}_steps.jsonl", frames, {"kind": "skill", "skill": skill},
                              post_success_hold=my.post_success_hold)
            rec.update(res)
            if not res["success"]:
                fp = res["final_predicates"]
                if skill == "grasp":
                    rec["failure_reason"] = "never held lid lifted for %d consecutive steps (contact=%s lifted=%s)" % (GRASP_HOLD_STEPS, fp["gripper_lid_contact"], fp["lid_lifted"])
                elif skill == "move_holding":
                    rec["failure_reason"] = "lid dropped" if not fp["lid_grasped"] else "lid never reached pre-place region (xy=%.3f dz=%.3f)" % (fp["lid_xy_to_closed_pos"], fp["lid_dz_to_closed_pos"])
                else:
                    rec["failure_reason"] = "lid_on_blender=%s gripper_far=%s dist_to_closed=%.3f" % (fp["lid_on_blender"], fp["gripper_lid_far_0.15"], fp["lid_dist_to_closed_pos"])
            video = out / f"{skill}.mp4"
            imageio.mimsave(video, frames, fps=args.video_fps)
            rec["video"] = video.name
            rec["video_frames"] = len(frames)
            rec["status"] = "done"
            (out / f"{skill}.json").write_text(json.dumps(rec, indent=2, default=str))
            results[skill] = {k: rec[k] for k in ("success", "steps", "success_step", "post_success_steps", "terminated_by", "instruction") if k in rec}
            print(json.dumps({skill: results[skill]}), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    finally:
        genv.close()
        client.close()


if __name__ == "__main__":
    main()
