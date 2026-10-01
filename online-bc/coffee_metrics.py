"""Ordered skill metrics, independent of the full-task clearance predicate."""
import argparse
import json
from pathlib import Path

def score(trace, hold_steps=5):
    streak = 0
    placed = pressed = None
    previous_on = False
    for row in trace:
        on = bool(row['coffee_machine_on'])
        released = not row['mug_contact'] and not row['mug_grasped']
        stable = row['mug_under_dispenser'] and released
        streak = streak + 1 if stable else 0
        # Successful episodes terminate immediately. The simulator's official
        # success already certifies placement and release, so do not reject them
        # just because the terminal video has fewer than five remaining frames.
        terminal_success = bool(row.get('official_success', False)) and stable
        if placed is None and (streak >= hold_steps or terminal_success):
            placed = row['step']
        if placed is not None and row['step'] > placed and on and not previous_on and row['mug_under_dispenser']:
            pressed = row['step']
            break
        previous_on = on
    return dict(cup_placed=placed is not None, button_after_placement=pressed is not None,
                cup_placement_step=placed, button_press_step=pressed, placement_hold_steps=hold_steps)

def aggregate(root, target=30):
    rows=[]
    for path in Path(root).glob('*/result.json'):
        result=json.loads(path.read_text())
        result.update(score(json.loads((path.parent/'trace.json').read_text())))
        rows.append(result)
    models={}
    for model in ['xiaomi','pi05','groot']:
        selected=[r for r in rows if r['model']==model]
        n=len(selected); cups=sum(r['cup_placed'] for r in selected); buttons=sum(r['button_after_placement'] for r in selected)
        models[model]=dict(completed=n,target=target,cup_placed=cups,button_after_placement=buttons,
            cup_placement_rate=cups/n if n else None,
            conditional_button_rate=buttons/cups if cups else None,
            joint_success_rate=buttons/n if n else None,
            full_task_success=sum(r['success'] for r in selected))
    return dict(criterion='Cup: official dispenser placement + released (no gripper contact or grasp), 5 consecutive steps, or official terminal success certifying placement/release. Button: OFF→ON transition strictly after placement, cup still under dispenser. Conditional denominator: cup-placement successes.',models=models,episodes=rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--out',required=True);args=ap.parse_args()
    Path(args.out).write_text(json.dumps(aggregate(args.root),indent=2))
