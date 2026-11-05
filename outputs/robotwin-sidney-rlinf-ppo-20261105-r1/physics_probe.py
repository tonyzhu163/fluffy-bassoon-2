"""Read exposed SAPIEN state; this is not a complete physics checkpoint."""
import numpy as np
import sapien


def pose(value):
    return dict(p=np.asarray(value.p).tolist(),q=np.asarray(value.q).tolist())


def snapshot(env):
    rows=[]
    for sub in env.venv.envs:
        task=sub.task
        articulations=[]
        for art in task.scene.get_all_articulations():
            articulations.append(dict(name=art.get_name(),qpos=art.get_qpos().tolist(),
                qvel=art.get_qvel().tolist(),root_pose=pose(art.get_root_pose()),
                root_linear_velocity=art.get_root_linear_velocity().tolist(),
                root_angular_velocity=art.get_root_angular_velocity().tolist()))
        actors=[]
        for actor in task.scene.get_all_actors():
            row=dict(name=actor.name,pose=pose(actor.get_pose()))
            body=actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if body is not None:
                row.update(linear_velocity=body.get_linear_velocity().tolist(),
                    angular_velocity=body.get_angular_velocity().tolist(),is_sleeping=bool(body.is_sleeping))
            actors.append(row)
        contacts=[]
        for contact in task.scene.get_contacts():
            contacts.append(dict(bodies=[body.entity.name for body in contact.bodies],
                points=[dict(position=p.position.tolist(),normal=p.normal.tolist(),
                    impulse=p.impulse.tolist(),separation=float(p.separation)) for p in contact.points]))
        fields=('run_steps','reward_step','take_action_cnt','eval_success','plan_success',
                'left_cnt','right_cnt','step_lim','ep_num','FRAME_IDX')
        bookkeeping={k:(getattr(task,k).item() if isinstance(getattr(task,k),np.generic)
                        else getattr(task,k)) for k in fields if hasattr(task,k)}
        rows.append(dict(requested_seed=int(sub.env_seed),actual_seed=int(task.ep_num),
            scene_info=task.get_info(),instruction=task.get_instruction(),
            articulations=articulations,actors=actors,contacts=contacts,bookkeeping=bookkeeping))
    wrapper={key:getattr(env,key).detach().cpu().tolist()
        for key in ('prev_step_reward','success_once','fail_once','returns','_elapsed_steps')
        if hasattr(env,key)}
    return dict(environments=rows,wrapper=wrapper,complete_native_physics_snapshot=False)
