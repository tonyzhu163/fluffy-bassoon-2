"""Narrow pre-GAE reward mixin for the existing official/native PPO actor.

No Flora import or scorer in the PPO environment. Environment/video/sampler and
full pending-batch recovery qualifications remain explicit readiness stages.
"""
import json
from pathlib import Path

from .rlinf_flora_ppo import digest,ready_rewards,sha,write_new


def read_ref(ref):
    path=Path(ref['path'])
    if sha(path)!=ref['sha256']:raise ValueError('Immutable batch/label reference changed')
    return json.loads(path.read_text())


def validate_pending_checkpoint(batch,checkpoint_ref):
    if batch['rollout']['checkpoint']!=checkpoint_ref:
        raise ValueError('Pending on-policy batch belongs to a different preceding model/Adam checkpoint')
    checkpoint=read_ref(checkpoint_ref)
    if batch['policy_sha256']!=digest(checkpoint['model_optimizer_files']):
        raise ValueError('Pending producing-policy file digest differs from restored boundary')
    for row in checkpoint['model_optimizer_files']:
        if sha(row['path'])!=row['sha256']:raise ValueError('Restored model/Adam checkpoint bytes changed')
    return checkpoint


class TerminalRewardActorMixin:
    def _reward_directory(self):
        return Path(self.cfg.flora_ppo.exchange_root)/self.cfg.flora_ppo.arm/'train'/f'iteration-{int(self.version)+1:04d}'

    def capture_pending_for_labels(self):
        from .rlinf_native_ppo import cpu_tree,tensor_hash,snapshot_rng
        from .rlinf_native_ppo_runtime import save_tensor
        root=self._reward_directory()/f'rank-{self._rank:02d}'
        if 'advantages' in self.rollout_batch or 'returns' in self.rollout_batch:
            raise ValueError('Only the original unprocessed fresh batch may be labeled')
        if (root/'pending.pt').exists():raise ValueError('Never overwrite or recollect a pending on-policy batch')
        ref=save_tensor(root/'pending.pt',dict(rollout_batch=cpu_tree(self.rollout_batch),rng=snapshot_rng(),
            producer_iteration=int(self.version),optimizer_steps=self.optimizer_steps))
        ref.update(canonical_sha256=tensor_hash(self.rollout_batch),producer_iteration=int(self.version),
            optimizer_steps=self.optimizer_steps,rank=self._rank)
        write_new(root/'pending.json',ref)
        return dict(path=str(root/'pending.json'),sha256=sha(root/'pending.json'),rank=self._rank)

    def install_reward_refs(self,batch_ref,ready_ref):
        batch=read_ref(batch_ref);ready=read_ref(ready_ref)
        if (batch['producer_iteration']!=int(self.version) or batch['arm']!=self.cfg.flora_ppo.arm
                or ready['batch_sha256']!=digest(batch) or ready.get('status')!='READY_FOR_GAE'):
            raise ValueError('Only this complete fresh on-policy batch may be installed')
        self._flora_refs=(batch_ref,ready_ref)
        return dict(rank=self._rank,producer_iteration=int(self.version),batch_sha256=digest(batch))

    def flora_capture(self,directory,**kwargs):
        ref=self.pilot_capture(directory,retain_rollout=False)
        return dict(ref,role='actor',rank=self._rank,complete=True,optimizer_sampler_RNG=True)

    def flora_memory(self):
        import torch
        return dict(rank=self._rank,current_bytes=torch.cuda.memory_allocated(),
            peak_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())

    def flora_restore(self,directory,pending_directory=None):
        result=self.pilot_restore(directory)
        if pending_directory is not None:
            import torch
            from .rlinf_native_ppo import restore_rng,tensor_hash
            ref=json.loads((Path(pending_directory)/f'rank-{self._rank:02d}'/'pending.json').read_text())
            if sha(ref['path'])!=ref['sha256']:raise ValueError('Pending rollout bytes changed')
            state=torch.load(ref['path'],map_location='cpu',weights_only=False)
            if state['producer_iteration']!=int(self.version) or state['optimizer_steps']!=self.optimizer_steps:
                raise ValueError('Pending batch does not belong to restored model/optimizer')
            self.rollout_batch=state['rollout_batch']
            if tensor_hash(self.rollout_batch)!=ref['canonical_sha256']:raise ValueError('Pending batch semantic hash changed')
            restore_rng(state['rng'])
        return result

    def compute_advantages_and_returns(self):
        import torch
        from .rlinf_native_ppo import tensor_hash
        from rlinf.utils.metric_utils import compute_loss_mask
        root=self._reward_directory()
        if getattr(self,'_flora_refs',None) is None:raise ValueError('Wait for all batch labels before GAE')
        batch,ready=map(read_ref,self._flora_refs)
        if (batch['arm']!=self.cfg.flora_ppo.arm or batch['producer_iteration']!=int(self.version)
                or batch['iteration']!=int(self.version)+1 or batch.get('pending_recovery_complete') is not True
                or 'advantages' in self.rollout_batch or 'returns' in self.rollout_batch):
            raise ValueError('All-rank complete pending-batch checkpoint and fresh policy version required before GAE')
        pending=read_ref({'path':str(root/f'rank-{self._rank:02d}'/'pending.json'),
            'sha256':batch['rank_pending_receipts'][str(self._rank)]['sha256']})
        if pending['canonical_sha256']!=tensor_hash(self.rollout_batch) or pending['optimizer_steps']!=self.optimizer_steps:
            raise ValueError('Actor batch/state changed while labeling; hold instead of falling back')
        ids=batch['episode_ids_by_rank'][str(self._rank)]
        raw=self.rollout_batch['rewards'];dones=self.rollout_batch['dones'];truncations=self.rollout_batch['truncations']
        if (dones.shape[0]!=raw.shape[0]+1 or not torch.all(truncations[-1,:,-1])
                or torch.any(truncations[:-1]) or torch.any(self.rollout_batch['terminations'])):
            raise ValueError('Require actual completeH400 horizon boundary without success stopping or abort')
        mask,_=compute_loss_mask(dones)  # Actual upstream transition mask; never manufacture valid slots.
        reward=ready_rewards(batch,ready,raw.detach().cpu().numpy(),mask.detach().cpu().numpy(),ids)
        untouched={k:tensor_hash(v) for k,v in self.rollout_batch.items() if k!='rewards'}
        self.rollout_batch['rewards']=torch.as_tensor(reward,device=raw.device,dtype=raw.dtype)
        if any(tensor_hash(self.rollout_batch[k])!=v for k,v in untouched.items()):raise RuntimeError('Reward adapter altered PPO inputs/masks/bootstrap')
        result=super().compute_advantages_and_returns() # Existing PilotActor -> official GAE, unchanged.
        self._flora_ready=(digest(batch),int(self.version))
        return result

    def run_training(self,*args,**kwargs):
        root=self._reward_directory()/f'rank-{self._rank:02d}'
        if getattr(self,'_flora_ready',None) is None or self._flora_ready[1]!=int(self.version):
            raise ValueError('No policy update before the entire batch is labeled and GAE completed')
        write_new(root/'UPDATE-STARTED.json',dict(batch_sha256=self._flora_ready[0],producer_iteration=int(self.version),
            optimizer_steps_before=self.optimizer_steps))
        result=super().run_training(*args,**kwargs)  # Existing PPO/critic/Adam unchanged.
        write_new(root/'UPDATE-COMPLETE.json',dict(batch_sha256=self._flora_ready[0],producer_iteration=int(self.version),
            optimizer_steps_after=self.optimizer_steps,full_iteration_recovery='Required from existing runner before next collection'))
        self._flora_ready=None
        self._flora_refs=None
        return result


def actor_class():
    # Lazy: importing metadata or synthetic helpers does not import RLinf.
    from .rlinf_native_ppo_runtime import PilotActor
    class MatchedFloraActor(TerminalRewardActorMixin,PilotActor):
        pass
    return MatchedFloraActor


def rollout_class():
    from .rlinf_native_ppo_runtime import PilotRollout,save_tensor
    from .rlinf_native_ppo import snapshot_rng,restore_rng
    class MatchedRollout(PilotRollout):
        def init_worker(self):
            result=super().init_worker()
            self._flora_eval_rng=snapshot_rng()
            return result

        def flora_evaluation_rng(self,restore_training=False):
            if restore_training:restore_rng(self._flora_training_rng)
            else:
                self._flora_training_rng=snapshot_rng()
                restore_rng(self._flora_eval_rng)
            return dict(rank=self._rank,restored_training=restore_training)

        def flora_capture(self,directory,**kwargs):
            ref=save_tensor(Path(directory)/f'rollout-{self._rank:02d}.pt',dict(rng=snapshot_rng(),
                eval_rng=self._flora_eval_rng,prediction_ordinal=self._pilot_prediction_ordinal,
                version=self.version,rank=self._rank))
            return dict(ref,role='rollout',rank=self._rank,complete=True,optimizer_sampler_RNG=True)

        def flora_restore(self,directory,**kwargs):
            import torch
            state=torch.load(Path(directory)/f'rollout-{self._rank:02d}.pt',map_location='cpu',weights_only=False)
            if state['rank']!=self._rank:raise ValueError('Rollout rank changed')
            self.version=state['version'];self._pilot_producing_policy_iteration=self.version
            self._flora_eval_rng=state['eval_rng'];self._pilot_prediction_ordinal=state['prediction_ordinal']
            restore_rng(state['rng'])
            return dict(role='rollout',rank=self._rank,restored=True)
    return MatchedRollout
