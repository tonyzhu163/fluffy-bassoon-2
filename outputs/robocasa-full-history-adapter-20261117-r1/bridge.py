"""Causal trial sampling and native closed-loop KV inference for this campaign."""
import json
from pathlib import Path

import numpy as np
import torch

from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, native_flow_targets


class Corpus:
    def __init__(self, trials, *, privileged=False):
        self.privileged = privileged
        self.trials, self.episodes = [], []
        for paths in trials:
            ids = []
            for path in paths:
                path = Path(path)
                meta = json.loads((path/'result.json').read_text())
                with np.load(path/'episode.npz') as archive:
                    data = {k: archive[k] for k in ('features','states','images','teacher_actions',
                                                  'action_masks','feedback','rewards','dones','noise')}
                assert len(data['features']) == len(meta['rows']) and data['dones'][-1]
                if privileged:
                    active = np.array([r['active'] for r in meta['rows']], np.float32)
                    extra = active[:, None] * np.asarray(meta['offset'], np.float32)[None] / .04
                    data['features'] = np.concatenate((data['features'], extra), -1)
                data['meta'] = meta
                ids.append(len(self.episodes)); self.episodes.append(data)
            if len(ids) > 1:
                first = self.episodes[ids[0]]['meta']
                assert all(self.episodes[i]['meta']['scene'] == first['scene'] and
                           self.episodes[i]['meta']['offset'] == first['offset'] and
                           self.episodes[i]['meta']['start_signature'] == first['start_signature'] for i in ids)
            self.trials.append(ids)
        self.strata = []
        for trial, ids in enumerate(self.trials):
            # A public-history learner is supervised only on attempts 2 and 3.
            queries = [(trial, ep, q) for ep in range(0 if privileged else 1, len(ids))
                       for q in range(len(self.episodes[ids[ep]]['features']))]
            assert queries
            self.strata.append(queries)

    def episode(self, index):
        trial, ep, q = index
        return self.episodes[self.trials[trial][ep]], q

    def tokens(self, index, mode):
        trial, ep, q = index
        data, _ = self.episode(index)
        dim = data['features'].shape[-1]
        if mode in ('current', 'privileged'):
            return np.concatenate((data['features'][q], np.zeros(602, np.float32), [q == 0]))[None].astype(np.float32)
        tokens, previous = [], np.zeros(602, np.float32)
        for j, eid in enumerate(self.trials[trial][:ep+1]):
            item = self.episodes[eid]
            end = q+1 if j == ep else len(item['features'])
            for k in range(end):
                tokens.append(np.concatenate((item['features'][k], previous, [k == 0])))
                previous = np.concatenate((item['feedback'][k], [item['rewards'][k], item['dones'][k]]))
        result = np.asarray(tokens, np.float32)
        assert result.shape[1] == dim+603 and len(result) <= 360
        return result


class BalancedQueries:
    """Shuffle each scene/condition's query positions once before recycling."""
    def __init__(self, corpus, seed):
        self.corpus = corpus
        self.rng = np.random.default_rng(seed)
        self.pending = [[] for _ in corpus.strata]
        self.stratum_order = []
        self.coverage = [np.zeros(len(s), np.int64) for s in corpus.strata]

    def draw(self, count):
        result = []
        for _ in range(count):
            if not self.stratum_order:
                self.stratum_order = self.rng.permutation(len(self.pending)).tolist()
            s = self.stratum_order.pop()
            if not self.pending[s]:
                self.pending[s] = self.rng.permutation(len(self.corpus.strata[s])).tolist()
            q = self.pending[s].pop(); self.coverage[s][q] += 1
            result.append(self.corpus.strata[s][q])
        return result

    def state(self):
        return dict(pending=self.pending, stratum_order=self.stratum_order,
                    coverage=self.coverage, rng=self.rng.bit_generator.state)

    def restore(self, state):
        self.pending = state['pending']; self.stratum_order = state['stratum_order']
        self.coverage = state['coverage']; self.rng.bit_generator.state = state['rng']


def batch(corpus, indices, policy, codec, stats, mode, generator):
    windows = [corpus.tokens(i, mode) for i in indices]
    width, length = len(indices), max(map(len, windows))
    device = next(policy.parameters()).device
    item = dict(tokens=torch.zeros(length,width,policy.core.input_dim,device=device),
                actions=torch.zeros(length,width,600,device=device))
    for key in ('rewards','returns','dones','starts','valid'):
        item[key] = torch.zeros(length,width,device=device,
            dtype=torch.bool if key in ('dones','starts','valid') else torch.float32)
    targets, masks = [], []
    for lane, (idx, tokens) in enumerate(zip(indices, windows)):
        n = len(tokens); item['tokens'][:n,lane] = torch.from_numpy(tokens).to(device)
        item['valid'][:n,lane] = True; item['starts'][:n,lane] = torch.from_numpy(tokens[:,-1].astype(bool)).to(device)
        ep, q = corpus.episode(idx)
        targets.append(native_flow_targets(codec.encode(ep['teacher_actions'][q]),codec,stats['actions']))
        masks.append(ep['action_masks'][q])
    item['query_rel'] = torch.tensor([len(w)-1 for w in windows],device=device)
    item['query_indices'] = indices
    item['target_actions'] = torch.from_numpy(np.stack(targets)).to(device)
    item['action_mask'] = torch.from_numpy(np.stack(masks)).to(device)
    item['noise'] = torch.randn(item['target_actions'].shape,device=device,generator=generator)
    item['timesteps'] = torch.rand(width,device=device,generator=generator)
    item['weights'] = torch.ones(width,device=device)
    return item


class Queries:
    def __init__(self, corpus, native, motor):
        self.corpus, self.native, self.motor = corpus, native, motor

    def observation(self, indices):
        selected = [self.corpus.episode(i) for i in indices]
        images = np.stack([e['images'][q] for e,q in selected])
        return dict(main_images=images[:,0],wrist_images=images[:,1],extra_view_images=images[:,2,None],
                    states=np.stack([e['states'][q] for e,q in selected]),
                    task_descriptions=[e['meta']['prompt'] for e,q in selected])

    def prepare(self, indices):
        with torch.no_grad():
            _, prepared = self.motor.prepare_batch(self.observation(indices))
        return prepared

    def __call__(self, indices):
        try:
            prepared = self.prepare(indices)
            _, mask, cache = prepared['prefix']
            return prepared['observation'], cache, mask
        finally:
            self.native._prepared = None


@torch.no_grad()
def commands(native, policy, prepared, context, noise):
    hidden, mask, cache = prepared['prefix']
    edited = policy.kv_adapter(cache, mask, context)
    original = native.model.build_prefix_cache
    try:
        native.model.build_prefix_cache = lambda *a, **kw: (hidden,mask,edited)
        result,_ = native.model.predict_action_batch(prepared['observation'],mode='eval',
            noise=torch.as_tensor(noise,device=native.model.device))
        return DirectActionCodec.mapped_physical(result.detach().float().cpu().numpy())
    finally:
        native.model.build_prefix_cache = original
        native._prepared = None


class Student:
    def __init__(self, native, policy, mode='full'):
        self.native, self.policy, self.mode = native, policy.eval(), mode
        self.privileged = mode == 'privileged'
        self.reset()

    def reset(self):
        self.state = None
        self.previous = np.zeros(602,np.float32)

    def set_teacher_condition(self, offset, active):
        assert self.privileged
        self.condition = np.asarray(offset,np.float32)*active/.04

    @torch.no_grad()
    def consume(self, feature, start):
        if self.mode in ('current','privileged'):
            self.reset()
        elif self.mode == 'cleared' and start:
            self.reset()
        token = np.concatenate((feature, self.previous, [start])).astype(np.float32)
        token = torch.from_numpy(token[None]).to(next(self.policy.parameters()).device)
        context,self.state = self.policy.encode_step(token,self.state)
        return context

    def prime(self, episodes):
        assert not self.privileged
        for episode in episodes:
            with np.load(Path(episode)/'episode.npz') as e:
                for q,feature in enumerate(e['features']):
                    self.consume(feature, q == 0)
                    self.feedback(e['feedback'][q],e['rewards'][q],e['dones'][q])

    def predict(self, feature, prepared, noise, *, start):
        if self.privileged:
            feature = np.concatenate((feature, self.condition))
        return commands(self.native,self.policy,prepared,self.consume(feature,start),noise[None])[0]

    def feedback(self, action, reward, terminal):
        self.previous = np.concatenate((action,[reward,terminal])).astype(np.float32)
