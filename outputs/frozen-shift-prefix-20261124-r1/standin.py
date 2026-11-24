"""Tiny PyTorch stand-in for the frozen policy: same prefix interface, flow loss and Euler sampler."""
import hashlib

import numpy as np
import torch

import prefix as px

WIDTH, TEXT, IMAGE, HORIZON, DIM = 16, 12, 4, 10, 7


def cast(value, like):
    return torch.as_tensor(value).to(like.dtype)


class Tiny(torch.nn.Module):
    action_horizon, action_dim = HORIZON, DIM

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.image = torch.nn.Linear(3, WIDTH)
        self.embed = torch.nn.Embedding(64, WIDTH)
        self.attend = torch.nn.MultiheadAttention(WIDTH, 2, batch_first=True)
        self.inp = torch.nn.Linear(DIM + 1, WIDTH)
        self.out = torch.nn.Linear(WIDTH, DIM)
        self.requires_grad_(False)

    def embed_prefix(self, obs):
        tokens = torch.cat([self.image(obs["image"]), self.embed(obs["tokens"])], 1)
        mask = torch.cat([torch.ones(len(tokens), IMAGE, dtype=torch.bool), obs["token_mask"]], 1)
        return tokens, mask, torch.zeros(tokens.shape[1], dtype=torch.bool)

    def velocity(self, obs, x, t, prefix=None):
        tokens, mask, ar = self.embed_prefix(obs)
        if prefix is not None:
            tokens, mask, ar = px.inject(torch.cat, cast, tokens, mask, ar, prefix, obs["tokens"].shape[1])
        query = self.inp(torch.cat([x, t.expand(len(x), HORIZON, 1)], -1))
        read, _ = self.attend(query, tokens, tokens, key_padding_mask=~mask)
        return self.out(query + read)

    def compute_loss(self, obs, actions, noise, t, prefix=None):
        x = t * noise + (1 - t) * actions
        return ((self.velocity(obs, x, t, prefix) - (noise - actions)) ** 2).mean()

    def sample_actions(self, obs, noise, prefix=None):
        x = noise
        for step in range(10):
            x = x - .1 * self.velocity(obs, x, torch.tensor(1 - .1 * step), prefix)
        return x


def observation(prompt, state, image, wrist):
    """Prompt characters and binned state share the text tokens, as in pi0.5."""
    ids = [1 + ord(c) % 31 for c in prompt[:TEXT - 4]] + [32 + int(np.clip(8 * v + 16, 0, 31)) for v in state[:4]]
    tokens = torch.zeros(1, TEXT, dtype=torch.long)
    tokens[0, :len(ids)] = torch.tensor(ids)
    mask = torch.zeros(1, TEXT, dtype=torch.bool)
    mask[0, :len(ids)] = True
    pixels = torch.tensor([[np.mean(image) / 255, np.mean(wrist) / 255, float(state[0])]], dtype=torch.float32)
    return dict(image=pixels[:, None].repeat(1, IMAGE, 1) * torch.arange(1, IMAGE + 1)[None, :, None],
                tokens=tokens, token_mask=mask)


class Policy:
    """Shaped like `openpi.policies.policy.Policy`: `infer` packs, calls `_sample_actions`, unpacks."""

    def __init__(self, model=None):
        self._model = model or Tiny()
        self._sample_actions = lambda obs, noise: self._model.sample_actions(obs, noise)
        self.prompts = []

    def infer(self, item, *, noise):
        self.prompts.append(item["prompt"])
        obs = observation(item["prompt"], item["observation/state"], item["observation/image"],
                          item["observation/wrist_image"])
        with torch.no_grad():
            actions = self._sample_actions(obs, torch.as_tensor(noise, dtype=torch.float32)[None])
        return {"actions": actions[0].numpy()}


def sampler(policy, prefix):
    tensors = {key: torch.tensor(value) for key, value in prefix.items()}
    return lambda obs, noise: policy._model.sample_actions(obs, noise, tensors)


class Engine:
    """The interface `fit.py` expects, on the stand-in."""

    def __init__(self, model):
        self.model = model

    def digest(self):
        return {name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                for name, value in self.model.state_dict().items()}

    def reference(self, row):
        tokens, mask, _ = self.model.embed_prefix(row["obs"])
        gate = mask[:, -TEXT:, None].float()
        return ((tokens[:, -TEXT:] * gate).sum((0, 1)) / gate.sum()).numpy()

    def draw(self, row, draw):
        word = hashlib.sha256(repr((row["episode"], row["t"], *draw)).encode()).digest()
        g = torch.Generator().manual_seed(int.from_bytes(word[:4], "little"))
        return torch.randn(1, HORIZON, DIM, generator=g), torch.rand((), generator=g) * .998 + .001

    def value_and_grad(self, prefix, row, draw):
        leaves = {key: torch.tensor(value, requires_grad=True) for key, value in prefix.items()}
        loss = self.model.compute_loss(row["obs"], row["actions"], *self.draw(row, draw), leaves)
        grads = torch.autograd.grad(loss, list(leaves.values()))
        return float(loss.detach()), {key: g.numpy() for key, g in zip(leaves, grads)}

    def loss(self, prefix, row, draw):
        tensors = None if prefix is None else {key: torch.tensor(value) for key, value in prefix.items()}
        with torch.no_grad():
            return float(self.model.compute_loss(row["obs"], row["actions"], *self.draw(row, draw), tensors))


def rows(count, episode, seed):
    """Actions carry an offset the unmodified stand-in was never built to produce."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(count):
        state = rng.uniform(-1, 1, 8)
        obs = observation("open the drawer", state, rng.integers(0, 255, (8, 8, 3)), rng.integers(0, 255, (8, 8, 3)))
        actions = torch.tensor(.5 + .2 * state[:7], dtype=torch.float32).expand(1, HORIZON, DIM).clone()
        out.append(dict(episode=episode, t=t, obs=obs, actions=actions))
    return out
