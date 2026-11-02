"""Share native VLA forwards while each case owns its serial history protocol.

Only the broker's caller accesses the stateful native prefix port. Case
threads keep independent functional student caches and publish actual outcomes
through the existing single-case protocol. No query or physical replay is added.
"""
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
import threading
import time

import numpy as np

from .robocasa_residual import combine_observations


@dataclass
class _Call:
    observation: dict
    owner: str
    features: Future = field(default_factory=Future)
    control: Future = field(default_factory=Future)
    commands: Future = field(default_factory=Future)
    execution_provenance: dict | None = None


class BatchedMotorBroker:
    def __init__(self, motor, spec, *, maximum_batch, deadline_unix=None):
        if type(maximum_batch) is not int or maximum_batch < 1:
            raise ValueError("Require a qualified positive native execution batch")
        self.motor, self.spec, self.maximum_batch = motor, spec, maximum_batch
        self.deadline, self.failure = deadline_unix, None
        self.queue, self.lock, self.calls = Queue(), threading.RLock(), {}
        self.batch_sizes, self.sequence = [], 0

    def check(self):
        with self.lock:
            if self.deadline is not None and time.time() >= self.deadline and self.failure is None:
                self.abort(TimeoutError("Operational readout checkpoint; retain complete and partial attempts"))
            if self.failure is not None:
                raise self.failure

    def abort(self, error):
        with self.lock:
            if self.failure is None:
                self.failure = error
            for call in self.calls.values():
                for future in (call.features, call.control, call.commands):
                    if not future.done():
                        future.set_exception(self.failure)
            self.calls.clear()

    def port(self, owner):
        broker = self

        class SerialPort:
            def prepare(self, observation):
                call = _Call(observation, owner)
                with broker.lock:
                    broker.check()
                    broker.calls[id(call)] = call
                    broker.queue.put(call)
                return call.features.result(), call

            def execute(self, prepared, *, noise, steering, spec):
                with broker.lock:
                    broker.check()
                    if (spec.record() != broker.spec.record() or prepared.owner != owner
                            or id(prepared) not in broker.calls or prepared.control.done()):
                        raise ValueError("Use only this case's unconsumed native prefix")
                    prepared.control.set_result((np.asarray(noise), np.asarray(steering)))
                return prepared.commands.result()

        return SerialPort()

    def service_next(self):
        """Run one ready native batch; False means no case is ready yet."""
        self.check()
        try:
            calls = [self.queue.get(timeout=.02)]
        except Empty:
            return False
        # A short rendezvous coalesces ready cases without waiting for slow
        # physical resets. It changes execution only, never a source batch.
        until = time.monotonic() + .001
        while len(calls) < self.maximum_batch:
            try:
                calls.append(self.queue.get(timeout=max(0., until - time.monotonic())))
            except Empty:
                break
        self.sequence += 1
        features, prepared = self.motor.prepare_batch(combine_observations([call.observation for call in calls]))
        if len(features) != len(calls):
            raise ValueError("The native prefix batch must preserve every case row")
        with self.lock:
            self.check()
            owners = [call.owner for call in calls]
            for row, call in enumerate(calls):
                call.execution_provenance = dict(native_batch=self.sequence, row=row,
                    batch_size=len(calls), case_owners=owners)
                call.features.set_result(np.asarray(features[row], np.float32))
        values = []
        for call in calls:
            while not call.control.done():
                self.check()
                time.sleep(.001)
            values.append(call.control.result())
        self.check()
        if any(noise.shape != (1, 50, 32) or control.shape != (self.spec.steering_dim,)
               for noise, control in values):
            raise ValueError("Preserve native H50 noise and steering dimensions in every case")
        chunks = np.asarray(self.motor.execute_batch(prepared,
            noise=np.concatenate([noise for noise, _ in values]),
            steering=np.stack([control for _, control in values]), spec=self.spec))
        if chunks.shape != (len(calls), 50, 12) or not np.isfinite(chunks).all():
            raise ValueError("Native execution must return one finite H50 chunk per case")
        with self.lock:
            self.check()
            for call, commands in zip(calls, chunks, strict=True):
                call.commands.set_result(commands.copy())
                del self.calls[id(call)]
        self.batch_sizes.append(len(calls))
        return True
