# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Device-side extension boundary for externally generated proposal distributions."""

from typing import Protocol

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class ExternalSpeculationIO(Protocol):
    """Worker-installed IO, called on the Runner thread without remote waits.

    The implementation owns transport registrations and asynchronous transfers.
    It must stage completed reads before candidate admission and fence all local
    consumers before releasing storage. Request removal invalidates request-bound
    state but does not by itself establish completion of exported DMA buffers.
    """

    capture_proposals: bool

    def add_request(self, req_id: str, params: SamplingParams) -> None:
        """Bind request metadata before its first sample, including after preemption."""
        ...

    def remove_request(self, req_id: str) -> None:
        """Invalidate local request state without prematurely releasing exports."""
        ...

    def on_sample(self, batch: InputBatch, output: SamplerOutput) -> None:
        """Observe actual samples and optional processed logits on the compute stream.

        Partial-prefill rows can have zero valid samples. Capture implementations
        must not export them as proposal positions or block on remote consumers.
        """
        ...

    def draft_logits(
        self, batch: InputBatch, temperatures: torch.Tensor
    ) -> torch.Tensor | None:
        """Return ready logits indexed by persistent request slot and draft position.

        The existing verifier divides these logits by Target temperature. An
        implementation transporting log(q) must therefore return temperature *
        log(q), where q is the actual proposal distribution including its own
        sampling transformations. Greedy Target rows ignore this distribution.
        Returned storage must survive the verification kernels on this stream.
        """
        ...
