<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-->

# External candidates in Model Runner V2

This experimental engine mode accepts ready greedy candidates from an async
engine client. It does not load or contact a Draft model. Candidate generation,
role placement and network transport belong to the caller. No remote wait occurs
inside the Model Runner.

## Supported boundary

Configure `speculative_config={"method": "external", "num_speculative_tokens": 3}`
with Model Runner V2, `enforce_eager=True`, `async_scheduling=False`, one worker,
and `stream_interval=1`. This implementation supports text generation with
`temperature=0`, `n=1`, and cumulative or delta output. Structured output,
streaming input, multimodal input, KV/EC transfer, CUDA graphs, distributed
execution, stochastic proposals and asynchronous local scheduling are not enabled.
The ordinary local-speculation path is unchanged.

Disabling asynchronous *local scheduling* does not make the engine wait for a
remote Draft: each request waits independently, and other requests remain
schedulable. Once only external waiters remain and finished-request cleanup has
run, EngineCore sleeps on its existing input queue.

## Client contract

`RequestOutput.external_draft_request` is either an immutable
`ExternalDraftRequest(request_id, generation)` ticket or `None`. The request ID
is the engine's internal identity, not the reusable user request ID. A ticket is
published only after the existing detokenizer and stop checks. Terminal outputs
never authorize another round.

An `AsyncLLM` client passes the ticket unchanged to:

```python
accepted = await engine.submit_external_draft_tokens(ticket, token_ids)
```

`True` means the ready candidates were admitted; verification has not completed.
`False` means the ticket is obsolete, already consumed, or its request ended.
Do not retry it. An empty list explicitly requests one target-only step. A list
longer than the configured candidate budget or containing invalid vocabulary IDs
is rejected. Candidate token IDs must use Target token semantics, and the caller
must generate them from the confirmed prefix associated with this ticket.

Consume every nonterminal output. `FINAL_ONLY` and output intervals above one
would hide tickets and deadlock this protocol, so they are rejected. On caller
failure, use the existing request cancellation lifecycle; no automatic timeout
or Draft fallback is introduced.

The first prefill produces a normal Target token. The client uses that confirmed
prefix to prepare a Draft round. Subsequent admitted candidates use the existing
MRV2 greedy rejection sampler. A scheduler budget or output limit can truncate a
round; synchronize Draft against the actual emitted token delta, not the number
of candidates submitted. The client owns how its Draft cache is aligned.

## Ownership and execution

1. Scheduler increments the per-request generation and marks the request waiting
   after producing nonterminal tokens. Its normal output carries the generation.
2. OutputProcessor attaches the ticket after stop handling. Existing public
   output aggregation retains the latest ticket.
3. AsyncLLM submits through the existing EngineCore utility channel. Processing
   that input wakes the idle engine and checks the generation on the owning loop.
4. Scheduler admits the ready token list and accounts for verification work using
   its existing speculative token budget and KV allocation.
5. After request-state updates, MRV2 resolves current request slots and fills
   `req_states.draft_tokens` before `combine_sampled_and_draft_tokens` builds input.
   It does not construct a local speculator or read back fictitious local drafts.
6. Existing verification, output truncation and cleanup complete the round.
   Preemption invalidates the ticket, clears candidates and recomputes context;
   only a later output grants a new ticket. Abort removes the live request.

`Scheduler.has_requests()` still describes lifecycle ownership, including live
waiters and delayed cleanup. `has_schedulable_requests()` separately tells the
engine whether another step is useful. Existing schedulers retain their previous
behavior through the interface default.

## What this extension does not establish

This is an engine API, not an OpenAI serving endpoint or a disaggregated service.
No RPC server for Draft roles, routing plan, session migration, tensor transport,
or model/tokenizer compatibility negotiation is included. There are no Foretoken
imports or Mooncake-specific fields in vLLM.

Deterministic proposals need only token IDs: the existing verifier treats their
proposal distribution as a point mass. A later stochastic-proposal extension must
carry the actual distribution and integrate Worker-owned device materialization;
a token-list API cannot silently claim that support. The current path does not
provide general transport callbacks, and large tensor readiness is not inferred
from control-message arrival.

Validation must distinguish engine semantics from full DT service behavior:
exercise full/partial acceptance, rejection, length/stop handling, interleaved
requests, stale/duplicate tickets, cancellation, and preemption. Compare output
with target-only steps under identical model state. Random-weight GPU models can
validate the control and verification path, but cannot establish model quality,
real Draft acceptance rates, or end-to-end speedup.
