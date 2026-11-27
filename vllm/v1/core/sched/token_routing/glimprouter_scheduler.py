# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.token_routing.base_routing_scheduler import (
    BaseTokenRoutingScheduler,
    RouteDecision,
)
from vllm.v1.core.sched.token_routing.peer_protocol import PeerReq
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput
from vllm.v1.request import Request


def _compute_topk_entropy(
    logprobs_lists: LogprobsLists | None,
    req_index: int,
    top_k: int,
) -> float:
    """Compute normalized Shannon entropy from vLLM's top-k logprobs slice.

    In vLLM's ``Sampler.gather_logprobs``, column 0 is the sampled token and
    columns ``1 : 1 + top_k`` are the top-k logprobs.
    """
    if logprobs_lists is None:
        raise ValueError(
            "GlimpRouterSLMScheduler requires ModelRunnerOutput.logprobs; "
            "ensure sampling_params.logprobs is enabled."
        )
    sliced = logprobs_lists.slice_request(req_index, 1)
    row = sliced.logprobs[0]
    # Use columns 1:1+top_k when available to avoid double-counting column 0.
    vals = row[1 : 1 + top_k] if len(row) > 1 else row
    probs = [math.exp(float(lp)) for lp in vals if math.isfinite(float(lp))]
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for p in probs:
        norm_p = p / total
        if norm_p > 0.0:
            entropy -= norm_p * math.log(norm_p)
    return entropy


class GlimpRouterSLMScheduler(BaseTokenRoutingScheduler):
    """SLM (entry-model) scheduler implementing step-level GlimpRouter routing.

    Config keys under ``vllm_config.additional_config["token_router"]``:
    - ``threshold`` (float, default 0.5): entropy threshold to hand off to LLM.
    - ``delimiter_str`` (str, default ``"\\n\\n"``): step boundary delimiter.
    - ``finished_str`` (str, default ``"</think>"``): thinking-end marker; once
      emitted, the final answer phase is handed off to LLM until completion.
    - ``probe_tokens`` (int, default 1): fixed probe length ``n >= 1``.
    - ``probe_agg`` (str, default ``"mean"``): ``"mean"`` or ``"max"`` entropy
      aggregation across the ``n`` probe tokens.
    - ``top_logprobs`` (int, default 20): top-k logprobs requested for entropy.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.threshold: float = float(self.router_config.get("threshold", 0.5))
        self.delimiter_str: str = str(self.router_config.get("delimiter_str", "\n\n"))
        self.finished_str: str = str(self.router_config.get("finished_str", "</think>"))
        self.probe_tokens: int = int(self.router_config.get("probe_tokens", 1))
        if self.probe_tokens < 1:
            raise ValueError(f"probe_tokens must be >= 1, got {self.probe_tokens}")
        self.probe_agg: str = str(self.router_config.get("probe_agg", "mean"))
        if self.probe_agg not in ("mean", "max"):
            raise ValueError(
                f"probe_agg must be 'mean' or 'max', got {self.probe_agg!r}"
            )
        self.top_logprobs: int = int(self.router_config.get("top_logprobs", 20))
        self._tail_window_tokens: int = int(
            self.router_config.get("tail_window_tokens", 8)
        )

        # Requests currently inside an n-token probe window: rid -> entropies list.
        self._probe_entropies: dict[str, list[float]] = {}
        # Total token length at the start of the current SLM step so delimiter
        # checks never re-trigger on a previous step's "\n\n".
        self._step_start_len: dict[str, int] = {}
        # Original user-requested logprobs setting so internal logprobs can be
        # stripped before returning outputs to the client.
        self._user_logprobs: dict[str, int | None] = {}

    def _enter_probe(self, request: Request) -> None:
        self._probe_entropies[request.request_id] = []
        self._step_start_len[request.request_id] = request.num_tokens
        request.peer_suppress_output = True

    def add_request(self, request: Request) -> None:
        rid = request.request_id
        sp = request.sampling_params
        if sp is not None:
            self._user_logprobs[rid] = sp.logprobs
            if sp.logprobs is None or sp.logprobs < self.top_logprobs:
                sp.logprobs = self.top_logprobs
        self._enter_probe(request)
        super().add_request(request)

    def on_peer_need_accepted(
        self,
        request: Request,
        msg: PeerReq,
        *,
        is_first_arrival: bool,
    ) -> None:
        # After LLM finishes a step and hands back, SLM probes the next step.
        self._enter_probe(request)

    def _cleanup_routing_state(self, request_id: str) -> None:
        super()._cleanup_routing_state(request_id)
        self._probe_entropies.pop(request_id, None)
        self._step_start_len.pop(request_id, None)
        self._user_logprobs.pop(request_id, None)

    def _decode_step_tail(self, request: Request) -> str:
        start_len = self._step_start_len.get(request.request_id, request.num_tokens)
        step_tokens = list(request.all_token_ids[start_len:])
        if not step_tokens:
            return ""
        tokenizer = self._get_tokenizer()
        tail = step_tokens[-self._tail_window_tokens :]
        return tokenizer.decode(tail, skip_special_tokens=False)

    def route_request(
        self,
        request: Request,
        generated_token_ids: list[int],
        req_index: int,
        model_runner_output: ModelRunnerOutput,
    ) -> RouteDecision:
        rid = request.request_id
        entropies = self._probe_entropies.get(rid)

        if entropies is not None:
            # Inside the fixed n-token probe window.
            entropy = _compute_topk_entropy(
                model_runner_output.logprobs, req_index, self.top_logprobs
            )
            entropies.append(entropy)

            # Check whether thinking already ended inside the probe window.
            tail_str = self._decode_step_tail(request)
            if self.finished_str and self.finished_str in tail_str:
                flushed = list(request.peer_unemitted_token_ids)
                request.peer_unemitted_token_ids.clear()
                request.peer_suppress_output = False
                self._probe_entropies.pop(rid, None)
                return RouteDecision(
                    handoff=True,
                    discard_num_tokens=0,
                    finish_thinking=True,
                    flushed_token_ids=flushed,
                )

            if len(entropies) < self.probe_tokens:
                return RouteDecision(handoff=False)

            # Collected all n probe tokens; aggregate and decide.
            score = (
                sum(entropies) / len(entropies)
                if self.probe_agg == "mean"
                else max(entropies)
            )
            num_probed = len(entropies)
            self._probe_entropies.pop(rid, None)

            if score >= self.threshold:
                # Hand off to LLM and roll back all n probe tokens.
                return RouteDecision(
                    handoff=True,
                    discard_num_tokens=num_probed,
                )

            # Probe passed: flush buffered probe tokens to client.
            flushed = list(request.peer_unemitted_token_ids)
            request.peer_unemitted_token_ids.clear()
            request.peer_suppress_output = False
            if self.delimiter_str and self.delimiter_str in tail_str:
                # Edge case: the n probe tokens already closed a step.
                self._enter_probe(request)
            return RouteDecision(
                handoff=False,
                flushed_token_ids=flushed,
            )

        # Normal SLM step generation: watch for </think> or step delimiter.
        tail_str = self._decode_step_tail(request)
        if self.finished_str and self.finished_str in tail_str:
            return RouteDecision(
                handoff=True,
                discard_num_tokens=0,
                finish_thinking=True,
            )
        if self.delimiter_str and self.delimiter_str in tail_str:
            self._enter_probe(request)
        return RouteDecision(handoff=False)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        user_logprobs = dict(self._user_logprobs)
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        # Strip internal entropy logprobs for requests whose user did not ask for them.
        for core_outputs in outputs.values():
            for out in core_outputs.outputs:
                if (
                    out.new_logprobs is not None
                    and user_logprobs.get(out.request_id) is None
                ):
                    out.new_logprobs = None
        return outputs


class GlimpRouterLLMScheduler(BaseTokenRoutingScheduler):
    """LLM (peer-model) scheduler for step-level GlimpRouter generation.

    Generates one reasoning step until ``delimiter_str`` (default ``"\\n\\n"``)
    and hands back all generated tokens to SLM without rollback. Once SLM sends
    ``finish_thinking=True`` (after ``"</think>"``), skips delimiter handbacks
    and generates the final answer straight through to completion.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.delimiter_str: str = str(self.router_config.get("delimiter_str", "\n\n"))
        self._tail_window_tokens: int = int(
            self.router_config.get("tail_window_tokens", 8)
        )
        # Requests in final-answer mode (after </think>): never hand back.
        self._finish_thinking_rids: set[str] = set()
        # Total token length at the start of the current LLM step so delimiter
        # detection only inspects tokens generated by LLM in this turn.
        self._step_start_len: dict[str, int] = {}

    def on_peer_need_accepted(
        self,
        request: Request,
        msg: PeerReq,
        *,
        is_first_arrival: bool,
    ) -> None:
        rid = request.request_id
        if msg.finish_thinking:
            self._finish_thinking_rids.add(rid)
        self._step_start_len[rid] = request.num_tokens

    def _cleanup_routing_state(self, request_id: str) -> None:
        super()._cleanup_routing_state(request_id)
        self._finish_thinking_rids.discard(request_id)
        self._step_start_len.pop(request_id, None)

    def route_request(
        self,
        request: Request,
        generated_token_ids: list[int],
        req_index: int,
        model_runner_output: ModelRunnerOutput,
    ) -> RouteDecision:
        rid = request.request_id
        # Fix over upstream TokenRouter: once </think> has triggered final-answer
        # handover, keep generating on LLM until EOS / max_tokens.
        if rid in self._finish_thinking_rids:
            return RouteDecision(handoff=False)

        start_len = self._step_start_len.get(rid, request.num_tokens)
        new_step_tokens = list(request.all_token_ids[start_len:])
        if not new_step_tokens:
            return RouteDecision(handoff=False)

        tokenizer = self._get_tokenizer()
        tail_str = tokenizer.decode(
            new_step_tokens[-self._tail_window_tokens :],
            skip_special_tokens=False,
        )
        if self.delimiter_str and self.delimiter_str in tail_str:
            self._step_start_len.pop(rid, None)
            return RouteDecision(handoff=True, discard_num_tokens=0)

        return RouteDecision(handoff=False)
