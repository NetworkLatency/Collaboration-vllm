# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.token_routing.peer_protocol import (
    PeerReq,
    PeerSamplingParams,
    finish_reason_to_request_status,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class RouteDecision:
    """Per-request routing decision returned synchronously after a step."""

    handoff: bool = False
    discard_num_tokens: int = 0
    finish_thinking: bool = False
    flushed_token_ids: list[int] | None = None


class BaseTokenRoutingScheduler(Scheduler):
    """Base vLLM V1 scheduler for peer-to-peer multi-model collaborative serving.

    Subclass this scheduler and select it via ``SchedulerConfig.scheduler_cls``.
    Router parameters are passed via ``VllmConfig.additional_config["token_router"]``.

    Transport is decoupled from scheduling logic:
    - EngineCoreProc drains outbound messages via :meth:`take_outgoing_peer_reqs`
      and pushes them to the peer over ZMQ.
    - EngineCoreProc delivers inbound messages via :meth:`receive_peer_messages`
      and flushes any resulting client outputs via
      :meth:`take_pending_peer_client_outputs`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        additional = self.vllm_config.additional_config
        raw_cfg = (
            additional.get("token_router", {}) if isinstance(additional, dict) else {}
        )
        self.router_config: dict[str, Any] = dict(raw_cfg)

        self.model_name: str = str(self.router_config.get("model_name", "slm"))
        self.peer_name: str = str(self.router_config.get("peer_name", "llm"))
        self.is_entry_model: bool = bool(
            self.router_config.get("is_entry_model", self.model_name == "slm")
        )

        # Shared per-request cross-engine bookkeeping.
        self._kvcache_loc: dict[str, dict[str, int | None]] = {}
        self._origin_prompt_len: dict[str, int] = {}
        # Remember the peer's parked (handoff_seq, base_len) so our reply
        # passes the peer's resume fence check.
        self._peer_return_fence: dict[str, tuple[int, int]] = {}
        # Outbound peer messages produced during update_from_output / abort.
        self._outgoing_peer_reqs: list[PeerReq] = []
        # Immediate client outputs on the entry model generated when a peer
        # returns tokens or finishes a parked request.
        self._pending_peer_client_outputs: dict[int, list[EngineCoreOutput]] = (
            defaultdict(list)
        )
        # Lazily resolved tokenizer and EOS token ID.
        self._tokenizer: Any | None = None
        self._eos_token_id: int | None = None

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from vllm.tokenizers import cached_tokenizer_from_config

            self._tokenizer = cached_tokenizer_from_config(
                self.vllm_config.model_config
            )
            self._eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        return self._tokenizer

    def _get_eos_token_id(self) -> int | None:
        if self._eos_token_id is None:
            # Prefer hf_config.eos_token_id to avoid loading tokenizer unless needed.
            hf_eos = getattr(
                self.vllm_config.model_config.hf_config, "eos_token_id", None
            )
            if isinstance(hf_eos, list):
                self._eos_token_id = int(hf_eos[0]) if hf_eos else None
            elif hf_eos is not None:
                self._eos_token_id = int(hf_eos)
            else:
                self._get_tokenizer()
        return self._eos_token_id

    # ------------------------------------------------------------------
    # Transport-facing queues (consumed by EngineCoreProc in Stage 3)
    # ------------------------------------------------------------------

    def take_outgoing_peer_reqs(self) -> list[PeerReq]:
        reqs = self._outgoing_peer_reqs
        self._outgoing_peer_reqs = []
        return reqs

    def take_pending_peer_client_outputs(self) -> dict[int, EngineCoreOutputs]:
        if not self._pending_peer_client_outputs:
            return {}
        packaged = {
            client_idx: EngineCoreOutputs(outputs=outs)
            for client_idx, outs in self._pending_peer_client_outputs.items()
            if outs
        }
        self._pending_peer_client_outputs.clear()
        return packaged

    # ------------------------------------------------------------------
    # Request lifecycle hooks
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        rid = request.request_id
        if rid not in self._kvcache_loc:
            self._kvcache_loc[rid] = {
                self.model_name: 0,
                self.peer_name: None,
            }
            self._origin_prompt_len[rid] = request.num_prompt_tokens
        if not self.is_entry_model:
            request.peer_suppress_output = True
        super().add_request(request)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        finished = super().finish_requests(request_ids, finished_status)
        for req in finished:
            rid = req.request_id
            if (
                finished_status == RequestStatus.FINISHED_ABORTED
                and self._kvcache_loc.get(rid, {}).get(self.peer_name) is not None
            ):
                fence_seq, fence_base = self._peer_return_fence.get(rid, (0, 0))
                self._outgoing_peer_reqs.append(
                    PeerReq(
                        rid=rid,
                        status="abort",
                        source_model=self.model_name,
                        destination_model=self.peer_name,
                        target_handoff_seq=fence_seq,
                        target_base_len=fence_base,
                    )
                )
            self._cleanup_routing_state(rid)
        return finished

    def _cleanup_routing_state(self, request_id: str) -> None:
        self._kvcache_loc.pop(request_id, None)
        self._origin_prompt_len.pop(request_id, None)
        self._peer_return_fence.pop(request_id, None)

    # ------------------------------------------------------------------
    # Inbound peer message handling
    # ------------------------------------------------------------------

    def receive_peer_messages(self, msgs: Iterable[PeerReq]) -> None:
        """Apply incoming PeerReq messages before the next scheduling step."""
        for msg in msgs:
            if msg.status == "abort":
                self._handle_peer_abort(msg)
            elif msg.status == "finished":
                self._handle_peer_finished(msg)
            elif msg.status == "need":
                self._handle_peer_need(msg)
            else:
                raise ValueError(f"Unknown PeerReq status: {msg.status!r}")

    def _handle_peer_abort(self, msg: PeerReq) -> None:
        if msg.rid in self.requests:
            super().finish_requests(msg.rid, RequestStatus.FINISHED_ABORTED)
        self._cleanup_routing_state(msg.rid)

    def _handle_peer_finished(self, msg: PeerReq) -> None:
        rid = msg.rid
        finished_status = finish_reason_to_request_status(msg.finish_reason)
        result = self.finish_parked_request(
            rid,
            handoff_seq=msg.target_handoff_seq or None,
            token_ids=msg.new_token_ids,
            finished_status=finished_status,
            stop_reason=msg.stop_reason,
        )
        if result is not None and self.is_entry_model:
            client_idx, core_output = result
            self._pending_peer_client_outputs[client_idx].append(core_output)
        self._cleanup_routing_state(rid)

    def _handle_peer_need(self, msg: PeerReq) -> None:
        rid = msg.rid
        existing = self.requests.get(rid)

        if existing is None:
            # First-time arrival at peer engine (typically SLM -> LLM).
            if msg.sampling_params is None:
                logger.warning(
                    "Ignoring first-time PeerReq for unknown rid=%s without "
                    "sampling_params",
                    rid,
                )
                return
            self._kvcache_loc[rid] = dict(msg.kvcache_loc)
            self._kvcache_loc[rid][self.model_name] = len(msg.new_token_ids)
            self._origin_prompt_len[rid] = (
                msg.origin_prompt_len
                if msg.origin_prompt_len is not None
                else len(msg.new_token_ids)
            )
            self._peer_return_fence[rid] = (
                msg.sender_handoff_seq,
                msg.sender_base_len,
            )
            sp = msg.sampling_params.to_sampling_params(
                eos_token_id=self._get_eos_token_id(),
                logprobs=self._peer_request_logprobs(),
            )
            new_req = Request(
                request_id=rid,
                prompt_token_ids=list(msg.new_token_ids),
                sampling_params=sp,
                pooling_params=None,
                block_hasher=self.block_hasher,
            )
            self.add_request(new_req)
            self.on_peer_need_accepted(new_req, msg, is_first_arrival=True)
            return

        if rid not in self.peer_waiting:
            # Stale or duplicate message for a request that is no longer parked.
            return

        # Guard: if peer tokens already hit terminal stop/length on this engine,
        # finalize cleanly instead of failing nonterminal resume checks.
        if self._peer_suffix_is_terminal(existing, msg.new_token_ids):
            self._handle_peer_finished(msg)
            return

        resumed = self.resume_requests(
            [
                {
                    "request_id": rid,
                    "handoff_seq": msg.target_handoff_seq,
                    "base_len": msg.target_base_len,
                    "token_ids": msg.new_token_ids,
                }
            ]
        )
        if not resumed:
            return

        self._kvcache_loc[rid] = dict(msg.kvcache_loc)
        self._kvcache_loc[rid][self.model_name] = existing.num_tokens
        self._peer_return_fence[rid] = (
            msg.sender_handoff_seq,
            msg.sender_base_len,
        )

        # On the entry model, emit LLM's accepted suffix to the client in order.
        if self.is_entry_model and msg.new_token_ids:
            self._pending_peer_client_outputs[existing.client_index].append(
                EngineCoreOutput(
                    request_id=rid,
                    new_token_ids=list(msg.new_token_ids),
                    trace_headers=existing.trace_headers,
                )
            )

        self.on_peer_need_accepted(existing, msg, is_first_arrival=False)

    def _peer_suffix_is_terminal(self, request: Request, token_ids: list[int]) -> bool:
        if (
            request.num_tokens + len(token_ids) >= self.max_model_len
            or request.num_output_tokens + len(token_ids) >= request.max_tokens
        ):
            return True
        sp = request.sampling_params
        if sp is None:
            return False
        stop_ids = sp.stop_token_ids or ()
        return any(tid == sp.eos_token_id or tid in stop_ids for tid in token_ids)

    def _peer_request_logprobs(self) -> int | None:
        return None

    def on_peer_need_accepted(
        self,
        request: Request,
        msg: PeerReq,
        *,
        is_first_arrival: bool,
    ) -> None:
        """Subclass hook called after a peer ``need`` message is accepted."""

    # ------------------------------------------------------------------
    # Synchronous post-step routing hook
    # ------------------------------------------------------------------

    @abstractmethod
    def route_request(
        self,
        request: Request,
        generated_token_ids: list[int],
        req_index: int,
        model_runner_output: ModelRunnerOutput,
    ) -> RouteDecision:
        """Decide whether a running request should hand off after a step."""

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # Snapshot requests that produced tokens before parent removes finished ones.
        req_id_to_index = model_runner_output.req_id_to_index
        sampled_token_ids = model_runner_output.sampled_token_ids
        step_requests: list[tuple[str, Request, list[int], int]] = []
        for req_id in scheduler_output.num_scheduled_tokens:
            req = self.requests.get(req_id)
            if req is None:
                continue
            req_idx = req_id_to_index.get(req_id)
            if req_idx is None:
                continue
            gen_ids = sampled_token_ids[req_idx]
            if gen_ids:
                step_requests.append((req_id, req, gen_ids, req_idx))

        engine_core_outputs = super().update_from_output(
            scheduler_output, model_runner_output
        )

        # Merge any queued peer-originated client outputs on the entry model.
        if self.is_entry_model and self._pending_peer_client_outputs:
            for client_idx, pending_outs in self._pending_peer_client_outputs.items():
                if not pending_outs:
                    continue
                if client_idx in engine_core_outputs:
                    engine_core_outputs[client_idx].outputs = (
                        pending_outs + engine_core_outputs[client_idx].outputs
                    )
                else:
                    engine_core_outputs[client_idx] = EngineCoreOutputs(
                        outputs=pending_outs
                    )
            self._pending_peer_client_outputs.clear()

        for req_id, req, gen_ids, req_idx in step_requests:
            if req.is_finished():
                self._emit_peer_finished_if_shared(req)
                self._cleanup_routing_state(req_id)
                continue

            if req.status != RequestStatus.RUNNING:
                continue

            decision = self.route_request(req, gen_ids, req_idx, model_runner_output)

            if decision.flushed_token_ids and self.is_entry_model:
                flushed_out = EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=list(decision.flushed_token_ids),
                    trace_headers=req.trace_headers,
                )
                if req.client_index in engine_core_outputs:
                    engine_core_outputs[req.client_index].outputs.append(flushed_out)
                else:
                    engine_core_outputs[req.client_index] = EngineCoreOutputs(
                        outputs=[flushed_out]
                    )

            if decision.handoff:
                self._suspend_and_enqueue_peer_req(req, decision)

        if not self.is_entry_model:
            # Non-entry peer engines never emit generation outputs to frontends.
            return {}

        return engine_core_outputs

    def _emit_peer_finished_if_shared(self, request: Request) -> None:
        rid = request.request_id
        kv_loc = self._kvcache_loc.get(rid)
        if not kv_loc or kv_loc.get(self.peer_name) is None:
            return
        peer_known_len = kv_loc[self.peer_name]
        assert peer_known_len is not None
        delta_tokens = list(request.all_token_ids[peer_known_len:])
        fence_seq, fence_base = self._peer_return_fence.get(rid, (0, 0))
        self._outgoing_peer_reqs.append(
            PeerReq(
                rid=rid,
                status="finished",
                source_model=self.model_name,
                destination_model=self.peer_name,
                new_token_ids=delta_tokens,
                kvcache_loc=dict(kv_loc),
                target_handoff_seq=fence_seq,
                target_base_len=fence_base,
                finish_reason=request.get_finished_reason(),
                stop_reason=request.stop_reason,
            )
        )

    def _suspend_and_enqueue_peer_req(
        self,
        request: Request,
        decision: RouteDecision,
    ) -> None:
        rid = request.request_id
        parked = self.suspend_requests(
            [rid], discard_num_tokens=decision.discard_num_tokens
        )[0]

        kv_loc = self._kvcache_loc.setdefault(
            rid, {self.model_name: 0, self.peer_name: None}
        )
        kv_loc[self.model_name] = parked["base_len"]

        peer_known_len = kv_loc.get(self.peer_name)
        first_send = peer_known_len is None
        if first_send:
            new_token_ids = list(request.all_token_ids)
            assert request.sampling_params is not None
            remaining_max_tokens = request.max_tokens - request.num_output_tokens
            peer_sp = PeerSamplingParams.from_sampling_params(
                request.sampling_params,
                max_tokens=remaining_max_tokens,
            )
            origin_prompt_len = self._origin_prompt_len.get(
                rid, request.num_prompt_tokens
            )
            target_seq, target_base = 0, 0
        else:
            assert peer_known_len is not None
            new_token_ids = list(request.all_token_ids[peer_known_len:])
            peer_sp = None
            origin_prompt_len = None
            target_seq, target_base = self._peer_return_fence.get(rid, (0, 0))

        self._outgoing_peer_reqs.append(
            PeerReq(
                rid=rid,
                status="need",
                source_model=self.model_name,
                destination_model=self.peer_name,
                new_token_ids=new_token_ids,
                kvcache_loc=dict(kv_loc),
                sender_handoff_seq=parked["handoff_seq"],
                sender_base_len=parked["base_len"],
                target_handoff_seq=target_seq,
                target_base_len=target_base,
                origin_prompt_len=origin_prompt_len,
                sampling_params=peer_sp,
                finish_thinking=decision.finish_thinking,
            )
        )
