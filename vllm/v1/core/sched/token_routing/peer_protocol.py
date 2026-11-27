# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Literal

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

PeerStatus = Literal["need", "finished", "abort"]


@dataclass
class PeerSamplingParams:
    """Minimal serializable subset of SamplingParams for first-time handoff."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    max_tokens: int = 16
    ignore_eos: bool = False
    stop_token_ids: list[int] | None = None
    seed: int | None = None

    @classmethod
    def from_sampling_params(
        cls,
        sp: SamplingParams,
        *,
        max_tokens: int,
    ) -> "PeerSamplingParams":
        return cls(
            temperature=sp.temperature,
            top_p=sp.top_p,
            top_k=sp.top_k,
            min_p=sp.min_p,
            max_tokens=max(1, max_tokens),
            ignore_eos=sp.ignore_eos,
            stop_token_ids=(
                list(sp.stop_token_ids) if sp.stop_token_ids is not None else None
            ),
            seed=sp.seed,
        )

    def to_sampling_params(
        self,
        *,
        eos_token_id: int | None = None,
        logprobs: int | None = None,
    ) -> SamplingParams:
        sp = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            ignore_eos=self.ignore_eos,
            stop_token_ids=(
                list(self.stop_token_ids) if self.stop_token_ids is not None else None
            ),
            seed=self.seed,
            logprobs=logprobs,
        )
        if eos_token_id is not None and not self.ignore_eos:
            sp.eos_token_id = eos_token_id
        return sp


@dataclass
class PeerReq:
    """Cross-engine handoff message exchanged between cooperative schedulers.

    Attributes:
        rid: Request ID shared across both engines.
        status: ``"need"`` (handoff control + delta tokens), ``"finished"``
            (peer reached terminal state), or ``"abort"`` (release parked KV).
        source_model: Logical name of the sending engine (e.g. ``"slm"``).
        destination_model: Logical name of the receiving engine (e.g. ``"llm"``).
        new_token_ids: Full token sequence on first send to a peer; delta token
            suffix beyond ``kvcache_loc[destination_model]`` on later sends.
        kvcache_loc: Mapping ``model_name -> known_total_len | None``.
        sender_handoff_seq: Sender's local ``peer_handoff_seq`` when parking;
            echoed back as ``target_handoff_seq`` on the peer's reply.
        sender_base_len: Sender's retained ``peer_base_len`` when parking;
            echoed back as ``target_base_len`` on the peer's reply.
        target_handoff_seq: Receiver's ``peer_handoff_seq`` being resumed.
        target_base_len: Receiver's ``peer_base_len`` being resumed.
        origin_prompt_len: Original user prompt length (set on first send).
        sampling_params: Minimal sampling config (set on first send).
        finish_thinking: When True, LLM owns the remaining final answer phase
            until completion without handing back on step delimiters.
        finish_reason: Terminal reason when ``status == "finished"``.
        stop_reason: Stop token/string when ``status == "finished"``.

    """

    rid: str
    status: PeerStatus
    source_model: str
    destination_model: str
    new_token_ids: list[int] = field(default_factory=list)
    kvcache_loc: dict[str, int | None] = field(default_factory=dict)
    sender_handoff_seq: int = 0
    sender_base_len: int = 0
    target_handoff_seq: int = 0
    target_base_len: int = 0
    origin_prompt_len: int | None = None
    sampling_params: PeerSamplingParams | None = None
    finish_thinking: bool = False
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None


def finish_reason_to_request_status(
    finish_reason: FinishReason | None,
) -> RequestStatus:
    if finish_reason == FinishReason.LENGTH:
        return RequestStatus.FINISHED_LENGTH_CAPPED
    if finish_reason == FinishReason.ABORT:
        return RequestStatus.FINISHED_ABORTED
    if finish_reason == FinishReason.ERROR:
        return RequestStatus.FINISHED_ERROR
    return RequestStatus.FINISHED_STOPPED
