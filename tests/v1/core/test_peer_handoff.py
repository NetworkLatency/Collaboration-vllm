# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.v1.core.utils import EOS_TOKEN_ID, create_requests, create_scheduler
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend


def _complete_step(scheduler, scheduler_output, token_id: int | None = 1000):
    req_ids = list(scheduler_output.num_scheduled_tokens)
    output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[] if token_id is None else [token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output, output)


def _step(scheduler, token_id: int = 1000) -> SchedulerOutput:
    scheduler_output = scheduler.schedule()
    _complete_step(scheduler, scheduler_output, token_id)
    return scheduler_output


def _parked_request():
    scheduler = create_scheduler(use_v2_model_runner=False)
    request = create_requests(num_requests=1, num_tokens=32, max_tokens=64)[0]
    scheduler.add_request(request)
    _step(scheduler)
    assert request.status == RequestStatus.RUNNING
    return scheduler, request


def _resume(parked, token_ids):
    return {
        "request_id": parked["request_id"],
        "handoff_seq": parked["handoff_seq"],
        "base_len": parked["base_len"],
        "token_ids": token_ids,
    }


def test_suspend_parks_request_and_keeps_kv():
    scheduler, request = _parked_request()
    blocks_before = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    computed_before = request.num_computed_tokens

    parked = scheduler.suspend_requests([request.request_id])

    assert request.output_token_ids == []
    assert list(request.all_token_ids) == request.prompt_token_ids
    assert request.num_tokens == computed_before == 32
    assert request.status == RequestStatus.PEER_WAIT
    assert request not in scheduler.running
    assert scheduler.peer_waiting[request.request_id] is request
    assert parked == [
        {
            "request_id": request.request_id,
            "handoff_seq": 1,
            "base_len": request.num_tokens,
            "computed_len": computed_before,
        }
    ]
    # KV blocks stay attached to the parked request.
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id) == blocks_before
    # The request remains live without requiring empty engine steps.
    assert scheduler.has_unfinished_requests()
    assert not scheduler.has_requests()


def test_resume_makes_a_parked_only_scheduler_runnable():
    scheduler, request = _parked_request()
    parked = scheduler.suspend_requests([request.request_id])[0]
    assert scheduler.get_request_counts() == (0, 1)
    assert scheduler.has_unfinished_requests()
    assert not scheduler.has_requests()

    scheduler.resume_requests([_resume(parked, [7, 8])])

    assert scheduler.has_requests()
    output = _step(scheduler, 1001)
    assert output.num_scheduled_tokens == {request.request_id: 2}
    assert request.output_token_ids == [7, 8, 1001]


@pytest.mark.parametrize("pause_state", list(PauseState))
def test_abort_last_parked_request_keeps_worker_cleanup_schedulable(pause_state):
    scheduler, request = _parked_request()
    scheduler.suspend_requests([request.request_id])
    scheduler.set_pause_state(pause_state)
    assert not scheduler.has_requests()

    assert scheduler.finish_requests(
        request.request_id, RequestStatus.FINISHED_ABORTED
    ) == [request]
    assert (
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        == []
    )
    assert scheduler.has_requests()

    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == 0
    assert output.finished_req_ids == {request.request_id}
    _complete_step(scheduler, output, token_id=None)
    assert not scheduler.has_requests()
    assert not any(scheduler.kv_cache_manager.get_block_ids(request.request_id))


def test_pause_new_still_drains_active_request_alongside_parked_request():
    scheduler = create_scheduler(use_v2_model_runner=False)
    parked_request, active_request = create_requests(
        num_requests=2, num_tokens=32, max_tokens=64
    )
    for request in (parked_request, active_request):
        scheduler.add_request(request)
    _step(scheduler)
    scheduler.suspend_requests([parked_request.request_id])
    scheduler.set_pause_state(PauseState.PAUSED_NEW)

    assert scheduler.has_requests()
    output = _step(scheduler, 1001)
    assert output.num_scheduled_tokens == {active_request.request_id: 1}
    assert active_request.output_token_ids == [1000, 1001]
    assert parked_request.status == RequestStatus.PEER_WAIT


def test_parked_request_is_skipped_while_another_request_progresses():
    scheduler = create_scheduler(use_v2_model_runner=False)
    parked_request, active_request = create_requests(
        num_requests=2, num_tokens=32, max_tokens=64
    )
    for request in (parked_request, active_request):
        scheduler.add_request(request)
    _step(scheduler)
    scheduler.suspend_requests([parked_request.request_id])
    parked_blocks = scheduler.kv_cache_manager.get_block_ids(parked_request.request_id)

    scheduler_output = _step(scheduler, token_id=1001)

    assert scheduler_output.num_scheduled_tokens == {active_request.request_id: 1}
    assert parked_request.request_id not in (
        scheduler_output.scheduled_cached_reqs.req_ids
    )
    assert parked_request.output_token_ids == []
    assert parked_request.num_computed_tokens == 32
    assert parked_request.status == RequestStatus.PEER_WAIT
    assert (
        scheduler.kv_cache_manager.get_block_ids(parked_request.request_id)
        == parked_blocks
    )
    assert active_request.output_token_ids == [1000, 1001]
    assert active_request.num_computed_tokens == 33


def test_resume_replaces_local_candidate_and_recomputes_peer_tokens():
    scheduler, request = _parked_request()
    assert request.output_token_ids == [1000]
    prefix = list(request.prompt_token_ids)
    parked = scheduler.suspend_requests([request.request_id])[0]
    assert request.output_token_ids == []
    assert list(request.all_token_ids) == prefix
    assert parked["base_len"] == 32

    resumed = scheduler.resume_requests(
        [
            {
                "request_id": request.request_id,
                "handoff_seq": parked["handoff_seq"],
                "base_len": parked["base_len"],
                "token_ids": [7, 8],
            }
        ]
    )

    assert resumed == [request.request_id]
    assert request.status == RequestStatus.RUNNING
    assert request.output_token_ids == [7, 8]
    assert list(request.all_token_ids) == prefix + [7, 8]
    # Peer tokens are history, not computed KV: this model must still run them.
    assert request.num_computed_tokens == parked["computed_len"]

    scheduler_output = scheduler.schedule()
    cached = scheduler_output.scheduled_cached_reqs
    assert cached.peer_resumed_req_ids == {request.request_id}
    assert cached.all_token_ids[request.request_id] == prefix + [7, 8]
    idx = cached.req_ids.index(request.request_id)
    assert cached.num_computed_tokens[idx] == parked["computed_len"]
    assert scheduler_output.num_scheduled_tokens[request.request_id] == (
        len(request.output_token_ids)
        + len(request.prompt_token_ids)
        - parked["computed_len"]
    )


def test_stale_and_duplicate_resumes_are_ignored():
    scheduler, request = _parked_request()
    parked = scheduler.suspend_requests([request.request_id])[0]
    tokens_before = list(request.output_token_ids)

    stale = [
        {
            "request_id": request.request_id,
            "handoff_seq": parked["handoff_seq"] - 1,
            "base_len": parked["base_len"],
            "token_ids": [7],
        }
    ]
    assert scheduler.resume_requests(stale) == []
    assert request.status == RequestStatus.PEER_WAIT
    assert request.output_token_ids == tokens_before

    valid = [
        {
            "request_id": request.request_id,
            "handoff_seq": parked["handoff_seq"],
            "base_len": parked["base_len"],
            "token_ids": [7],
        }
    ]
    assert scheduler.resume_requests(valid) == [request.request_id]
    assert scheduler.resume_requests(valid) == []
    assert request.output_token_ids == tokens_before + [7]


def test_resume_with_mismatched_base_raises():
    scheduler, request = _parked_request()
    parked = scheduler.suspend_requests([request.request_id])[0]

    with pytest.raises(ValueError, match="does not match parked base"):
        scheduler.resume_requests(
            [
                {
                    "request_id": request.request_id,
                    "handoff_seq": parked["handoff_seq"],
                    "base_len": parked["base_len"] - 1,
                    "token_ids": [7],
                }
            ]
        )


def test_abort_releases_parked_request():
    scheduler, request = _parked_request()
    scheduler.suspend_requests([request.request_id])

    aborted = scheduler.finish_requests(
        request.request_id, RequestStatus.FINISHED_ABORTED
    )

    assert aborted == [request]
    assert request.request_id not in scheduler.peer_waiting
    assert not scheduler.has_unfinished_requests()
    assert not any(scheduler.kv_cache_manager.get_block_ids(request.request_id))


@pytest.mark.parametrize("second_tokens", [[9, 10], []])
def test_consecutive_resumes_deliver_all_peer_tokens_once(second_tokens):
    scheduler, request = _parked_request()
    computed_before = request.num_computed_tokens

    for index, token_ids in enumerate(([7, 8], second_tokens)):
        parked = scheduler.suspend_requests(
            [request.request_id], discard_last_token=index == 0
        )[0]
        resume = {
            "request_id": request.request_id,
            "handoff_seq": parked["handoff_seq"],
            "base_len": parked["base_len"],
            "token_ids": token_ids,
        }
        assert scheduler.resume_requests([resume]) == [request.request_id]
        assert scheduler.resume_requests([resume]) == []
        assert request.num_computed_tokens == computed_before

    expected_output = [7, 8] + second_tokens
    expected_history = list(request.prompt_token_ids) + expected_output
    assert request.output_token_ids == expected_output
    output = _step(scheduler)
    cached = output.scheduled_cached_reqs
    assert cached.peer_resumed_req_ids == {request.request_id}
    assert cached.all_token_ids[request.request_id] == expected_history
    idx = cached.req_ids.index(request.request_id)
    assert cached.num_computed_tokens[idx] == computed_before
    assert output.num_scheduled_tokens[request.request_id] == (
        len(request.prompt_token_ids) + len(expected_output) - computed_before
    )
    assert _step(scheduler).scheduled_cached_reqs.peer_resumed_req_ids == set()


@pytest.mark.parametrize("park_before_abort", [False, True])
def test_abort_discards_peer_tokens_before_request_id_reuse(park_before_abort):
    scheduler, request = _parked_request()
    parked = scheduler.suspend_requests([request.request_id])[0]
    scheduler.resume_requests(
        [
            {
                "request_id": request.request_id,
                "handoff_seq": parked["handoff_seq"],
                "base_len": parked["base_len"],
                "token_ids": [7, 8],
            }
        ]
    )
    if park_before_abort:
        scheduler.suspend_requests([request.request_id], discard_last_token=False)

    assert scheduler.finish_requests(
        request.request_id, RequestStatus.FINISHED_ABORTED
    ) == [request]
    replacement = create_requests(
        num_requests=1, num_tokens=32, max_tokens=64, req_ids=[request.request_id]
    )[0]
    scheduler.add_request(replacement)
    _step(scheduler)

    output = _step(scheduler)
    assert replacement.request_id in output.scheduled_cached_reqs.req_ids
    assert output.scheduled_cached_reqs.peer_resumed_req_ids == set()
    assert replacement.output_token_ids == [1000, 1000]


def test_replacement_preserves_32_generated_tokens_and_invalidates_tail_hash():
    scheduler = create_scheduler(use_v2_model_runner=False)
    request = create_requests(num_requests=1, num_tokens=15, max_tokens=64)[0]
    scheduler.add_request(request)
    accepted = list(range(1000, 1032))
    for token_id in accepted + [2000]:
        _step(scheduler, token_id)
    prefix = list(request.prompt_token_ids) + accepted
    old_tail_hash = request.block_hashes[-1]
    blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    assert request.num_computed_tokens == len(prefix) == 47
    assert len(request.block_hashes) == 3

    parked = scheduler.suspend_requests([request.request_id])[0]

    assert parked["base_len"] == parked["computed_len"] == 47
    assert request.output_token_ids == accepted
    assert len(request.block_hashes) == 2
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id) == blocks
    scheduler.resume_requests([_resume(parked, [7, 8])])
    assert request.output_token_ids == accepted + [7, 8]
    assert list(request.all_token_ids) == prefix + [7, 8]
    assert request.block_hashes[-1] != old_tail_hash
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {request.request_id: 2}
    assert output.scheduled_cached_reqs.num_computed_tokens == [47]


def test_keep_candidate_allows_empty_resume():
    scheduler, request = _parked_request()
    parked = scheduler.suspend_requests([request.request_id], discard_last_token=False)[
        0
    ]
    assert parked["base_len"] == 33
    assert request.output_token_ids == [1000]
    assert scheduler.resume_requests([_resume(parked, [])]) == [request.request_id]
    output = _step(scheduler, 1001)
    assert output.num_scheduled_tokens == {request.request_id: 1}
    assert request.output_token_ids == [1000, 1001]


def test_empty_resume_after_rejection_stays_parked_until_valid_continuation():
    scheduler, request = _parked_request()
    parked = scheduler.suspend_requests([request.request_id])[0]
    with pytest.raises(ValueError, match="requires uncomputed continuation tokens"):
        scheduler.resume_requests([_resume(parked, [])])
    assert request.status == RequestStatus.PEER_WAIT
    assert request.output_token_ids == []
    assert request.num_computed_tokens == 32
    assert scheduler.resume_requests([_resume(parked, [7])]) == [request.request_id]
    assert request.output_token_ids == [7]


@pytest.mark.parametrize("partial_compute", [False, True])
def test_accepted_peer_tail_is_not_a_local_candidate(partial_compute):
    scheduler = create_scheduler(
        use_v2_model_runner=False, max_num_batched_tokens=16, max_model_len=128
    )
    request = create_requests(num_requests=1, num_tokens=16, max_tokens=64)[0]
    scheduler.add_request(request)
    _step(scheduler)
    parked = scheduler.suspend_requests([request.request_id])[0]
    peer_tokens = list(range(100, 117)) if partial_compute else [7]
    scheduler.resume_requests([_resume(parked, peer_tokens)])
    if partial_compute:
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {request.request_id: 16}
        _complete_step(scheduler, output, token_id=None)
    assert request.num_tokens - request.num_computed_tokens == 1
    with pytest.raises(ValueError, match="has no uncomputed local candidate"):
        scheduler.suspend_requests([request.request_id])
    assert request.status == RequestStatus.RUNNING
    assert request.output_token_ids == peer_tokens
    parked = scheduler.suspend_requests([request.request_id], discard_last_token=False)[
        0
    ]
    assert scheduler.resume_requests([_resume(parked, [])]) == [request.request_id]
    _step(scheduler, 2000)
    scheduler.suspend_requests([request.request_id])
    assert request.output_token_ids == peer_tokens


def test_suspend_rejects_scheduled_step_until_output_is_collected():
    scheduler, request = _parked_request()
    output = scheduler.schedule()
    with pytest.raises(ValueError, match="has in-flight tokens"):
        scheduler.suspend_requests([request.request_id])
    assert request.status == RequestStatus.RUNNING
    assert request.output_token_ids == [1000]
    _complete_step(scheduler, output, 1001)
    scheduler.suspend_requests([request.request_id])
    assert request.output_token_ids == [1000]


def test_old_resume_cannot_match_a_new_request_with_the_same_id_and_base():
    scheduler, request = _parked_request()
    old_parked = scheduler.suspend_requests([request.request_id])[0]
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    replacement = create_requests(
        num_requests=1, num_tokens=32, max_tokens=64, req_ids=[request.request_id]
    )[0]
    scheduler.add_request(replacement)
    _step(scheduler)
    new_parked = scheduler.suspend_requests([replacement.request_id])[0]
    assert new_parked["base_len"] == old_parked["base_len"]
    assert new_parked["handoff_seq"] > old_parked["handoff_seq"]
    assert scheduler.resume_requests([_resume(old_parked, [7])]) == []
    assert replacement.status == RequestStatus.PEER_WAIT
    assert replacement.output_token_ids == []
    assert scheduler.resume_requests([_resume(new_parked, [8])]) == [request.request_id]
    assert replacement.output_token_ids == [8]


@pytest.mark.parametrize("max_model_len,max_tokens", [(34, 64), (128, 2)])
def test_peer_continuation_must_leave_local_sampling_budget(max_model_len, max_tokens):
    scheduler = create_scheduler(use_v2_model_runner=False, max_model_len=max_model_len)
    request = create_requests(num_requests=1, num_tokens=32, max_tokens=max_tokens)[0]
    scheduler.add_request(request)
    _step(scheduler)
    parked = scheduler.suspend_requests([request.request_id])[0]
    with pytest.raises(ValueError, match="must leave room for a local token"):
        scheduler.resume_requests([_resume(parked, [7, 8])])
    assert request.status == RequestStatus.PEER_WAIT
    assert request.output_token_ids == []
    assert request.num_computed_tokens == 32
    assert scheduler.resume_requests([_resume(parked, [7])]) == [request.request_id]


@pytest.mark.parametrize("stop_token", [EOS_TOKEN_ID, 999])
def test_peer_continuation_rejects_terminal_tokens(stop_token):
    scheduler = create_scheduler(use_v2_model_runner=False)
    request = create_requests(
        num_requests=1, num_tokens=32, max_tokens=64, stop_token_ids=[999]
    )[0]
    scheduler.add_request(request)
    _step(scheduler)
    parked = scheduler.suspend_requests([request.request_id])[0]
    with pytest.raises(ValueError, match="must not contain stop tokens"):
        scheduler.resume_requests([_resume(parked, [7, stop_token, 8])])
    assert request.status == RequestStatus.PEER_WAIT
    assert request.output_token_ids == []
    assert request.num_computed_tokens == 32


def test_structured_output_handoff_is_rejected_without_mutating_history():
    scheduler = create_scheduler(use_v2_model_runner=False)
    manager = scheduler.structured_output_manager
    backend = XgrammarBackend(
        manager.vllm_config,
        tokenizer=manager.tokenizer,
        vocab_size=manager.vllm_config.model_config.get_vocab_size(),
    )
    request = Request(
        request_id="structured",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=64, structured_outputs=StructuredOutputsParams(regex="[0-9]+")
        ),
        pooling_params=None,
    )
    structured = request.structured_output_request
    assert structured is not None
    structured.grammar = backend.compile_grammar(*structured.structured_output_key)
    scheduler.add_request(request)
    output = scheduler.schedule()
    _complete_step(scheduler, output, token_id=None)
    assert request.status == RequestStatus.RUNNING
    with pytest.raises(ValueError, match="does not support structured output"):
        scheduler.suspend_requests([request.request_id], discard_last_token=False)
    assert request.status == RequestStatus.RUNNING
    assert list(request.all_token_ids) == [0, 1]


def test_pending_peer_history_survives_preemption():
    scheduler, request = _parked_request()
    parked = scheduler.suspend_requests([request.request_id])[0]
    scheduler.resume_requests([_resume(parked, [7, 8])])
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert request.status == RequestStatus.PREEMPTED
    output = scheduler.schedule()
    cached = output.scheduled_cached_reqs
    assert cached.resumed_req_ids == cached.peer_resumed_req_ids == {request.request_id}
    assert cached.all_token_ids[request.request_id] == list(
        request.prompt_token_ids
    ) + [
        7,
        8,
    ]
    assert cached.num_computed_tokens == [0]
