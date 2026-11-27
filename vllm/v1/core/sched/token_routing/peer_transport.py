# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import zmq

from vllm.logger import init_logger
from vllm.v1.core.sched.token_routing.peer_protocol import PeerReq

logger = init_logger(__name__)


@dataclass(frozen=True)
class PeerTransportConfig:
    """ZMQ PUSH/PULL endpoint configuration parsed from VllmConfig."""

    bind_address: str | None = None
    connect_address: str | None = None

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "PeerTransportConfig":
        additional = getattr(vllm_config, "additional_config", None)
        if not isinstance(additional, dict):
            return cls()
        raw = additional.get("token_router")
        if not isinstance(raw, dict):
            return cls()
        bind_addr = raw.get("peer_bind_address")
        connect_addr = raw.get("peer_connect_address")
        return cls(
            bind_address=str(bind_addr) if bind_addr else None,
            connect_address=str(connect_addr) if connect_addr else None,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.bind_address or self.connect_address)


def encode_peer_msgs(msgs: list[PeerReq]) -> bytes:
    return pickle.dumps(msgs, protocol=pickle.HIGHEST_PROTOCOL)


def decode_peer_msgs(payload: bytes) -> list[PeerReq]:
    decoded = pickle.loads(payload)
    if isinstance(decoded, list):
        return decoded
    return [decoded]


@contextmanager
def open_peer_pull_socket(
    ctx: zmq.Context,
    bind_address: str,
) -> Iterator[zmq.Socket]:
    """Bind a ZMQ PULL socket for receiving PeerReq batches from peer engines."""
    if bind_address.startswith("ipc://"):
        ipc_path = bind_address[len("ipc://") :]
        parent = os.path.dirname(ipc_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    sock = ctx.socket(zmq.PULL)
    try:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 0)
        sock.bind(bind_address)
        logger.info("TokenRouter peer PULL socket bound to %s", bind_address)
        yield sock
    finally:
        sock.close(linger=0)


def drain_peer_pull_socket(sock: zmq.Socket) -> list[PeerReq]:
    """Read one ready message and coalesce any immediately queued peer batches."""
    msgs: list[PeerReq] = []
    msgs.extend(decode_peer_msgs(sock.recv()))
    while True:
        try:
            payload = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        msgs.extend(decode_peer_msgs(payload))
    return msgs


class PeerTransportSender:
    """Asynchronous ZMQ PUSH sender used by EngineCoreProc to forward PeerReqs."""

    def __init__(self, connect_address: str) -> None:
        self.connect_address = connect_address
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 0)
        self._socket.connect(connect_address)
        logger.info("TokenRouter peer PUSH socket connected to %s", connect_address)

    def send_batch(self, msgs: list[PeerReq]) -> None:
        if not msgs:
            return
        self._socket.send(encode_peer_msgs(msgs))

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None
