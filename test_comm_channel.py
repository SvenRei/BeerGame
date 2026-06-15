"""
Smoke tests for the comm_qmix communication channel.

Run from the repo root:
    pytest scripts/test_comm_channel.py -q
or directly:
    python scripts/test_comm_channel.py

Guards the things that, if silently broken, make the language useless:
  1. MAC public return arity (the unpack-bug regression) -> 5.
  2. Gradient actually reaches msg_stream through the incoming-message path
     with straight-through (hard=True) training.
  3. vocab_size=1 is a true no-op channel.
  4. Deterministic eval (sample=False) really is deterministic, and the
     per-edge incoming message has width 2.
"""
import os, sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.rl.qmix import CommQMixLocalAgent, QMixCommMAC, MessageDecoder

N, OBS, HID, A, V = 4, 6, 16, 21, 3


def _build(vocab=V):
    return QMixCommMAC(CommQMixLocalAgent(OBS, HID, A, vocab_size=vocab), num_agents=N)


def test_mac_return_arity_and_incoming_width():
    mac = _build()
    out = mac(torch.randn(2, N, OBS), torch.zeros(2, N, HID),
              tau=1.0, msg_in=torch.zeros(2, N, 1), hard=True)
    assert len(out) == 5, f"MAC must return 5 values, got {len(out)}"
    q, nh, mo, sl, inc = out
    assert q.shape == (2, N, A)
    assert inc.shape == (2, N, 2), f"per-edge incoming must be width 2, got {tuple(inc.shape)}"


def test_gradient_reaches_msg_stream():
    """t0 emits, t1 consumes. A loss on the incoming message at t1 must give a
    non-zero gradient on the sender's msg_stream (straight-through, hard=True)."""
    torch.manual_seed(0)
    mac = _build(); agent = mac.agent
    dec = MessageDecoder(OBS, A, msg_dim=2, hidden=32)
    obs = torch.randn(2, N, OBS); h = torch.zeros(2, N, HID)
    _, h0, mo0, _, _ = mac(obs, h, tau=1.0, msg_in=torch.zeros(2, N, 1), hard=True)
    q1, _, _, _, inc1 = mac(obs, h0, tau=1.0, msg_in=mo0, hard=True)
    logits = dec(obs / 100.0, inc1)
    teacher = q1.argmax(-1).detach()
    loss = F.cross_entropy(logits.reshape(-1, A), teacher.reshape(-1)) + q1.mean()
    mac.zero_grad(); dec.zero_grad(); loss.backward()
    g = agent.msg_stream.weight.grad
    assert g is not None and float(g.norm()) > 0, "msg_stream got no gradient -- channel is dead!"


def test_vocab1_is_inert():
    mac = _build(vocab=1)
    _, _, mo, _, _ = mac(torch.randn(2, N, OBS), torch.zeros(2, N, HID),
                         tau=1.0, msg_in=torch.zeros(2, N, 1), hard=True)
    assert bool((mo == 0).all()), "vocab=1 must emit constant 0 (no-comm control)"


def test_eval_is_deterministic():
    """sample=False must give identical messages regardless of RNG state."""
    torch.manual_seed(0)
    mac = _build()
    obs = torch.randn(2, N, OBS); h = torch.zeros(2, N, HID)
    torch.manual_seed(1); _, _, a, _, _ = mac(obs, h, tau=0.5, msg_in=torch.zeros(2, N, 1), hard=True, sample=False)
    torch.manual_seed(2); _, _, b, _, _ = mac(obs, h, tau=0.5, msg_in=torch.zeros(2, N, 1), hard=True, sample=False)
    assert bool(torch.equal(a, b)), "eval (sample=False) must be deterministic"


if __name__ == "__main__":
    test_mac_return_arity_and_incoming_width(); print("PASS  return arity = 5, incoming width = 2")
    test_gradient_reaches_msg_stream();          print("PASS  gradient reaches msg_stream")
    test_vocab1_is_inert();                      print("PASS  vocab=1 inert")
    test_eval_is_deterministic();                print("PASS  eval deterministic")
    print("\nAll smoke tests passed.")