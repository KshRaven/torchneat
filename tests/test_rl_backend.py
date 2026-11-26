import numpy as np
import pytest

from torchneat.rl.backend import CpuRLBackend, RLBackend, get_rl_backend, resolve_device


class Reference(RLBackend):
    """The device agnostic kernels of the base class, on NumPy."""
    name = 'reference'


def _data(seed=0):
    rng = np.random.default_rng(seed)
    G, cap = 4, 12
    lengths = np.array([12, 9, 12, 7], np.int64)
    ep = np.zeros((G, cap), np.int64)
    for g in range(G):
        ep[g, :lengths[g]] = np.sort(rng.integers(0, 3, lengths[g]))
    rows = np.array([0, 1, 3], np.int64)
    return rng, G, cap, lengths, ep, rows


@pytest.fixture(scope='module')
def backends():
    return CpuRLBackend(), Reference()


def _same(a, b):
    np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=1e-9, atol=1e-9, equal_nan=True)


def test_episode_runs_and_means(backends):
    cpu, ref = backends
    rng, G, cap, lengths, ep, rows = _data()
    runs_c, runs_r = cpu.episode_runs(ep, lengths, rows), ref.episode_runs(ep, lengths, rows)
    for name in ('starts', 'ends', 'ids', 'counts', 'slot'):
        np.testing.assert_array_equal(getattr(runs_c, name), getattr(runs_r, name))

    values = rng.normal(size=(G, cap, 3))
    _same(cpu.record_mean(values, lengths, rows), ref.record_mean(values, lengths, rows))
    per_record = cpu.record_mean(values, lengths, rows)
    _same(cpu.segment_mean(per_record, runs_c, rows), ref.segment_mean(per_record, runs_r, rows))

    matrix = rng.normal(size=(G, 5))
    counts = np.array([5, 3, 4, 1], np.int64)
    for ddof in (0, 1):
        for a, b in zip(cpu.row_mean_std(matrix, counts, rows, ddof), ref.row_mean_std(matrix, counts, rows, ddof)):
            _same(a, b)


@pytest.mark.parametrize('kind', ['continuous', 'binary', 'discrete', 'bipolar'])
def test_accuracy(backends, kind):
    cpu, ref = backends
    rng, G, cap, lengths, _, rows = _data()
    pred, target = rng.uniform(-1, 1, (G, cap, 2)), rng.uniform(-1, 1, (G, cap, 2))
    target[:, ::2] = pred[:, ::2]
    _same(cpu.accuracy(pred, target, lengths, rows, 0.1, kind), ref.accuracy(pred, target, lengths, rows, 0.1, kind))


@pytest.mark.parametrize('mode', ['ppo', 'a2c', 'trpo'])
def test_policy_terms(backends, mode):
    cpu, ref = backends
    rng, G, cap, lengths, _, rows = _data()
    new, old = rng.normal(size=(G, cap, 2)), rng.normal(size=(G, cap, 2))
    adv, ent = rng.normal(size=(G, cap, 1)), rng.uniform(size=(G, cap, 1))
    new[0, 0, 0] = np.nan
    for a, b in zip(cpu.policy_terms(new, old, adv, ent, lengths, rows, mode),
                    ref.policy_terms(new, old, adv, ent, lengths, rows, mode)):
        _same(a, b)


def test_value_td_and_ghost_terms(backends):
    cpu, ref = backends
    rng, G, cap, lengths, ep, rows = _data()
    values, rets = rng.normal(size=(G, cap, 1)), rng.normal(size=(G, cap, 1))
    _same(cpu.value_terms(values, rets, lengths, rows, 1e-12), ref.value_terms(values, rets, lengths, rows, 1e-12))

    q = rng.normal(size=(G, cap, 3))
    actions = rng.integers(0, 3, (G, cap, 1))
    rewards = rng.normal(size=(G, cap, 1))
    for gamma in (0.0, 0.9):
        _same(cpu.td_terms(q, actions, rewards, ep, lengths, rows, gamma),
              ref.td_terms(q, actions, rewards, ep, lengths, rows, gamma))

    means = rng.normal(size=(G, cap, 3))
    src = np.array([[1, -1], [0, 3], [-1, -1], [0, 1]], np.int64)
    use = np.array([True, True, False, True])
    for discrete in (False, True):
        _same(cpu.ghost_reduction(means, src, use, lengths, rows, discrete),
              ref.ghost_reduction(means, src, use, lengths, rows, discrete))


def test_td_error_uses_next_record_of_the_same_episode():
    cpu = CpuRLBackend()
    q = np.array([[[1.0, 0.0], [0.0, 3.0], [5.0, 5.0]]])
    actions = np.array([[[0], [1], [0]]])
    rewards = np.array([[[1.0], [2.0], [4.0]]])
    ep = np.array([[0, 0, 1]])
    td = cpu.td_terms(q, actions, rewards, ep, np.array([3]), np.array([0]), 0.5)
    # record 0: 1 + 0.5 * 3 - 1 = 1.5; record 1 is the last of its episode: 2 - 3 = -1; record 2: 4 - 5 = -1
    np.testing.assert_allclose(td[0], [2.25, 1.0, 1.0])


def test_rank_order_gather_scatter_round_trip(backends):
    cpu, ref = backends
    rng, G, cap, lengths, ep, rows = _data()
    scores = rng.normal(size=(G, cap))
    d = ref.asarray
    for descending in (False, True):
        order_c = cpu._k_rank_order(scores, lengths, rows, descending)
        order_r = ref._k_rank_order(scores, lengths, rows, descending)
        np.testing.assert_array_equal(order_c, order_r)
    arr = rng.normal(size=(G, cap, 2))
    order = cpu._k_rank_order(scores, lengths, rows, False)
    gathered = cpu._k_gather(arr, order, lengths, rows)
    _same(gathered, ref._k_gather(arr, order, lengths, rows))
    restored = cpu._k_scatter(gathered, order, lengths, rows)
    valid = np.arange(cap)[None, :] < lengths[:, None]
    selected = np.isin(np.arange(G), rows)[:, None]
    _same(restored[valid & selected], arr[valid & selected])


def test_backend_follows_set_device_and_override():
    import torchneat

    assert resolve_device() == 'cpu'
    assert get_rl_backend().name == 'cpu'
    assert get_rl_backend('cpu').name == 'cpu'
    with pytest.raises(ValueError):
        resolve_device('tpu')
    if torchneat.cuda_is_available():
        torchneat.set_device('cuda')
        assert get_rl_backend().name == 'cuda'
        torchneat.set_device('cpu')


@pytest.mark.gpu
def test_cuda_backend_matches_cpu():
    from torchneat.rl.backend import CudaRLBackend

    cpu, gpu = CpuRLBackend(), CudaRLBackend()
    rng, G, cap, lengths, ep, rows = _data()
    rewards = rng.normal(size=(G, cap, 2))
    genus = np.array([0, 0, 1, 1], np.int64)
    for kwargs in (dict(), dict(kappa=0.5, normalize=1), dict(normalize=3, alpha=1.1, order=1),
                   dict(order=3, normalize=2), dict(gamma=0.0, kappa=0.0, normalize=4)):
        _same(cpu.compute_returns(rewards, ep, lengths, rows, genus, **kwargs),
              gpu.compute_returns(rewards, ep, lengths, rows, genus, **kwargs))
    values = rng.normal(size=(G, cap, 2))
    for a, b in zip(cpu.compute_gae(rewards, values, ep, lengths, rows), gpu.compute_gae(rewards, values, ep, lengths, rows)):
        _same(a, b)
    runs_c, runs_g = cpu.episode_runs(ep, lengths, rows), gpu.episode_runs(ep, lengths, rows)
    per_record = cpu.record_mean(rewards, lengths, rows)
    _same(cpu.segment_mean(per_record, runs_c, rows), gpu.segment_mean(per_record, runs_g, rows))
