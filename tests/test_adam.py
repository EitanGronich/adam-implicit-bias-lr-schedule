"""Independent numerical oracles for the shared discrete Adam update."""
import math

import pytest
import torch

from adam_lr.optim.adam import (
    InspectableAdam, NumericalError, logistic_loss_and_logits, schedule_learning_rate,
)


@pytest.mark.parametrize("betas", [(0., 0.), (.9, .999), (.8, .9)])
@pytest.mark.parametrize("epsilon", [0., 1e-8])
def test_matches_pytorch_with_changing_schedule(betas, epsilon):
    torch.manual_seed(12)
    parameter = torch.nn.Parameter(torch.randn(13, dtype=torch.float64))
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    actual = InspectableAdam([parameter], *betas, epsilon)
    reference = torch.optim.Adam([reference_parameter], lr=.01, betas=betas,
                                 eps=epsilon, foreach=False)
    for step in range(1, 31):
        gradient = torch.randn(13, dtype=torch.float64) / step
        parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
        rate = .01 * step ** (-.7)
        reference.param_groups[0]["lr"] = rate
        actual.step(rate)
        reference.step()
        torch.testing.assert_close(parameter, reference_parameter, rtol=2e-13, atol=2e-15)
        torch.testing.assert_close(actual.first[0], reference.state[reference_parameter]["exp_avg"], rtol=1e-13, atol=1e-15)
        torch.testing.assert_close(actual.second[0], reference.state[reference_parameter]["exp_avg_sq"], rtol=1e-13, atol=1e-15)


def test_scalar_recurrence_oracle():
    parameter = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    adam = InspectableAdam([parameter], .4, .6, 0.)
    expected, first, second = 2., 0., 0.
    for t, gradient in enumerate([3., -.1, 0., -2.], start=1):
        first = .4*first + .6*gradient
        second = .6*second + .4*gradient*gradient
        expected -= (.2/t**.5)*(first/(1-.4**t))/math.sqrt(second/(1-.6**t))
        parameter.grad = torch.tensor([gradient], dtype=torch.float64)
        adam.step(.2/t**.5)
        assert parameter.item() == pytest.approx(expected, abs=3e-15)


def test_exact_zero_history_is_well_defined_and_later_activates():
    p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    adam = InspectableAdam([p], .9, .999, 0.)
    p.grad = torch.tensor([2., 0.], dtype=torch.float64)
    adam.step(.1)
    assert p.tolist() == pytest.approx([.9, 2.])
    p.grad = torch.tensor([0., -3.], dtype=torch.float64)
    adam.step(.1)
    assert p[1].item() > 2


def test_second_moment_underflow_is_failure_not_stationarity():
    p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    adam = InspectableAdam([p], .9, .999, 0.)
    p.grad = torch.tensor([1e-200], dtype=torch.float64)
    with pytest.raises(NumericalError, match="underflow"):
        adam.step(.1)
    assert p.item() == 1 and adam.step_count == 0
    assert adam.first[0].item() == 0 and adam.second[0].item() == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_bad_second_tensor_does_not_partially_commit_first(bad):
    ps = [torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64)) for _ in range(2)]
    adam = InspectableAdam(ps, .9, .999, 0.)
    ps[0].grad = torch.tensor([1.], dtype=torch.float64)
    ps[1].grad = torch.tensor([bad], dtype=torch.float64)
    with pytest.raises(NumericalError):
        adam.step(.1)
    assert [p.item() for p in ps] == [1., 1.]
    assert adam.step_count == 0


def test_column_output_logistic_loss_does_not_broadcast_labels():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., -1.]], dtype=torch.float64))
    x = torch.tensor([[2., 1.], [0., 1.]], dtype=torch.float64)
    y = torch.tensor([1., -1.], dtype=torch.float64)
    loss, logits = logistic_loss_and_logits(model, x, y)
    assert logits.shape == (2,)
    assert loss.item() == pytest.approx(math.log1p(math.exp(-1)))


def test_schedule_index_and_signed_power():
    assert schedule_learning_rate(.2, .7, 1) == .2
    assert schedule_learning_rate(.2, .5, 100) == pytest.approx(.02)
    assert schedule_learning_rate(.2, -.25, 16) == pytest.approx(.4)
    for step in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            schedule_learning_rate(.1, .5, step)


@pytest.mark.parametrize("field,bad", [("epsilon", float("nan")), ("epsilon", float("inf")), ("epsilon", -1.), ("beta1", float("nan")), ("beta2", 1.)])
def test_invalid_hyperparameters(field, bad):
    params = dict(beta1=.9, beta2=.999, epsilon=0.)
    params[field] = bad
    with pytest.raises(ValueError):
        InspectableAdam([torch.nn.Parameter(torch.ones(1))], **params)
