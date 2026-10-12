import math
import json
import hashlib
import numpy as np
import pytest
from adam_lr.mnist.incoming_analysis import display_indices, transition_values, load_trace, main


def test_adjacent_transition_identity_and_terminal_exclusion():
    # x=2,4,8 -> both delta ln(x)=ln2; incoming mass=3,6.
    losses = -np.array([2., 4., 8.])*math.log(10)
    logs = np.log([3., 6., 1e200])
    result = transition_values(losses, logs, reference_loss=.1)
    np.testing.assert_allclose(result["log_F"], np.log([3.,6.]/np.log(2)))
    np.testing.assert_allclose(result["log10_relative_F"], [0,math.log10(2)])
    assert result["step"].tolist() == [0,1]


def test_reversals_zero_mass_and_plateaus_remain_gaps():
    x = np.array([2.,3.,3.,2.5,4.,5.,6.])
    log_d = np.array([0.,0.,0.,-np.inf,0.,0.,999.])
    result = transition_values(-x*math.log(10), log_d, reference_loss=.1)
    assert result["valid"].tolist() == [True,False,False,False,True,True]
    assert np.isnan(result["log10_relative_F"][[1,2,3]]).all()
    assert result["invalid_transitions"] == 3


def test_reference_uses_first_valid_transition_after_threshold():
    x = np.array([1.,2.,3.,4.,5.])
    result = transition_values(-x*math.log(10), [0.,-np.inf,5.,7.,99.], reference_loss=.01)
    assert result["reference_step"] == 2
    assert np.isnan(result["log10_relative_F"][0])
    assert result["log10_relative_F"][1] == 0


def test_log_magnitudes_remain_representable_without_exponentiating():
    result = transition_values([-10.,-20.,-30.], [10000.,10001.,10002.], reference_loss=.1)
    assert np.isfinite(result["log_F"]).all()
    assert np.isfinite(result["log10_relative_F"]).all()


def test_one_ulp_loss_change_remains_a_valid_transition():
    second = np.nextafter(-100., -np.inf)
    result = transition_values([-100., second, second-1.], [0., 0., 0.], reference_loss=.5)
    assert result["valid"].tolist() == [True, True]
    expected_delta = math.log1p((-second-100.)/100.)
    assert result["log_F"][0] == pytest.approx(-math.log(expected_delta))


def test_thinning_preserves_gap_boundaries():
    values = np.ones(1000)
    values[37:41] = np.nan
    selected = display_indices(values, 10)
    assert {0,36,37,40,41,999} <= set(selected)


@pytest.mark.parametrize("reference", [0.,1.,-1.,float("nan")])
def test_invalid_reference(reference):
    with pytest.raises(ValueError):
        transition_values([-1.,-2.], [0.,0.], reference_loss=reference)


def test_no_tail_after_reference_rejected():
    with pytest.raises(ValueError, match="no transition remains"):
        transition_values([-1.,-2.], [0.,0.], reference_loss=1e-8)


def make_run(path):
    path.mkdir()
    columns = ["step", "log_loss", "log_D_total", "log_D_output"]
    rows = [dict(zip(columns, [step, -2.*(step+1), 3.+step, 3.+step])) for step in range(3)]
    trace = path/"incoming_second_moment.jsonl"
    trace.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    meta = path/"incoming_second_moment.meta.json"
    meta.write_text(json.dumps(dict(version=1, signature={"seed":1}, layer_names=["output"],
                                    columns=columns, rows_written=3, last_step_written=2)), encoding="utf-8")
    result = dict(signature={"seed":1}, completion_reason="max_steps", final_step=2,
                  final_loss=math.exp(-6), artifacts={})
    for file in (trace,meta):
        result["artifacts"][file.name] = dict(size_bytes=file.stat().st_size,
                                             sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (path/"results.json").write_text(json.dumps(result), encoding="utf-8")
    return rows,result


def test_loader_and_plot_integration(tmp_path):
    run = tmp_path/"run"
    make_run(run)
    loss,incoming,_ = load_trace(run)
    np.testing.assert_array_equal(loss, [-2,-4,-6])
    np.testing.assert_array_equal(incoming, [3,4,5])
    output = tmp_path/"figures"
    main([str(run), "--reference-loss", "0.5", "--output-dir", str(output)])
    assert (output/"incoming_normalized.png").stat().st_size > 1000
    assert (output/"incoming_normalized.pdf").stat().st_size > 1000
    assert (output/"manifest.json").exists()


def test_missing_incoming_field_is_not_silently_zero(tmp_path):
    run = tmp_path/"run"
    rows,_ = make_run(run)
    del rows[0]["log_D_total"]
    (run/"incoming_second_moment.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="contiguous"):
        load_trace(run)


def test_missing_artifact_identity_rejected(tmp_path):
    run = tmp_path/"run"
    _,result = make_run(run)
    result["artifacts"] = {}
    (run/"results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_trace(run)


def test_valid_but_tampered_trace_rejected(tmp_path):
    run = tmp_path/"run"
    rows,_ = make_run(run)
    rows[0]["log_D_total"] = 99.
    (run/"incoming_second_moment.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_trace(run)
