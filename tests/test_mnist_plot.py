"""Read and display helpers preserve exact saved semantics."""
import json
from pathlib import Path
import numpy as np
import pytest
from adam_lr.mnist.plot import (read_incoming_for_display, x_values, scalar_fields, load_coordinate_panels, _display_coordinates)

def test_loss_clock_masks_undefined_loss():
    x = x_values({"loss":np.array([1., np.exp(-2), 0., np.nan])},"loss-clock")
    np.testing.assert_allclose(x[:2],[0.,2.])
    assert np.isnan(x[2:]).all()

def test_scalar_fields_rejects_vector_sketches():
    runs = [(Path("run/results.json"),{}, {"step":np.arange(3), "sketch":np.ones((3,4))})]
    assert scalar_fields(runs,["step"]) == ["step"]
    with pytest.raises(ValueError,match="scalar field"):
        scalar_fields(runs,["sketch"])
    with pytest.raises(ValueError):
        scalar_fields(runs,["missing"])

def test_streaming_display_is_bounded_and_retains_endpoints(tmp_path):
    path = tmp_path/"trace.jsonl"
    with path.open("w") as stream:
        for step in range(10001):
            stream.write(json.dumps({"step":step,"log_D_total":None if step==0 else -float(step)})+"\n")
    x,y,total = read_incoming_for_display(path,32)
    assert len(x) <= 32
    assert total == 10001
    assert x[0] == 0 and x[-1] == 10000
    assert np.isnan(y[0])
    np.testing.assert_array_equal(y[1:],-x[1:].astype(float))
    assert np.all(np.diff(x)>0)

def test_coordinate_schema_and_stratified_display(tmp_path):
    path = tmp_path/"coordinates.npz"
    np.savez(path,step=[0,2],loss=[.6,.5],hidden1_indices=[7,9,11,12],hidden1_sample_type=["persistent","persistent","random","random"],hidden1_raw_second=np.ones((2,4)))
    series,layers=load_coordinate_panels(path)
    assert layers == ["hidden1"]
    assert series["hidden1_raw_second"].shape == (2,4)
    assert _display_coordinates(series["hidden1_sample_type"],2) == [0,2]
