"""Relocation and cache completeness for the anonymous distribution."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts.check_e_cell import complete
from scripts.check_a_inputs import check
from utils.paths import project_path, PROJECT_ROOT
from utils.provenance import git_commit, git_is_dirty


def test_resources_do_not_follow_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path(project_path("data")) == PROJECT_ROOT / "data"


def test_source_free_copy_does_not_read_parent_git(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: pytest.fail("parent repository must not be queried"))
    assert git_commit(tmp_path) == "unknown"
    assert git_is_dirty(tmp_path) is None


def record(seeds):
    return {"seeds": seeds, "acc": [0.5] * len(seeds), "f1": [0.4] * len(seeds)}


def test_three_seed_cache_cannot_satisfy_five_seed_request():
    c = {"models": {"mlp": {"configs": {"a": record([42, 43, 44])}}}}
    assert not complete(c, ["mlp"], [42,43,44,45,46], [.1], ["a"])
    c["models"]["mlp"]["configs"]["a"] = record([42,43,44,45,46])
    assert complete(c, ["mlp"], [42,43,44,45,46], [.1], ["a"])


def test_missing_coverage_and_duplicate_seed_do_not_pass():
    c = {"models": {"mlp": {"configs": {"c@0.1": record([42,43,44,45,46])}}}}
    assert not complete(c, ["mlp"], [42,43,44,45,46], [.1,.5,.9], ["c"])
    c["models"]["mlp"]["configs"]["c@0.1"] = record([42,43,44,45,45])
    assert not complete(c, ["mlp"], [42,43,44,45,46], [.1], ["c"])


def test_random_control_is_read_from_primary_model_block():
    seeds=[42,43,44,45,46]
    c={"configs":{"f@0.1":record(seeds)},"models":{"mlp":{"configs":{}}}}
    assert complete(c,["mlp","linear"],seeds,[.1],["f"])


def test_set_b_rejects_missing_seed_and_sidecar(tmp_path):
    j=tmp_path/"json"; n=tmp_path/"npz";j.mkdir();n.mkdir()
    for seed in [42,43,44,45]:
        name=f"text_example_s{seed}_n0.2"
        (j/(name+".json")).write_text("{}")
        (n/(name+".npz")).write_bytes(b"test")
    with pytest.raises(ValueError,match="expected seeds"):
        check(j,n,[42,43,44,45,46])
    (j/"text_example_s46_n0.2.json").write_text("{}")
    with pytest.raises(ValueError,match="Missing per-query"):
        check(j,n,[42,43,44,45,46])
    (n/"text_example_s46_n0.2.npz").write_bytes(b"test")
    assert check(j,n,[42,43,44,45,46])==1
