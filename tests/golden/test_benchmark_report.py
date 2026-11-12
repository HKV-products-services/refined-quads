import json

from golden import benchmarks


def write_results(path, times):
    data = {
        "machine_info": {
            "system": "Linux",
            "cpu": {"brand_raw": "test cpu"},
            "python_version": "3.12",
        },
        "benchmarks": [
            {
                "name": f"test_generate[{name}]",
                "extra_info": {"scenario": name},
                "stats": {"min": t},
            }
            for name, t in times.items()
        ],
    }
    path.write_text(json.dumps(data))
    return path


def test_change_above_the_threshold_is_reported_as_slower(tmp_path, capsys):
    now = write_results(tmp_path / "now.json", {"a": 1.06, "b": 1.04, "c": 0.5})
    before = write_results(tmp_path / "before.json", {"a": 1.0, "b": 1.0, "c": 1.0})

    code = benchmarks.main([str(now), "--baseline", str(before)])

    out = capsys.readouterr().out
    assert code == 0
    assert "| a | 1.000 | 1.060 | +6.0% | slower |" in out
    assert "| b | 1.000 | 1.040 | +4.0% |  |" in out
    assert "| c | 1.000 | 0.500 | -50.0% |  |" in out


def test_fail_option_exits_1_when_slower(tmp_path):
    now = write_results(tmp_path / "now.json", {"a": 1.2})
    before = write_results(tmp_path / "before.json", {"a": 1.0})

    assert benchmarks.main([str(now), "--baseline", str(before), "--fail"]) == 1
    assert (
        benchmarks.main([str(now), "--baseline", str(before), "--threshold", "25", "--fail"]) == 0
    )


def test_github_actions_gets_warnings_and_a_step_summary(tmp_path, capsys, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    now = write_results(tmp_path / "now.json", {"a": 1.2})
    before = write_results(tmp_path / "before.json", {"a": 1.0})

    benchmarks.main([str(now), "--baseline", str(before)])

    assert "::warning title=Benchmark a::+20.0%" in capsys.readouterr().out
    assert "| a | 1.000 | 1.200 | +20.0% | slower |" in summary.read_text()


def test_default_baseline_is_the_legacy_runtime(tmp_path, capsys):
    now = write_results(tmp_path / "now.json", {"l_shaped_domain": 0.01})

    benchmarks.main([str(now)])

    out = capsys.readouterr().out
    assert "baseline: qgis_dhydamo 9a324bc" in out
    assert "| l_shaped_domain |" in out


def test_scenario_without_baseline_is_listed_without_change(tmp_path, capsys):
    now = write_results(tmp_path / "now.json", {"new": 1.0})
    before = write_results(tmp_path / "before.json", {"old": 1.0})

    benchmarks.main([str(now), "--baseline", str(before)])

    assert "| new | - | 1.000 | - |  |" in capsys.readouterr().out
