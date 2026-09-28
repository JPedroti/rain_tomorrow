"""Static checks of the OOT boundary in the code (run without data or artifacts)."""
import re

from src import config

SRC = config.PROJECT_ROOT / "src"
SCRIPTS = config.PROJECT_ROOT / "scripts"


def test_training_code_never_touches_the_oot():
    source = (SCRIPTS / "train.py").read_text(encoding="utf-8")
    for forbidden in ("load_oot_data", "load_raw", "split_development_oot", "read_csv", "evaluation"):
        assert not re.search(rf"\b{forbidden}\b", source), forbidden
    for module in ("modeling.py", "preprocessing.py", "metrics.py"):
        assert "load_oot_data" not in (SRC / module).read_text(encoding="utf-8")


def test_oot_stage_code_never_fits():
    for path in (SCRIPTS / "evaluate.py", SCRIPTS / "predict.py", SRC / "evaluation.py"):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"\.fit\(|fit_transform|build_pipeline|run_model_selection|evaluate_candidates", source), path.name


def test_oot_is_only_loaded_by_the_evaluation_script():
    users = [p.relative_to(config.PROJECT_ROOT).as_posix()
             for p in [*SRC.glob("*.py"), *SCRIPTS.glob("*.py")]
             if "load_oot_data(" in p.read_text(encoding="utf-8") and p.name != "data.py"]
    assert users == ["scripts/evaluate.py"]
