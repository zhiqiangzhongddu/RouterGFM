import math
import os
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm"


def _data_row_count(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _array_max(text: str) -> int:
    match = re.search(r"^#SBATCH\s+--array=0-(\d+)", text, re.MULTILINE)
    assert match, "launcher must use a zero-based contiguous default array"
    return int(match.group(1))


def test_dependency_pins_are_transformers_compatible():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "transformers==4.36.2" in requirements
    assert "huggingface_hub==0.36.2" in requirements
    assert "huggingface_hub==1.2.3" not in requirements
    for direct_dependency in (
        "scipy==1.15.3",
        "scikit-learn==1.7.2",
        "networkx==3.4.2",
        "PyYAML==6.0.3",
    ):
        assert direct_dependency in requirements
    # context generation (openai) and the old cluster evaluation (munkres)
    # are not part of this repository.
    for dropped_dependency in ("openai", "munkres"):
        assert dropped_dependency not in requirements


def test_trimmed_tsv_launchers_have_minimal_default_arrays():
    for path in sorted(SLURM.glob("*.slurm")):
        text = path.read_text(encoding="utf-8")
        experiment = re.search(
            r'EXPERIMENT_FILE="\$\{ROOT_DIR\}/slurm/([^"}]+\.tsv)"', text
        )
        rows_per_gpu = re.search(
            r'ROWS_PER_GPU="\$\{ROWS_PER_GPU:-(\d+)\}"', text
        )
        if not experiment or not rows_per_gpu:
            continue

        table = SLURM / experiment.group(1)
        rows = _data_row_count(table)
        capacity = 4 * int(rows_per_gpu.group(1))
        expected_max = max(0, math.ceil(rows / capacity) - 1)
        assert _array_max(text) == expected_max, path.name
        assert (
            re.search(
                r"if \(\( ARRAY_MAX \+ 1 < EXPECTED_ARRAYS \)\).*?exit 2",
                text,
                re.DOTALL,
            )
            or "row_undersized_array.failure" in text
        ), path.name


def test_shell_launchers_parse_with_bash():
    launchers = sorted(SLURM.glob("*.slurm")) + sorted(SLURM.glob("*.sh"))
    for launcher in launchers:
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_experiment_parser_launchers_share_clean_zero_row_noop():
    callers = []
    for launcher in sorted(SLURM.glob("*.slurm")) + sorted(SLURM.glob("*.sh")):
        if launcher.name == "_common.sh":
            continue
        text = launcher.read_text(encoding="utf-8")
        if "parse_experiment_file" not in text:
            continue
        callers.append(launcher)
        assert "_common.sh" in text, launcher.name

        parse_calls = list(
            re.finditer(r"^parse_experiment_file\b", text, re.MULTILINE)
        )
        assert parse_calls, launcher.name
        if launcher.suffix == ".slurm":
            srun = re.search(r"^if srun\b", text, re.MULTILINE)
            assert srun, launcher.name
            assert parse_calls[0].start() < srun.start(), launcher.name
            assert len(parse_calls) == 2, launcher.name
            assert "export EXPERIMENT_FILE EXPERIMENT_COLUMNS" in text, launcher.name
    assert callers, "expected at least one launcher using parse_experiment_file"

    with tempfile.TemporaryDirectory() as tmp:
        fixtures = {
            "empty.tsv": "",
            "header_only.tsv": "# dataset model task_level\n",
            "uncommented_header_only.tsv": "dataset\tmodel\ttask_level\n",
            "comments_only.tsv": "# intentionally empty experiment\n\n",
        }
        for filename, contents in fixtures.items():
            experiment = Path(tmp) / filename
            experiment.write_text(contents, encoding="utf-8")
            command = f'''
function conda() {{ :; }}
export -f conda
export CONDA_BASE="{tmp}/missing-conda"
export SLURM_SUBMIT_DIR="{ROOT}"
source "{SLURM}/_common.sh"
parse_experiment_file "{experiment}" "dataset|model|task_level" "[test]"
echo SHOULD_NOT_RUN
'''
            result = subprocess.run(
                ["bash", "-c", command],
                check=False,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, filename
            assert "No data rows" in result.stdout, filename
            assert "nothing to run" in result.stdout, filename
            assert "SHOULD_NOT_RUN" not in result.stdout, filename


def test_all_tsv_batch_launchers_noop_before_starting_srun():
    launchers = []
    for launcher in sorted(SLURM.glob("*.slurm")):
        text = launcher.read_text(encoding="utf-8")
        if not re.search(r"^parse_experiment_file\b", text, re.MULTILINE):
            continue
        launchers.append((launcher, text))
    assert launchers, "expected at least one TSV batch launcher"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        tmp_slurm = tmp_root / "slurm"
        tmp_bin = tmp_root / "bin"
        tmp_slurm.mkdir()
        tmp_bin.mkdir()
        (tmp_slurm / "_common.sh").write_text(
            (SLURM / "_common.sh").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        conda_stub = tmp_bin / "conda"
        conda_stub.write_text(
            "#!/bin/bash\n"
            "if [[ \"${1:-}\" == shell.bash && \"${2:-}\" == hook ]]; then\n"
            "  printf 'conda() { return 0; }\\n'\n"
            "fi\n",
            encoding="utf-8",
        )
        conda_stub.chmod(0o755)
        srun_stub = tmp_bin / "srun"
        srun_stub.write_text(
            "#!/bin/bash\necho 'SRUN_MUST_NOT_START' >&2\nexit 97\n",
            encoding="utf-8",
        )
        srun_stub.chmod(0o755)

        env = os.environ.copy()
        env.update(
            {
                "CONDA_BASE": str(tmp_root / "missing-conda"),
                "PATH": f"{tmp_bin}:{env.get('PATH', '')}",
                "SLURM_SUBMIT_DIR": str(tmp_root),
            }
        )

        for launcher, text in launchers:
            experiment_match = re.search(
                r'EXPERIMENT_FILE="\$\{ROOT_DIR\}/slurm/([^"}]+)"', text
            )
            columns_match = re.search(r'^EXPERIMENT_COLUMNS="([^"]+)"', text, re.MULTILINE)
            assert experiment_match, launcher.name
            assert columns_match, launcher.name

            experiment = tmp_slurm / experiment_match.group(1)
            experiment.write_text(
                f"# {columns_match.group(1).replace('|', ' ')}\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                ["bash", str(launcher)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            assert result.returncode == 0, (launcher.name, result.stderr)
            assert "No data rows" in result.stdout, launcher.name
            assert "nothing to run" in result.stdout, launcher.name
            assert "SRUN_MUST_NOT_START" not in result.stderr, launcher.name
