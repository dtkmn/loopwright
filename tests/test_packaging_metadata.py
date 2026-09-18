import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pkg_name(dep_str: str) -> str:
    name = re.split(r"[><=!;\s\[]", dep_str)[0].strip()
    return name.lower().replace("-", "_")


def pyproject() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def uv_lock() -> dict:
    return tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())


def locked_dependency_strings(entries: list[dict]) -> tuple[str, ...]:
    return tuple(f"{entry['name']}{entry.get('specifier', '')}" for entry in entries)


def locked_project_package() -> dict:
    for package in uv_lock()["package"]:
        if package["name"] == "loopwright":
            return package
    raise AssertionError("loopwright package missing from uv.lock")


def test_static_frontend_assets_are_packaged():
    package_data = pyproject()["tool"]["setuptools"]["package-data"]

    assert "web_static/*" in package_data["src"]
    assert (PROJECT_ROOT / "src" / "web_static" / "index.html").exists()
    assert (PROJECT_ROOT / "src" / "web_static" / "app.js").exists()
    assert (PROJECT_ROOT / "src" / "web_static" / "styles.css").exists()


def test_docker_build_context_excludes_local_env_files():
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text().splitlines()

    assert ".env" in dockerignore
    assert ".env.*" in dockerignore
    assert "!.env.example" in dockerignore
    assert ".venv/" in dockerignore
    assert (PROJECT_ROOT / ".env.example").exists()


def test_docker_user_creation_is_noninteractive():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()

    assert 'adduser --uid 1000 --gid 1000 --disabled-password --gecos ""' in (
        dockerfile
    )


def test_removed_model_stack_is_not_direct_dependency():
    removed_direct_dependencies = {
        "accelerate",
        "gradio",
        "huggingface_hub",
        "langchain_huggingface",
        "sentence_transformers",
        "torch",
        "transformers",
    }
    direct_dependency_names = {
        _pkg_name(dependency)
        for dependency in pyproject()["project"]["dependencies"]
    }

    assert direct_dependency_names.isdisjoint(removed_direct_dependencies)
    assert {"fastapi", "uvicorn"}.issubset(direct_dependency_names)


def test_uv_lock_matches_project_metadata():
    project = pyproject()
    locked_project = locked_project_package()

    assert uv_lock()["requires-python"].replace(" ", "") == project["project"][
        "requires-python"
    ].replace(" ", "")
    assert sorted(
        locked_dependency_strings(locked_project["metadata"]["requires-dist"])
    ) == sorted(project["project"]["dependencies"])
    assert sorted(
        locked_dependency_strings(
            locked_project["metadata"]["requires-dev"]["dev"]
        )
    ) == sorted(project["dependency-groups"]["dev"])


def test_locked_setup_rejects_stale_metadata_without_rewriting_lock(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None, "Run the project tests through uv."
    # A stale real-project lock can require registry metadata to re-resolve.
    # Use a dependency-free fixture so this check also works with CI's empty
    # cache and disabled network. Real lock metadata is checked separately.
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[project]\nname = "loopwright-lock-check"\nversion = "0.1.0"\n'
        'requires-python = ">=3.11,<3.13"\ndependencies = []\n'
    )
    # Ignore developer uv overrides that could redirect the project or disable
    # locking. The subprocess must check this disposable project, not the repo.
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("UV_") and key != "VIRTUAL_ENV"
    }
    isolated_options = [
        "--offline", "--cache-dir", str(tmp_path / "uv-cache"),
        "--python", sys.executable,
    ]
    locked = subprocess.run(
        [uv, "lock", *isolated_options], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert locked.returncode == 0, locked.stderr
    original_lock = (tmp_path / "uv.lock").read_bytes()
    command = [
        uv, "sync", "--locked", "--no-dev", "--no-install-project",
        "--dry-run", *isolated_options,
    ]
    valid = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert valid.returncode == 0, valid.stderr
    manifest.write_text(
        manifest.read_text().replace(
            'version = "0.1.0"',
            'version = "999.0.0"',
            1,
        )
    )
    stale = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert stale.returncode != 0
    assert "lockfile" in stale.stderr and "--locked" in stale.stderr
    assert (tmp_path / "uv.lock").read_bytes() == original_lock


def workflow(name):
    # BaseLoader preserves GitHub's `on` key instead of parsing it as a YAML 1.1
    # boolean. Expressions are also intentionally left as strings.
    return yaml.load(
        (PROJECT_ROOT / ".github" / "workflows" / name).read_text(),
        Loader=yaml.BaseLoader,
    )


@pytest.mark.parametrize(
    "name", ["tests.yml", "dependency-audit.yml", "docker-publish.yml"]
)
@pytest.mark.parametrize("dependency_file", ["pyproject.toml", "uv.lock"])
def test_dependency_changes_trigger_pull_request_validation(name, dependency_file):
    trigger = workflow(name)["on"]["pull_request"]
    assert "main" in trigger["branches"]
    assert dependency_file in trigger["paths"]


def test_docker_publication_remains_restricted_to_main_pushes():
    config = workflow("docker-publish.yml")
    assert config["on"]["push"]["branches"] == ["main"]
    steps = config["jobs"]["build-and-push"]["steps"]
    login = next(step for step in steps if "docker/login-action@" in step.get("uses", ""))
    build = next(step for step in steps if "docker/build-push-action@" in step.get("uses", ""))
    main_push = "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    assert login["if"] == main_push
    assert build["with"]["push"] == "${{ " + main_push + " }}"
