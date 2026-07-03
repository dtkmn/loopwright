import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def requirement_lines(path: str) -> tuple[str, ...]:
    lines = []
    for raw_line in (PROJECT_ROOT / path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        lines.append(line)
    return tuple(lines)


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


def test_runtime_requirements_match_pyproject_dependencies():
    assert tuple(pyproject()["project"]["dependencies"]) == requirement_lines(
        "requirements.txt"
    )


def test_dev_requirements_match_pyproject_dev_group():
    assert tuple(pyproject()["dependency-groups"]["dev"]) == requirement_lines(
        "requirements-dev.txt"
    )


def test_console_scripts_are_declared():
    scripts = pyproject()["project"]["scripts"]
    assert scripts["loopwright"] == "src.app:main"
    assert scripts["ai-loop-engine"] == "src.app:main"
    assert scripts["loopwright-eval"] == "src.loop_eval:main"
    assert scripts["ai-loop-eval"] == "src.loop_eval:main"
    assert scripts["loopwright-ollama-eval"] == "src.ollama_model_eval:main"
    assert scripts["ai-loop-ollama-eval"] == "src.ollama_model_eval:main"


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
    assert (PROJECT_ROOT / ".env.example").exists()


def test_removed_model_stack_is_not_direct_dependency():
    removed_direct_dependencies = {
        "accelerate",
        "gradio",
        "huggingface-hub",
        "langchain-huggingface",
        "sentence-transformers",
        "torch",
        "transformers",
    }
    direct_dependency_names = {
        dependency.split("==", 1)[0]
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
