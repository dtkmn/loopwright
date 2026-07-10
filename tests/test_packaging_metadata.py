import re
import tomllib
from pathlib import Path


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
    assert (PROJECT_ROOT / ".env.example").exists()


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
