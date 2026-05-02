1.  **Refactor `_build_manifest_lookup` in `huggingface_models.py`**
    - Replace `for entry in os.listdir(cache_dir):` with `with os.scandir(cache_dir) as entries:` and `for entry in entries:`.
    - Change `if not entry.startswith("manifest=")` to use `entry.name.startswith`.
    - Change `os.path.join(cache_dir, entry)` to use `entry.path`.

2.  **Refactor `get_llama_cpp_models_from_cache` in `huggingface_models.py`**
    - Replace `for entry in os.listdir(cache_dir):` with `with os.scandir(cache_dir) as entries:` and `for entry in entries:`.
    - Use `entry.name` for string checks (`.endswith`).
    - Use `entry.path` instead of `os.path.join(cache_dir, entry)`.
    - Use `entry.is_file()` instead of `os.path.isfile(file_path)`.
    - Use `entry.stat().st_size` instead of `os.path.getsize(file_path)`.

3.  **Refactor `get_llamacpp_language_models_from_llama_cache` in `huggingface_models.py`**
    - Replace `for entry in os.listdir(cache_dir):` with `with os.scandir(cache_dir) as entries:` and `for entry in entries:`.
    - Use `entry.name` for string checks.
    - Use `entry.path` instead of `os.path.join(cache_dir, entry)`.
    - Use `entry.is_file()` instead of `os.path.isfile(file_path)`.

4.  **Refactor `has_cache` in `hf_cache.py`**
    - Replace `for revision in os.listdir(snapshots_dir):` with `with os.scandir(snapshots_dir) as entries:` and `for entry in entries:`.
    - Use `entry.path` instead of `os.path.join(snapshots_dir, revision)`.
    - Use `entry.is_dir()` instead of `os.path.isdir(revision_path)`.

5.  **Refactor `get_llamacpp_language_models_from_llama_cache` in `huggingface_models.py`**
    - Re-verify `test_integrations` tests pass after modifications.

6.  **Create Journal Entry**
    - Document the learning in `.jules/bolt.md` using `>>` to append. The learning should note that replacing `os.listdir` with `os.scandir` when subsequent `stat` or `is_file`/`is_dir` calls are needed reduces overhead by accessing cached filesystem metadata directly.

7.  **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
    - Run `uv run ruff format src/nodetool/integrations/huggingface/`
    - Run `uv run ruff check src/nodetool/integrations/huggingface/`
    - Run tests using `uv run pytest tests/integrations/ -v`.

8.  **Submit PR**
    - Submit the changes using the exact title format `"⚡ Bolt: [performance improvement]"` and include `💡 What`, `🎯 Why`, `📊 Impact`, and `🔬 Measurement` in the description.
