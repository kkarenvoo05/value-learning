"""vl.train_baseline.code_snapshot: the uncommitted code a run actually trained with."""

import hashlib
import subprocess


def git(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)


def test_code_snapshot_has_uncommitted_and_untracked_code_and_its_hash(tmp_path):
    from vl.train_baseline import code_snapshot
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("x = 1\n")
    (tmp_path / "notes.md").write_text("n\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    (tmp_path / "src/a.py").write_text("x = 2\n")             # tracked, modified
    (tmp_path / "src/new.py").write_text("y = 3\n")           # untracked
    (tmp_path / "notes.md").write_text("changed\n")           # outside src/configs/scripts
    text, sha = code_snapshot(tmp_path)
    assert "+x = 2" in text and "untracked src/new.py\ny = 3" in text and "changed" not in text
    assert sha == hashlib.sha256(text.encode()).hexdigest()
    (tmp_path / "src/new.py").write_text("y = 4\n")
    assert code_snapshot(tmp_path)[1] != sha
