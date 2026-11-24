"""実行時の来歴(コード・環境)の記録

取得できない項目は捏造せず、null と理由を記録する。
git の情報は呼び出し側(実行スクリプト)が run_fig6._get_git_info で取って渡す。
"""
from __future__ import annotations

import datetime
import platform
import sys
from typing import Optional


def _module_version(name: str) -> dict:
    try:
        module = __import__(name)
    except Exception as exc:  # noqa: BLE001 - 未導入でも記録を続ける
        return {"version": None, "reason": f"import 失敗: {exc}"}
    version = getattr(module, "__version__", None)
    if version is None:
        return {"version": None, "reason": "__version__ がありません"}
    return {"version": str(version), "reason": None}


def collect_provenance(git_info: Optional[dict], cli_argv: Optional[list],
                       config_path: Optional[str]) -> dict:
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "cli_argv": cli_argv,
        "cli_argv_reason": None if cli_argv is not None else "関数の直接呼び出しのため引数列不明",
        "config_path": config_path,
        "config_path_reason": None if config_path is not None else "設定ファイルパス不明",
        "git": git_info if git_info is not None else {
            "commit_sha": None, "commit_sha_reason": "git 情報が渡されなかった",
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "packages": {name: _module_version(name) for name in ("numpy", "scipy", "yaml", "matplotlib")},
        },
    }
