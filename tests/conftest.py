import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _k in ("DEEPSEEK_API_KEY", "GLM_PLAN_API_KEY", "OPENCODE_GO_API_KEY"):
    os.environ[_k] = "test-" + _k.lower()
os.environ["THALAMUS_ADMIN_TOKEN"] = "admin-test-token"


import pytest


@pytest.fixture(autouse=True)
def _isolate_runtime_env():
    """每个用例前把运行期配置重置成测试值。

    `test_eval.test_scripts_import_cleanly` 会 import 每个 scripts/*.py，
    而 manage.py / calibrate.py 顶部都调 `load_env(ROOT/".env")` —— 那会把**生产**
    .env 的变量注入测试进程（THALAMUS_TOTP_ONLY=1 之类），于是后面所有用 bearer
    头的用例都被 require_admin 拒掉。这里每例前重置，把这个副作用挡在用例之外。
    """
    for _k in ("THALAMUS_TOTP_SECRET", "THALAMUS_TOTP_ONLY"):
        os.environ.pop(_k, None)
    os.environ["THALAMUS_ADMIN_TOKEN"] = "admin-test-token"
    yield
