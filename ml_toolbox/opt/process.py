# -*- coding: utf-8 -*-
"""ProcessObjective（评估侧接入）：外部程序本身就是黑盒目标。

"每跑一次得一条信息"的主形态：参数写进命令行/配置文件 -> 启动进程 ->
从 stdout 正则或结果文件解析分数。与 ML 完全无关。

约定：
- 非零退出码 / 超时 / 解析不到 -> 抛异常，runner 记 status='failed'（P10）；
  失败现场（命令 + stdout/stderr 尾部）存 last_error，事后排错用；
- 参数注入模板：`{key}` 占位（str.format），值经 str() 转换；
  模板里可直接写格式说明符（`{amp:.4f}`），或构造时传 fmt={"amp": ".4f"}
  让裸 `{amp}` 用默认格式（真实模拟软件常拒绝 17 位小数）；
- 需要 json 文件时用 {params_file}（写临时文件后给路径）；
- 编码：子进程输出按 utf-8 解码 + errors=replace（Windows 管道默认
  GBK 会把模拟软件的 UTF-8 中文输出炸成 UnicodeDecodeError）。
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time

import numpy as np

from .contracts import Objective, ParamSpace

_LOG = logging.getLogger("ml_toolbox.opt.process")

# 分数解析兜底：支持科学计数法（1.23e-4 / -5.6E+2）
_SCORE_RE = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


class ProcessObjective(Objective):
    """cmd_template 占位符：{key}（各参数）、{params_file}（json 路径）。

    parse:        stdout 正则（含 1 个捕获组）；默认匹配 "score: <数>"。
    result_file:  非空则从该文件读 json 取 score_key；支持 {key} 占位
                  （模拟软件按参数命名输出文件的情形）。
    fmt:          {参数名: 格式说明符}，如 {"amp": ".4f"}——模板里裸
                  {amp} 渲染时套用；模板已写 {amp:.4f} 则模板优先。
    """

    def __init__(self, cmd_template: str, space: ParamSpace,
                 parse: str = r"score[:=]\s*" + _SCORE_RE,
                 cwd: str = "", result_file: str = "",
                 score_key: str = "score",
                 fmt: dict | None = None, env: dict | None = None,
                 name: str = "process", minimize: bool = True,
                 timeout: float = 120.0, shell: bool = True,
                 constraint_signal: str = "",
                 on_infeasible: str = "censor", penalty_mult: float = 10.0):
        """on_infeasible: 约束违反点的处理策略（需配 constraint_signal 识别）。
            censor    -> 记 infeasible，喂 GP-BO 的 cEI 分类器（constrain 开关）
            penalize  -> 返回自适应惩罚分（最差可行分×penalty_mult），
                         目标 GP 全量点可学，薄可行带下通常比 cEI 省评估
            raise     -> 当普通失败（failed，censored；默认无 constraint_signal 时）
        """
        super().__init__(name=name, space=space, minimize=minimize,
                         has_constraints=bool(constraint_signal))
        self.cmd_template = cmd_template
        self.parse = re.compile(parse)
        self.cwd = cwd or None
        self.result_file = result_file
        self.score_key = score_key
        self.fmt = dict(fmt or {})
        self.env_extra = dict(env or {})
        self.timeout = float(timeout)
        self.shell = shell
        self.constraint_signal = constraint_signal  # 识别"约束拒绝"的信号子串
        self.on_infeasible = on_infeasible
        self.penalty_mult = float(penalty_mult)
        self._worst_ok = -float("inf")      # 历史最差可行分（自适应惩罚基准）
        self.last_cmd = ""                 # UI 展示/排错用
        self.last_error = ""               # 最近一次失败现场（含输出尾部）

    def _penalty(self) -> float:
        base = self._worst_ok if np.isfinite(self._worst_ok) and self._worst_ok > 0 \
            else 1.0
        return base * self.penalty_mult + 1.0

    def evaluate(self, params: dict):
        try:
            s = self._run_once(params)
            if np.isfinite(s):
                self._worst_ok = max(self._worst_ok, s)
            return s
        except Exception as e:
            if (self.on_infeasible == "penalize"
                    and self.is_constraint_error(e)):
                return self._penalty()
            raise

    def is_constraint_error(self, exc: Exception) -> bool:
        """黑盒主动拒绝（约束违反）vs 真崩溃：按信号子串区分。

        例如模拟软件对违反 π 脉冲条件的输入以退出码 3 + 特定 stderr 拒绝，
        传 constraint_signal="违反" 即可把这些点标为 infeasible（喂 cEI 分类器），
        而超时/段错误等仍算 failed（censored）。
        """
        return bool(self.constraint_signal
                    and self.constraint_signal in str(exc))

    # ------------------------------------------------ 渲染
    def _render(self, tpl: str, fmt: dict) -> str:
        """模板渲染：裸 {key} 套用 self.fmt[key] 默认格式说明符。"""
        def repl(m):
            key, spec = m.group(1), m.group(2)
            if spec is None:
                spec = self.fmt.get(key, "")
            v = fmt[key]
            return format(v, spec) if spec else str(v)
        return re.sub(r"\{(\w+)(?::([^{}]*))?\}", repl, tpl)

    # ------------------------------------------------ 主流程（真正的进程执行）
    def _run_once(self, params: dict):
        fmt = dict(params)
        tmp = None
        try:
            if "{params_file}" in self.cmd_template:
                fd, tmp = tempfile.mkstemp(suffix=".json", prefix="optp_")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(params, f, ensure_ascii=False)
                fmt["params_file"] = tmp.replace("\\", "/")   # 模板内统一正斜杠
            cmd = self._render(self.cmd_template, fmt)
            self.last_cmd = cmd
            env = None
            if self.env_extra:
                env = dict(os.environ)
                env.update(self.env_extra)
            try:
                r = subprocess.run(cmd, capture_output=True, cwd=self.cwd,
                                   timeout=self.timeout, shell=self.shell,
                                   env=env)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"超时 {self.timeout}s（可调 timeout 或减小单次仿真规模）")
            out = self._dec(r.stdout)
            err = self._dec(r.stderr)
            if r.returncode != 0:
                raise RuntimeError(f"退出码 {r.returncode}: {(err or out)[-400:]}")
            if self.result_file:
                rf = self._render(self.result_file, fmt)
                with open(rf, encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                return float(data[self.score_key])
            m = self.parse.search(out)
            if not m:
                raise RuntimeError(f"stdout 未匹配解析式: {out[-400:]}")
            return float(m.group(1))
        except Exception as e:
            self.last_error = f"cmd={self.last_cmd}\n{type(e).__name__}: {e}"
            raise
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    @staticmethod
    def _dec(b) -> str:
        """bytes -> str：utf-8 优先，GBK 兜底（Windows 控制台程序常见）。"""
        if not b:
            return ""
        if isinstance(b, str):
            return b
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return b.decode("gbk", errors="replace")

    # ------------------------------------------------ 身份指纹
    @staticmethod
    def _py_paths(cmd: str) -> list:
        """从命令模板提取 .py 路径：先取引号包裹的（可含空格），再取裸 token。

        裸 token 用 `[\w./\\-]+`（不含空格）——否则 `python scripts\a.py`
        会被连读成 `python scripts\a.py` 整串，join 后不存在，
        内容哈希退化成 missing 占位 = 黑盒改代码检测不到（N2 失效）。
        """
        found, rest = [], cmd
        for m in list(re.finditer(r'"([^"]+\.py)"', cmd)):
            found.append(m.group(1))
            rest = rest.replace(m.group(0), " ")
        found += re.findall(r"([\w./\\-]+\.py)", rest)
        return sorted(set(found))

    @staticmethod
    def _hash_scripts(cmd: str, cwd: str) -> str:
        """cmd 模板里出现的 .py 路径逐个做内容哈希（黑盒改一行 -> 指纹变）。

        文件读不到（相对路径歧义/跨机器）时以占位串参与哈希：宁可指纹不同
        触发校验提示，不可静默相同放行脏历史。
        """
        import hashlib
        h = hashlib.sha256()
        for m in ProcessObjective._py_paths(cmd):
            p = m if os.path.isabs(m) else os.path.join(cwd or ".", m)
            try:
                with open(p, "rb") as f:
                    h.update(f.read())
            except OSError:
                h.update(b"<missing:" + p.encode("utf-8", "replace") + b">")
        return h.hexdigest()[:12]

    def _fp_payload(self) -> dict:
        return {"cmd": self.cmd_template, "cwd": self.cwd or "",
                "parse": self.parse.pattern,
                "result_file": self.result_file, "score_key": self.score_key,
                "constraint_signal": self.constraint_signal,
                "on_infeasible": self.on_infeasible,
                "scripts": self._hash_scripts(self.cmd_template, self.cwd or "")}

    def fingerprint(self) -> str:
        import hashlib
        import json
        base = super().fingerprint()
        extra = json.dumps(self._fp_payload(), sort_keys=True,
                           ensure_ascii=False, default=str)
        return hashlib.sha256((base + extra).encode("utf-8")).hexdigest()[:16]


class InfeasiblePoint(RuntimeError):
    """约束违反（黑盒输出判定）：该点不可行，但是有用信息。"""


def _set_nested(d: dict, path: str, value):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


class BatchProcessObjective(Objective):
    """一次进程吃一批点的黑盒（pts.json 进 / res.json 出 + 多字段解析）。

    与 ProcessObjective 的分工：后者 = "一个点一次进程 + stdout 一个数"；
    本类 = "一批点一次（或几次）进程 + 结果文件多字段"，专为昂贵黑盒的
    并行评估设计（runner workers>1 时经 evaluate_many 进入）。

    约定（freeze 黑盒同款接口）：
    - cmd_template 含 {points_file}/{out_file} 占位；每个子进程吃一个
      单点 pts.json（黑盒按点切分到多进程是交接方的推荐并行方式）；
    - point_map: {参数名: 点内路径}，路径支持 "ell.alpha_deg" 式嵌套；
      point_extra: 固定字段（如 {"pol": "椭圆偏", "fwhm_ns": 40.0}）；
    - score_field: 目标字段名；constraints: [{"field","op","value"}]
      任一不满足 -> 该点不可行（on_infeasible 决定 censor/penalize）；
    - ok=false 的行 = 真失败（censored），与约束违反区分；
    - stagger: 相邻子进程启动间隔秒（SQLite 等共享资源的首次访问竞争）；
    - evaluate_many 返回的分数**已是统一最小化方向**（框架契约）。
    """

    def __init__(self, cmd_template: str, space: ParamSpace, *,
                 cwd: str = "", point_map: dict | None = None,
                 point_extra: dict | None = None,
                 score_field: str = "C_dual", minimize: bool = True,
                 constraints: list | None = None,
                 on_infeasible: str = "penalize", penalty_mult: float = 10.0,
                 stagger: float = 4.0, timeout: float = 7200.0,
                 name: str = "batch_process"):
        super().__init__(name=name, space=space, minimize=minimize,
                         has_constraints=bool(constraints))
        self.cmd_template = cmd_template
        self.cwd = cwd or None
        self.point_map = dict(point_map or {})
        self.point_extra = dict(point_extra or {})
        self.score_field = score_field
        self.constraints = list(constraints or [])
        self.on_infeasible = on_infeasible
        self.penalty_mult = float(penalty_mult)
        self.stagger = float(stagger)
        self.timeout = float(timeout)
        self._worst_ok = -float("inf")     # 历史最差可行分（最小化方向取 max）
        self.last_error = ""

    # ------------------------------------------------ 点构造 / 解析
    def _to_point(self, params: dict, pid) -> dict:
        pt = {"id": pid}
        pt.update(self.point_extra)
        pm = self.point_map or {k: k for k in params}   # 无映射时参数名即字段名
        for k, path in pm.items():
            if k in params:
                _set_nested(pt, path, params[k])
        return pt

    @staticmethod
    def _get_nested(d: dict, path: str):
        for k in path.split("."):
            d = d[k]
        return d

    def _violations(self, row: dict) -> list:
        bad = []
        for c in self.constraints:
            try:
                v = self._get_nested(row, c["field"])
            except (KeyError, TypeError):
                continue                     # 字段缺失不判（黑盒版本差异容错）
            op = c.get("op", ">=")
            ok = {"<=": v <= c["value"], ">=": v >= c["value"],
                  "<": v < c["value"], ">": v > c["value"],
                  "==": v == c["value"]}.get(op, True)
            if not ok:
                bad.append(f"{c['field']}={v:.4g} {op} {c['value']} 不满足")
        return bad

    def _penalty(self) -> float:
        base = self._worst_ok if np.isfinite(self._worst_ok) and self._worst_ok > 0 \
            else 1.0
        return base * self.penalty_mult + 1.0

    def is_constraint_error(self, exc: Exception) -> bool:
        """串行路径的约束识别：evaluate 抛的 InfeasiblePoint = 约束违反。

        不识别的话 runner 会记 failed（censored），cEI 分类器拿不到
        不可行样本（架构审视 M13 指出的真缺陷）。
        """
        return isinstance(exc, InfeasiblePoint)

    # ------------------------------------------------ 单进程求值
    def _popen(self, cmd: str):
        import subprocess
        flags = 0
        if os.name == "nt":
            # 独立进程组：父进程被强杀时不连带；kill_all 可整树清理
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.cwd, shell=True, creationflags=flags)

    @staticmethod
    def _kill_tree(proc):
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(f"taskkill /F /T /PID {proc.pid}",
                               capture_output=True, shell=True, timeout=10)
            else:
                proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def kill_all(self):
        """中止所有在飞子进程（GUI 停止/关窗时调用）。"""
        for p in list(getattr(self, "_procs", [])):
            self._kill_tree(p)

    def _run_point(self, params: dict, pid: int):
        """-> (score_min_dir, status)。status: ok|infeasible|failed。"""
        pt = self._to_point(params, pid)
        tmpdir = tempfile.mkdtemp(prefix="optb_")
        pf = os.path.join(tmpdir, "pts.json")
        of = os.path.join(tmpdir, "res.json")
        proc = None
        try:
            with open(pf, "w", encoding="utf-8") as f:
                json.dump([pt], f, ensure_ascii=False)
            cmd = (self.cmd_template.replace("{points_file}", pf.replace("\\", "/"))
                   .replace("{out_file}", of.replace("\\", "/")))
            proc = self._popen(cmd)
            self._procs = getattr(self, "_procs", [])
            self._procs.append(proc)
            _LOG.info("黑盒启动 pid=%s params=%s", proc.pid,
                      {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in params.items()})
            try:
                out_b, err_b = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self._kill_tree(proc)
                self.last_error = f"超时 {self.timeout}s: {cmd}"
                _LOG.warning("黑盒超时 pid=%s（%.0fs）", proc.pid, self.timeout)
                return np.inf, "failed"
            finally:
                try:
                    self._procs.remove(proc)
                except ValueError:
                    pass
            if not os.path.exists(of):
                tail = ProcessObjective._dec(err_b)[-300:]
                self.last_error = f"无输出文件；rc={proc.returncode} {tail}"
                # rc=-1(0xFFFFFFFF) 多为进程被外部强杀；stderr 尾巴是定位关键，必须进日志
                _LOG.warning("黑盒无输出 rc=%s pid=%s stderr_tail=%r",
                             proc.returncode, proc.pid, tail)
                return np.inf, "failed"
            with open(of, encoding="utf-8", errors="replace") as f:
                rows = json.load(f)
            row = rows[0] if isinstance(rows, list) and rows else rows
            if not row.get("ok", True):
                self.last_error = f"黑盒报失败: {row.get('err', row)}"
                _LOG.warning("黑盒报失败 pid=%s err=%s", proc.pid,
                             row.get("err", "?"))
                return np.inf, "failed"
            bad = self._violations(row)
            if bad:
                _LOG.info("黑盒完成 pid=%s 约束违反=%s", proc.pid, bad)
                if self.on_infeasible == "penalize":
                    s = float(row.get(self.score_field, np.nan))
                    if np.isfinite(s):
                        v = s if self.minimize else -s
                        self._worst_ok = max(self._worst_ok, v)
                    return self._penalty(), "ok"   # 惩罚分可微地喂 GP
                return np.inf, "infeasible"
            s = float(row[self.score_field])
            v = s if self.minimize else -s
            self._worst_ok = max(self._worst_ok, v)
            _LOG.info("黑盒完成 pid=%s %s=%.4f", proc.pid, self.score_field, s)
            return v, "ok"
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            _LOG.exception("黑盒评估异常")
            return np.inf, "failed"
        finally:
            for p in (pf, of):
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

    # ------------------------------------------------ 主接口
    def evaluate(self, params: dict) -> float:
        """单点 = 长度 1 的批。返回原始方向分数（infeasible 抛异常）。"""
        v, st = self._run_point(params, 0)
        if st == "failed":
            raise RuntimeError(f"黑盒评估失败: {self.last_error}")
        if st == "infeasible":
            raise InfeasiblePoint("约束违反")
        return v if self.minimize else -v

    def fingerprint(self) -> str:
        """批量黑盒指纹：cmd 模板 + 点构造映射 + 解析/约束 + 脚本内容哈希。

        历史数据复用的安全阀——warm_start 前比对源 run 存档的指纹，
        黑盒脚本（如 freeze_blackbox.py）改一行即判定不同源。
        """
        import hashlib
        import json
        base = Objective.fingerprint(self)
        payload = {
            "cmd": self.cmd_template, "cwd": self.cwd or "",
            "point_map": self.point_map, "point_extra": self.point_extra,
            "score_field": self.score_field, "constraints": self.constraints,
            "on_infeasible": self.on_infeasible,
            "scripts": ProcessObjective._hash_scripts(
                self.cmd_template, self.cwd or "")}
        extra = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                           default=str)
        return hashlib.sha256((base + extra).encode("utf-8")).hexdigest()[:16]

    def evaluate_many(self, plist: list) -> list:
        """并行一批：每点一个子进程，错峰启动（ThreadPool 只管调度，
        真并行在子进程层——黑盒内部单线程，无 GIL/BLAS 干扰）。"""
        from concurrent.futures import ThreadPoolExecutor
        n = len(plist)
        done = [0]

        def _progress(p):
            done[0] += 1
            cb = getattr(self, "on_progress", None)
            if cb:
                try:
                    cb(done[0], n, p)
                except Exception:
                    pass

        if n == 1:
            r = [self._run_point(plist[0], 0)]
            _progress(plist[0])
            return r
        results: list = [None] * n

        def go(i):
            if self.stagger > 0:
                time.sleep(i * self.stagger)
            results[i] = self._run_point(plist[i], i)
            _progress(plist[i])

        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(go, range(n)))
        for i in range(n):
            if results[i] is None:            # 线程异常兜底
                results[i] = (np.inf, "failed")
        return results
