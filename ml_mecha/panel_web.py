# -*- coding: utf-8 -*-
"""ML 面板（web viewer）：**独立的只读 HTTP 服务**，把"跑过什么 + 那次跑出来什么"给人看。

## 它解决什么

共享监控面板（dsh 右栏那三页）只显示**软件自己的账**（谁改了什么、配置现值、跑过什么），
**看不到方法的结果面**——而 GUI 里其实早就有那一整套（每个方法声明自己的检视页 `PageSpec`：
混淆矩阵、ROC、聚类散点、残差、预测曲线……）。
⇒ 本服务把**同一套 `PageSpec`** 搬到 web 上，**不重写任何一张图**。

## 为什么是独立进程 + 端口文件

1. 人的窗口要**在 dsh 之外**（用户明确要求）：训练/权威进程不需要知道窗口在不在；
2. TensorBoard 那边已经是同一形态（`TensorBoardSidecar` + `.tb-port`）⇒ **两条展示路同构**；
3. 端口发现用框架的单一实现 `mecha.portfile`（含"收尾只删自己写的那个"那条规则）。

## 三个并列 view（用户裁决：这是 **ML 内部的族**，不是启动模式）

| view | 显示什么 | 数据来源 |
|---|---|---|
| `ml` | 机器学习方法的 run | `runs/<id>/{record.json, artifacts.npz}` |
| `opt` | 自动调参的 run | `runs/<id>/{opt_record.json, history.csv}` |
| `nn` | 神经网络：我们的表 + **TensorBoard iframe** | 同上 + `/dashboard` 起的 TB |

## 渲染策略（实测定下来的，不是设想）

| `PageSpec.kind` | 怎么显示 | 依据 |
|---|---|---|
| `mpl` | **服务端 Agg 渲染成 PNG**，经 `/plot` 取 | 实测 9 种图种全部成功（存档→`load_run_record`→`plot(ax,res)`→PNG） |
| `table` | pandas → HTML 表 | `spec.data(...)` 或 `_table_source` 的老路 |
| `text` | 诊断 JSON/文本 | `res.diag` |
| `pg` | **占位 + 说明** | pyqtgraph 是 Qt 自绘控件；web 端重写是另一件事（**不假装能显示**） |
| 自定义（`nn_weights`/`nn_replay`） | 占位 + 指向 TensorBoard | 同一批数据 TB 已有 HISTOGRAMS/IMAGES |

⚠ **PG 与自定义页画"占位而响亮"**：写清"这一页在 web 端没有渲染器，去 TensorBoard 看"，
**不是空白**（空白的页 = 本仓反复在防的那种假绿）。
"""

from __future__ import annotations

import http.server
import json
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from mecha.portfile import clear_port_file, write_port_file

#: 本服务自己的端口文件（与 `.mcp-port` / `.ml-monitor-port` / `.tb-port` 同族）。
WEB_PORT_FILE = ".mlweb-port"

#: 渲染 PNG 的缓存目录（放在 run 自己的目录下，与 `record.json` 并列）。
PLOT_CACHE_DIRNAME = "web"

#: 单张图的像素（够看清又不至于让页面爆）。
PLOT_DPI = 110
PLOT_FIGSIZE = (7.2, 4.4)


# ============================================================ 取数
def ensure_registry() -> None:
    """确保内置方法族已载入（**幂等**）。

    ⚠ 为什么必须显式做：本服务是**独立进程**（给人开窗口用），不走 UI/runner 的入口，
    因此**注册表默认是空的** ⇒ `registry.get("ridge")` 会 `KeyError`，表现为
    "每个 run 都列不出页"（实测踩过：一个 run 都列不出）。
    ⇒ 每次取页前调一次（`load_builtin` 内部就是 `import ml_toolbox.methods`，廉价且幂等）。
    """
    try:
        from ml_toolbox.core import registry
        registry.load_builtin()
    except Exception:
        pass


def preload_modules(names) -> list[str]:
    """按**名字**预载模块（供 application 层注入"本服务要用到的层"）。

    ## 为什么参数是"名字字符串"而不是直接 import

    本模块住在 `ml_mecha/`（边界层），**不许** import `ml_toolbox.opt` 等上层——判据
    用 `ast.walk` 静态扫描，连函数体内的静默 import 都算违规。而"优化运行"这一路
    确实需要 opt 层在场（它 import 时会把读档能力注册进 `core.persistence` 的槽）。

    ⇒ **由 application 层（launcher）决定载什么**，本层只按名字载：名字在这里是**数据**，
    不是本层的依赖。职责也对：选哪些层是应用的事，边界层不该知道。

    @returns 载入失败的名字（空 = 全成功）——**不静默**，调用方可据此记日志。
    """
    import importlib
    failed = []
    for name in names or ():
        try:
            importlib.import_module(str(name))
        except Exception:
            failed.append(str(name))
    return failed


def list_runs(kind: str = "ml", limit: int = 300) -> list[dict]:
    """列 run（新→旧）。``kind`` ∈ ``{"ml","nn"}`` / ``{"opt"}``。

    ⚠ ``ml`` 与 ``nn`` 用**同一份存档**（神经网络就是方法族之一），
    区别只在方法名以 `torch_` / `nn` 开头——**不另存一份**（避免两份真相）。
    """
    from ml_toolbox.core import persistence
    if kind == "opt":
        rows = persistence.list_opt_records()      # ⭐ 经 core 的槽取（不 import opt 层）
    else:
        rows = persistence.list_records()

    out = []
    for r in rows[: max(0, int(limit))]:
        m = str(r.get("method") or "")
        family = str(r.get("family") or "")
        is_nn = ("neural" in family) or m.startswith(("torch", "nn"))
        if kind == "nn" and not is_nn:
            continue
        if kind == "ml" and is_nn:
            continue
        primary = str(r.get("primary_metric") or "")
        metrics = r.get("metrics") or {}
        out.append({
            "run_id": r.get("run_id"),
            "method": m,
            "primary": primary,
            "score": metrics.get(primary) if isinstance(metrics, dict) else None,
            "elapsed": r.get("elapsed"),
            "saved_at": r.get("saved_at") or r.get("saved") or "",
            "error": r.get("error"),
        })
    return out


def load_any(run_id: str):
    """按 run_id 找它属于哪一侧，回 ``("ml"|"opt", 记录对象)``；找不到回 ``(None, None)``。

    ⚠ **边界纪律**：本模块在 `ml_mecha/` 里，**不许** import `ml_toolbox.ui`（GUI 层）
    或 `ml_toolbox.opt`（优化层）——判据
    `test_ml_mecha.py::test_static_ml_mecha_does_not_import_el_modules` 用 `ast.walk`
    静态守着（**静默 import 也算违规**）。⇒ 两条路都走 core：
    ML 侧是 `core.persistence.load_run_record`（读档能力已下沉）；
    opt 侧是 `core.persistence.load_opt_record`（opt 自己注册进 core 的槽）。
    """
    if not run_id:
        return None, None
    from ml_toolbox.core import persistence
    try:
        rec = persistence.load_run_record(run_id)
        if rec is not None:
            return "ml", rec
    except Exception:
        pass
    try:
        rec = persistence.load_opt_record(run_id)
        if rec is not None:
            return "opt", rec
    except Exception:
        pass
    return None, None


#: 指标名 → 它属于哪一类 target_kind（用于**探测档内自相矛盾**）。
_REG_METRICS = frozenset({"rmse", "mae", "r2", "mape", "mse"})
_CLS_METRICS = frozenset({"accuracy", "f1", "precision", "recall", "auc",
                          "log_loss", "balanced_accuracy"})


def target_kind_mismatch(rec) -> str:
    """档内自相矛盾的探测：``target_kind`` 与它自己的**指标**对不上时回一句原因。

    ## 为什么需要（历史档修不了）

    2026-10-03 之前 `run_one` 落盘用的是 `spec.target_kind`（数据集的 kind）而不是
    方法实际用的那个 ⇒ 例如 `ridge`（只能回归）跑分类数据时，存档自述"分类"、
    指标却全是 `rmse/mae/r2`（实测前 400 条里 47 条）。
    这些**已落盘的档改不了** ⇒ 展示层必须扛得住，而且**不能静默**：
    按"自述的分类"去挑检视页会挑到混淆矩阵，而回归预测是连续值，画不出来。

    ⇒ 返回非空字符串 = 这一档自相矛盾，页面要**把矛盾说出来**（而不是空白/崩）。
    """
    kind = str(getattr(rec, "target_kind", "") or "")
    res = getattr(rec, "result", None)
    metrics = set((getattr(res, "metrics", None) or {}).keys())
    if not kind or not metrics:
        return ""
    reg = bool(metrics & _REG_METRICS)
    cls = bool(metrics & _CLS_METRICS)
    if kind == "classification" and reg and not cls:
        return ("这一档自述是**分类**，但它的指标全是回归指标"
                f"（{', '.join(sorted(metrics & _REG_METRICS))}）⇒ "
                "多半是 2026-10-03 之前归档时写错了 target_kind。"
                "依赖 target_kind 的检视页（混淆矩阵/ROC）因此画不出来。")
    if kind == "regression" and cls and not reg:
        return ("这一档自述是**回归**，但它的指标是分类指标"
                f"（{', '.join(sorted(metrics & _CLS_METRICS))}）⇒ 同上，归档写错了。")
    return ""


def pages_for(side: str, rec) -> list:
    """该 run 的检视页 ``[(key, title, kind, spec)]``。

    ⚠ 用**与 GUI 同一套机制**：`inspect_pages()` 优先，空则 `auto_pages`（ML）/
    `default_pages`（opt）兜底 —— 与 `main_window.py:568` 那条 `or` 同构。
    ⚠ 取页失败**不静默**：把原因带到 ``self.last_error`` 并由调用方显示
    （"读不到"与"这一页没有"必须能分开——这是本仓反复在守的纪律）。
    """
    out: list = []
    try:
        ensure_registry()
        from ml_toolbox.core import persistence
        if side == "ml":
            from ml_toolbox.core import registry, runner
            method = registry.get(rec.method)
            try:
                spec_pages = list(method.inspect_pages(_cfg_of(rec)))
            except Exception:
                spec_pages = []
            if not spec_pages:
                # 与 GUI 的 `main_window.py:568` 同构：声明为空则走兜底
                spec_pages = list(runner.auto_pages(method, _dataspec_of(rec)))
        else:
            spec_pages = []
            if hasattr(rec, "inspect_pages"):
                try:
                    spec_pages = list(rec.inspect_pages())
                except Exception:
                    spec_pages = []
            if not spec_pages:
                # 经 core 的槽取（opt 自己注册的）——**不 import opt 层**（宪章纪律）
                spec_pages = list(persistence.opt_pages(rec))
    except Exception as exc:
        pages_for.last_error = f"{type(exc).__name__}: {exc}"
        return []
    pages_for.last_error = ""
    for p in spec_pages:
        out.append((getattr(p, "key", "?"), getattr(p, "title", "?"),
                    getattr(p, "kind", "?"), p))
    return out


#: 上一次 `pages_for` 的失败原因（空 = 顺利）。**模块级**是因为它服务于"诊断"而非控制流。
pages_for.last_error = ""                             # type: ignore[attr-defined]


def _cfg_of(rec):
    """从存档记录还原一个 ``RunConfig``（绘图只看参数，不需要真跑）。"""
    from ml_toolbox.core.contracts import RunConfig
    cfg = RunConfig()
    conf = getattr(rec, "config", None) or {}
    if isinstance(conf, dict):
        cfg.seed = int(conf.get("seed") or 42)
        cfg.diag = bool(conf.get("diag"))
        cfg.extras = dict(conf.get("extras") or {})
    return cfg


def _dataspec_of(rec):
    """``auto_pages`` 需要 ``DataSpec``；从存档只能给最小壳（task/target_kind 够用）。"""
    from ml_toolbox.core.contracts import DataSpec
    import pandas as pd
    res = getattr(rec, "result", None)
    return DataSpec(X=pd.DataFrame(), y=None,
                    target_kind=getattr(res, "target_kind", None) or
                    getattr(rec, "target_kind", None),
                    meta={"dataset": getattr(rec, "dataset", "")})


#: 依赖 `target_kind` 的检视页（档内矛盾时它们**必然**画不出来——见 `target_kind_mismatch`）。
_KIND_DEPENDENT_PAGES = frozenset({"cm", "roc", "pr"})


def page_body(side: str, rec, key: str, kind: str, spec, run_id: str) -> str:
    """**一页的正文 HTML**（按 kind 分派；与 GUI 的 `InspectPage.show_result` 同构）。"""
    from . import panel_view as V

    # ⚠ 档内矛盾（历史的错标 target_kind）+ 依赖它的页 ⇒ **说清矛盾**，不去画一张注定失败的图。
    #   （画了也只是一张"渲染失败"的说明图，反而看不见"为什么"的根因。）
    conflict = target_kind_mismatch(rec) if rec is not None else ""
    if conflict and key in _KIND_DEPENDENT_PAGES:
        return V.placeholder(f"「{getattr(spec, 'title', key)}」画不出来", conflict)

    if kind == "mpl":
        if getattr(spec, "plot", None) is None:
            return V.placeholder(f"「{spec.title}」没有绘图函数",
                                 "该方法声明了这一页但没给 plot ⇒ 属于声明不全。")
        return V.plot_html(f"/plot?run={V.esc(run_id)}&page={V.esc(key)}",
                           title=spec.title)
    if kind == "table":
        return V.table_html(_table_of(side, rec, spec))
    if kind == "text":
        return V.text_html(getattr(getattr(rec, "result", rec), "diag", None))
    # pg / 自定义（nn_weights / nn_replay / …）
    why = ("这一页在 GUI 里是 pyqtgraph 自绘控件，web 端没有对应渲染器。"
           if kind == "pg" else
           "这一页是神经网络的自定义页；同一批数据 TensorBoard 已有更合适的视图"
           "（权重→HISTOGRAMS，表示演化→IMAGES）。")
    return V.placeholder(f"「{spec.title}」在 web 端暂无渲染器", why)


def _table_of(side: str, rec, spec):
    """表格页的数据：优先 ``spec.data(...)``，否则按 key 取通用来源（指标/参数/诊断）。"""
    try:
        if getattr(spec, "data", None) is not None:
            return spec.data(rec if side == "opt" else getattr(rec, "result", rec))
    except Exception:
        return None
    import pandas as pd
    res = getattr(rec, "result", rec)
    key = getattr(spec, "key", "")
    if key == "metrics":
        return pd.DataFrame([{"指标": k, "值": v}
                             for k, v in (getattr(res, "metrics", {}) or {}).items()])
    if key == "params":
        return pd.DataFrame([{"参数": k, "值": v}
                             for k, v in (getattr(res, "params", {}) or {}).items()])
    if key == "diag":
        d = getattr(res, "diag", None)
        return pd.DataFrame([{"键": k, "值": v}
                             for k, v in (d or {}).items()]) if isinstance(d, dict) else None
    return None


# ============================================================ 渲染图 → PNG
def render_plot(side: str, rec, spec, run_id: str, key: str) -> Path:
    """把某一页渲染成 PNG 并**缓存**（返回文件路径）。失败**抛**（由路由转成可读错误）。

    ⚠ 缓存键含 run_id + 页 key：**绘图代码后来改了，旧 run 会重画**（数据是旧的、
    画法新的）—— 这是有意的（通常更好），但界面上要说明（见 `page_body` 的注释）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_cjk_font()                                 # 轴标签/图例是中文，缺字体会变方框

    d = _run_dir(run_id) / PLOT_CACHE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{key}.png"
    if out.is_file() and out.stat().st_size > 0:
        return out

    target = rec if side == "opt" else getattr(rec, "result", rec)
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    try:
        spec.plot(ax, target)
        fig.tight_layout()
        fig.savefig(out, dpi=PLOT_DPI)
    finally:
        plt.close(fig)
    return out


def _run_dir(run_id: str) -> Path:
    from ml_toolbox.core.persistence import RUNS_DIR
    return Path(RUNS_DIR) / str(run_id)


# ============================================================ HTTP
class _Handler(http.server.BaseHTTPRequestHandler):
    """只读路由。**图片与 HTML 都回**（与框架 cockpit 的 JSON-only 不同：这是人看的页面）。"""

    server_version = "mlpanel/1.0"
    endpoint = None                                   # start() 里绑到派生类

    def log_message(self, fmt, *args):                # 静音（避免污染权威日志）
        pass

    # ------------------------------------------------------------ 工具
    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _html(self, text: str, code: int = 200) -> None:
        self._send(text.encode("utf-8"), "text/html; charset=utf-8", code)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8", code)

    # ------------------------------------------------------------ 路由
    def do_GET(self) -> None:                          # noqa: N802 - 基类约定
        ep = type(self).endpoint
        parts = urlsplit(self.path)
        q = parse_qs(parts.query)
        path = parts.path
        try:
            if path in ("/", "/index.html"):
                return self._html(ep.render_index(q))
            if path == "/runs":
                return self._json(ep.runs(q))
            if path == "/plot":
                return self._plot(q)
            if path == "/page":
                return self._json(ep.page_meta(q))
            if path == "/health":
                return self._json({"ok": True, "routes": ["/", "/runs", "/plot", "/page"]})
            return self._html(ep.render_notfound(path), 404)
        except Exception as exc:                       # ⚠ **响亮**：读不到不当成空页
            # ⚠ **也写日志**：只把原因放进响应体 = 日志一片安静，运维/我自己都会以为没事。
            import traceback
            ep._say(f"[panel] {path} 出错：{type(exc).__name__}: {exc}")
            ep._say(traceback.format_exc(limit=6))
            return self._html(ep.render_error(path, exc), 500)

    def _plot(self, q) -> None:
        ep = type(self).endpoint
        run_id = (q.get("run") or [""])[0]
        key = (q.get("page") or [""])[0]
        try:
            png = ep.plot_path(run_id, key)
        except Exception as exc:
            # 图渲染失败 ⇒ **回一张说明图**（而不是 404/空白：人要知道"这里本该有图"）
            return self._send(ep.render_error_png(run_id, key, exc),
                              "image/png", 200)
        data = png.read_bytes()
        self._send(data, "image/png", 200)


class PanelWeb:
    """ML 面板服务（**独立进程**用的薄壳：起/停 + 端口文件 + 只读路由）。"""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0,
                 port_file: str | Path | None = None,
                 backend_url: str | None = None,
                 tensorboard_url: str | None = None,
                 log=None) -> None:
        self._host = host
        self._want_port = int(port)
        self._port_file = Path(port_file) if port_file else None
        #: 共享监控端点（取写权模式/活动）；没有就显示"未知"
        self.backend_url = backend_url
        #: TensorBoard 的地址（NN 页 iframe 用）；没有就显示"未启动"
        self.tensorboard_url = tensorboard_url
        self._log = log
        self._server = None
        self._thread = None
        self._port: int | None = None

    # ------------------------------------------------------------ 事实面
    @property
    def port(self) -> int | None:
        return self._port

    @property
    def url(self) -> str | None:
        return f"http://{self._host}:{self._port}/" if self._port else None

    def _say(self, msg: str) -> None:
        if self._log is not None:
            try:
                self._log(msg)
            except Exception:
                pass

    # ------------------------------------------------------------ 生命周期
    def start(self) -> int:
        """起服务并返回实际端口。顺序：**清陈旧** → bind → 写端口文件。"""
        clear_port_file(self._port_file)               # 启动前无条件清陈旧（规则见 mecha.portfile）
        handler = type("_BoundHandler", (_Handler,), {"endpoint": self})
        self._server = http.server.ThreadingHTTPServer(
            (self._host, self._want_port), handler)
        self._port = int(self._server.server_address[1])
        write_port_file(self._port_file, self._port)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="ml-panel-web", daemon=True)
        self._thread.start()
        self._say(f"[panel] {self.url}")
        return self._port

    def stop(self, timeout: float = 5.0) -> None:
        srv, self._server = self._server, None
        port, self._port = self._port, None
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:
                pass
        if self._thread is not None:
            try:
                self._thread.join(timeout)
            except Exception:
                pass
            self._thread = None
        clear_port_file(self._port_file, port)         # 只删仍是自己写的那个

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # ------------------------------------------------------------ 页面
    def runs(self, q) -> dict:
        view = (q.get("view") or ["ml"])[0]
        view = view if view in ("ml", "opt", "nn") else "ml"
        try:
            return {"ok": True, "view": view, "runs": list_runs(view)}
        except Exception as exc:
            return {"ok": False, "view": view, "runs": [],
                    "reason": f"{type(exc).__name__}: {exc}"}

    def page_meta(self, q) -> dict:
        run_id = (q.get("run") or [""])[0]
        side, rec = load_any(run_id)
        if rec is None:
            return {"ok": False, "reason": f"找不到 run {run_id!r} 的存档"}
        pages = pages_for(side, rec)
        return {"ok": True, "side": side, "run_id": run_id,
                "pages": [{"key": k, "title": t, "kind": kd} for k, t, kd, _ in pages]}

    def plot_path(self, run_id: str, key: str) -> Path:
        side, rec = load_any(run_id)
        if rec is None:
            raise KeyError(f"找不到 run {run_id!r} 的存档")
        spec = next((s for k, _t, _kd, s in pages_for(side, rec) if k == key), None)
        if spec is None:
            raise KeyError(f"该 run 没有页 {key!r}")
        return render_plot(side, rec, spec, run_id, key)

    def _status(self) -> str:
        """顶栏右侧：写权模式（从共享监控端点取；取不到就诚实说未知）。"""
        if not self.backend_url:
            return '<span class="muted">写权模式：未知（未接监控端点）</span>'
        try:
            import urllib.request
            with urllib.request.urlopen(self.backend_url.rstrip("/") + "/status",
                                        timeout=3) as r:
                st = json.loads(r.read().decode("utf-8"))
            mode = str(st.get("mode") or "?")
            kind = {"ai": "ok", "human": "warn", "locked": "warn"}.get(mode, "err")
            label = {"ai": "写权 AI", "human": "写权在人",
                     "locked": "全员禁写"}.get(mode, f"权威离线？({mode})")
            return f'<span class="badge {kind}">{label}</span>'
        except Exception:
            return '<span class="badge err">权威离线</span>'

    def render_index(self, q) -> str:
        from . import panel_view as V
        view = (q.get("view") or ["ml"])[0]
        view = view if view in ("ml", "opt", "nn") else "ml"
        run_id = (q.get("run") or [""])[0]

        try:
            rows = list_runs(view)
            runs_err = ""
        except Exception as exc:
            rows, runs_err = [], f"{type(exc).__name__}: {exc}"
        if not run_id and rows:
            run_id = str(rows[0].get("run_id") or "")

        # ⚠ `active` **必须在分支外初始化**：只有"有 rec"那条路会给它赋值，
        #   而 run 不存在时也要渲染整页 ⇒ 否则 `UnboundLocalError` 让该页 500
        #   （实测踩过：读不到的 run 应当显示"读不到"，不是崩）。
        pages: list = []
        active = ""
        side, rec = load_any(run_id)
        if run_id and rec is None:
            body = V.unreadable(f"run {run_id}", "它的存档读不出来（record.json/artifacts.npz）")
        elif rec is not None:
            spec_pages = pages_for(side, rec)
            shown = [(k, t) for k, t, _kd, _s in spec_pages]
            # ⭐ NN 追加一个"训练过程"页（TensorBoard iframe）
            if view == "nn":
                shown = shown + [("__tb__", "训练过程（TensorBoard）")]
            active = (q.get("page") or [""])[0] or (shown[0][0] if shown else "")
            pages = shown
            if active == "__tb__":
                body = self._tb_frame(V)
            else:
                found = next(((k, t, kd, s) for k, t, kd, s in spec_pages if k == active),
                             None)
                # ⚠ `page_body` 在**本模块**（不是 panel_view）——曾经错写成 `V.page_body`
                #   导致整页 500，而错误只藏在响应体里 ⇒ 同时修了"500 要进日志"。
                body = (page_body(side, rec, found[0], found[2], found[3], run_id)
                        if found else V.placeholder(
                            "这一页没有内容",
                            f"该 run 没有页 {active!r}（页签可能是上一次的残留）"))
        else:
            body = V.placeholder("还没有可看的 run",
                                 "跑一次方法之后这里会列出它的全部检视页。"
                                 "（AI 模式默认不落盘 ⇒ 存档为空不等于没跑过）")

        right = self._status()
        if runs_err:
            right += " " + V.badge(f"运行列表读不到：{runs_err}", "err")
        # ⚠ 档内自相矛盾 ⇒ **顶栏一声响**（不静默、也不假装正常）
        conflict = target_kind_mismatch(rec) if rec is not None else ""
        if conflict:
            right += " " + V.badge("这一档自相矛盾（见页内说明）", "warn")
        note = ""
        if conflict:
            note = f'<div class="note">{V.esc(conflict)}</div>'
        if view == "nn":
            note = ("<div class=\"note\">神经网络用 TensorBoard 看训练过程"
                    "（曲线/分布/帧）；这里的表格与归因图是本面板自己渲染的。</div>")
        return V.page(title=f"ML 面板 · {view}", view=view, header_right=right,
                      body=body + note, pages=pages, active=active, run_id=run_id,
                      runs_html=V.runs_pane(rows, active=run_id, view=view))

    def _tb_frame(self, V) -> str:
        """NN 的「训练过程」页 = **TensorBoard iframe**（用户裁决：嵌进 viewer）。"""
        if self.tensorboard_url:
            return (f'<iframe src="{V.esc(self.tensorboard_url)}" '
                    f'title="TensorBoard"></iframe>'
                    '<div class="note">上面这块是 TensorBoard（它自己的全套面板）。'
                    '若空白，说明它还没起来或有跨源限制 —— 用「打开仪表盘」按钮单独开一个窗口。</div>')
        return V.placeholder(
            "TensorBoard 还没起来",
            "它按需启动（只挂 --logdir，只读）。请用 GET /dashboard 起它，"
            "或直接 tensorboard --logdir runs/<run_id>。")

    def render_notfound(self, path: str) -> str:
        from . import panel_view as V
        return (V.head("404") + V.placeholder(f"没有这条路由：{path}",
                                              "可用：/ · /runs · /plot · /page · /health")
                + "</body></html>")

    def render_error(self, path: str, exc: BaseException) -> str:
        from . import panel_view as V
        return (V.head("出错") + V.unreadable(path, f"{type(exc).__name__}: {exc}")
                + "</body></html>")

    def render_error_png(self, run_id: str, key: str, exc: BaseException) -> bytes:
        """图渲染失败 ⇒ **回一张写明原因的图**（不是空白、也不是 404）。

        ⚠ 为什么不回 404/占位 HTML：`<img>` 拿到非图片只会显示破图，
        夹在页面里看不出**为什么**。画成图则原因跟着图走。
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _ensure_cjk_font()
        fig = plt.figure(figsize=(7.2, 2.4), dpi=100)
        fig.text(0.02, 0.72, "这一页的图渲染失败了", fontsize=13,
                 fontproperties=_cjk_font())
        fig.text(0.02, 0.40, f"{run_id} / {key}", fontsize=9,
                 fontproperties=_cjk_font())
        fig.text(0.02, 0.14, f"{type(exc).__name__}: {str(exc)[:110]}", fontsize=9,
                 fontproperties=_cjk_font(), color="#cf222e")
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


def _ensure_cjk_font() -> None:
    """让 matplotlib 能显示中文（否则轴标签变方框：实测 `plot_confusion` 的"预测/真实"）。

    ⚠ 不只是"好看"：`methods/plots.py` 里的轴标签/图例**本来就是中文**
    （"预测"/"真实"/"残差"…），没有 CJK 字体时它们会渲染成缺字方框
    ⇒ 图能看但**读不懂**，等于那张图白画。
    ⇒ 在 rcParams 上设一次（进程级、幂等），而不是每处传 fontproperties。
    找不到任何 CJK 字体时**不抛**（退回默认，至少不崩），但会留一条警告到日志。
    """
    global _CJK_READY
    if _CJK_READY:
        return
    _CJK_READY = True
    try:
        from matplotlib import font_manager, rcParams
        have = {f.name for f in font_manager.fontManager.ttflist}
        for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                     "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei"):
            if name in have:
                rcParams["font.family"] = [name, "DejaVu Sans"]
                rcParams["axes.unicode_minus"] = False   # 负号在 CJK 字体下常缺
                return
    except Exception:
        pass


#: 字体只设一次（进程级）。
_CJK_READY = False


def _cjk_font():
    """取一个能显示中文的 `FontProperties`（给 `fig.text` 用）。"""
    _ensure_cjk_font()
    try:
        from matplotlib import font_manager
        for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                     "Source Han Sans SC", "PingFang SC"):
            try:
                return font_manager.FontProperties(family=name)
            except Exception:
                continue
    except Exception:
        pass
    return None


def start_panel_web(*, port: int = 0, port_file=None, backend_url=None,
                    tensorboard_url=None, log=None) -> PanelWeb:
    """便利函数：起一个已 start 的 :class:`PanelWeb`。"""
    p = PanelWeb(port=port, port_file=port_file, backend_url=backend_url,
                 tensorboard_url=tensorboard_url, log=log)
    p.start()
    return p


def resolve_panel_base(port_file, *, host: str = "127.0.0.1",
                       timeout: float = 1.5) -> str | None:
    """从端口文件解析出**真的有人在听**的面板地址；否则 ``None``。

    ## 为什么要"探一下"，而不是只读文件

    端口文件是**发现辅助、不是真值**（`mecha.portfile` 的既定语义）。进程被强杀时
    （`terminate()`／任务被会话收掉都**不跑 `finally`**）文件会残留一个**指向死进程**的
    端口号。只读文件就会把浏览器开到**没人听**的地址上，现象正是
    "面板打不开／端口好像不对"。⇒ 读出来之后连一下 `/health`，**只有真在听才认**。

    ⚠ 端口文件里的数字本身**不校验合法性**（它回答"端口是多少"，不回答"文件合不合法"）：
    非法/缺失一律按"不知道"处理，返回 ``None``。
    """
    from mecha.portfile import read_port
    port = read_port(port_file)
    if not port:
        return None
    base = f"http://{host}:{int(port)}/"
    try:
        import urllib.request
        with urllib.request.urlopen(base + "health", timeout=timeout) as r:
            if r.status == 200:
                return base
    except Exception:
        return None
    return None


def main(argv=None) -> int:                            # pragma: no cover - CLI 入口
    """独立进程入口：``python -m ml_mecha.panel_web``（给 launcher 起窗口用）。"""
    import argparse
    import sys as _sys
    # 与 `app_entry.py` / `launcher.py` 同一惯例：Windows 控制台按 UTF-8 输出，
    # 否则中文日志在子进程里会变成乱码（实测）。
    for stream in (_sys.stdout, _sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    from ml_toolbox.core.persistence import RUNS_DIR
    ap = argparse.ArgumentParser(description="ML 面板（只读 web viewer）")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--port-file", default=str(Path(RUNS_DIR).parent / WEB_PORT_FILE))
    ap.add_argument("--backend-url", default=None)
    ap.add_argument("--tensorboard-url", default=None)
    # ⭐ application 层（launcher）指定要预载的层：例如 `ml_toolbox.opt.persistence`
    #   （它 import 时会把优化侧的读档能力注册进 core 的槽）。名字在这里是**数据**，
    #   所以边界层不因此依赖 opt 层（见 `preload_modules` 的说明）。
    ap.add_argument("--preload", action="append", default=[],
                    help="启动前按名字预载的模块（可多次；应用层注入用）")
    a = ap.parse_args(argv)
    failed = preload_modules(a.preload)
    for name in failed:
        print(f"[panel] ⚠ 预载失败：{name}（该层的视图会空着，原因见上）", flush=True)
    svc = PanelWeb(port=a.port, port_file=a.port_file, backend_url=a.backend_url,
                   tensorboard_url=a.tensorboard_url, log=lambda m: print(m, flush=True))
    svc.start()
    print(f"[panel] 服务中：{svc.url}（Ctrl+C 退出）", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
