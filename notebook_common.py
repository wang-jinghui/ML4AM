"""逐章 Notebook 的公共数据准备与实验辅助，不在导入时打印或配置环境。

完整性和低方差筛选仅拟合训练收益；保留原实验的样本方差 > 1e-12 口径。
依赖版本集中维护在 requirement.txt，不在 Notebook 中检测或打印。
"""
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skfolio.pre_selection import DropZeroVariance, SelectComplete
from threadpoolctl import threadpool_limits


@dataclass
class ExperimentData:
    """共享同一资产顺序的数据；完整资产清单和排除日期可按需查看。"""

    prices: pd.DataFrame
    X_train: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_test: pd.DataFrame
    eligible_count: int
    smoke: bool = False

    @property
    def assets(self) -> pd.Index:
        return self.X_train.columns

    @property
    def excluded_test_dates(self) -> pd.DatetimeIndex:
        return self.X_test_raw.index.difference(self.X_test.index)

    def summary(self) -> pd.Series:
        """只展示与实验相关的数据规模、时间范围和测试覆盖率。"""
        return pd.Series({
            '数据模式': '小规模验证' if self.smoke else '全部合格基金',
            '训练期合格基金数': self.eligible_count,
            '本次使用基金数': len(self.assets),
            '训练收益行数': len(self.X_train),
            '训练收益区间': f'{self.X_train.index[0]:%Y-%m-%d} 至 {self.X_train.index[-1]:%Y-%m-%d}',
            '测试候选区间': f'{self.X_test_raw.index[0]:%Y-%m-%d} 至 {self.X_test_raw.index[-1]:%Y-%m-%d}',
            '测试候选日期数': len(self.X_test_raw),
            '测试共同有效日期数': len(self.X_test),
            '排除测试日期数': len(self.excluded_test_dates),
            '测试日期覆盖率': f'{len(self.X_test) / len(self.X_test_raw):.2%}',
        }, name='数据概览')


def setup_notebook(seed: int = 42) -> None:
    """显式设置随机种子、单线程计算和中文绘图，不打印环境信息。"""
    np.random.seed(seed)
    threadpool_limits(limits=1)
    plt.rcParams.update({
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'figure.figsize': (10, 4),
    })
    pd.set_option('display.max_rows', 24)


def prepare_data(
    prices: pd.DataFrame,
    *,
    window: int = 1260,
    train_prices: int = 1008,
    variance_threshold: float = 1e-12,
    smoke: bool = False,
) -> ExperimentData:
    """按训练期筛选资产，测试期只筛选共同有效日期，不重新选择资产。

    输入为时间索引的价格表；保留时区并排序。非有限/非正价格先标记为缺失，
    pct_change(fill_method=None) 不跨缺失日期计算收益。
    SelectComplete 显式排除内部缺失；DropZeroVariance 的总体方差阈值
    换算为 variance_threshold * (n-1)/n，以保持原 ddof=1 的样本方差标准。
    smoke 仅用于自动化验证，正常调用使用全部合格基金。
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError('价格表必须使用时间索引')
    if prices.index.hasnans or not prices.index.is_unique or not prices.columns.is_unique:
        raise ValueError('日期和基金代码必须唯一，日期不能为空')
    if not 3 <= train_prices < window <= len(prices):
        raise ValueError('需满足 3 <= 训练价格数 < 窗口长度 <= 数据长度')
    if not np.isfinite(variance_threshold) or variance_threshold < 0:
        raise ValueError('方差阈值必须有限且非负')

    prices_window = prices.sort_index().tail(window)
    valid_prices = prices_window.where(np.isfinite(prices_window) & prices_window.gt(0))
    returns = valid_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    train_returns = returns.iloc[1:train_prices]

    complete = SelectComplete(drop_assets_with_internal_nan=True).set_output(transform='pandas')
    complete.fit(train_returns)
    if not complete.get_support().any():
        raise ValueError('训练期没有收益完整的基金')
    complete_train = complete.transform(train_returns)
    n = len(complete_train)
    variance = DropZeroVariance(threshold=variance_threshold * (n - 1) / n)
    variance.fit(complete_train)
    assets = complete_train.columns[variance.get_support()].sort_values()
    eligible_count = len(assets)
    if smoke:
        assets = assets[:24]
    if len(assets) < 4:
        raise ValueError('合格基金少于4只，无法运行当前章节对照')

    # 方差过滤器不接受测试段的 NaN，按训练期学得的列掩码固定资产集合。
    selected_returns = returns.loc[:, assets]
    X_train = selected_returns.iloc[1:train_prices].copy()
    X_test_raw = selected_returns.iloc[train_prices:].copy()
    X_test = X_test_raw.dropna(how='any').copy()
    if len(X_test) < 3:
        raise ValueError('共同有效的测试收益日期少于3天')
    return ExperimentData(
        prices=prices_window.loc[:, assets].copy(),
        X_train=X_train,
        X_test_raw=X_test_raw,
        X_test=X_test,
        eligible_count=eligible_count,
        smoke=smoke,
    )


def load_data(
    path: str | Path | None = None,
    *,
    window: int = 1260,
    train_prices: int = 1008,
    variance_threshold: float = 1e-12,
    smoke: bool | None = None,
) -> ExperimentData:
    """读取项目价格数据；ML4AM_SMOKE=1 仅用于不保存输出的冒烟测试。"""
    if path is None:
        path = Path(__file__).resolve().parent / 'data' / 'hs_funds_prices.parquet'
    if smoke is None:
        smoke = os.environ.get('ML4AM_SMOKE', '0') == '1'
    prices = pd.read_parquet(path).set_index('timestamp')
    return prepare_data(prices, window=window, train_prices=train_prices,
                        variance_threshold=variance_threshold, smoke=smoke)


def timed(label, function):
    """显示实验阶段及运行时间，返回结果和秒数。"""
    start = perf_counter()
    print(f'开始：{label}', flush=True)
    value = function()
    elapsed = perf_counter() - start
    print(f'完成：{label}，耗时 {elapsed:.2f} 秒', flush=True)
    return value, elapsed


def quiet(function):
    """压缩原算法的冗长标准输出；不拦截警告和异常。"""
    with redirect_stdout(io.StringIO()) as buffer:
        result = function()
    lines = buffer.getvalue().splitlines()
    if lines:
        print(f'原函数日志共{len(lines)}行，末尾：', '\n'.join(lines[-4:]))
    return result


def matrix_stats(matrix):
    """返回特征值、条件数及 Cholesky 诊断，不修复输入矩阵。"""
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    try:
        np.linalg.cholesky(matrix)
        positive_definite = True
    except np.linalg.LinAlgError:
        positive_definite = False
    return {'最小特征值': eigenvalues.min(), '最大特征值': eigenvalues.max(),
            '条件数': np.linalg.cond(matrix), '可Cholesky分解': positive_definite}
