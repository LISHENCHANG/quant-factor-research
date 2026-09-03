"""严格的通用因子回测引擎。

因子既可按 ``YYYY-MM-DD.csv`` 逐日保存，也可集中保存为一个 Parquet；
交易日历、可交易性、未来收益和市值暴露全部直接来自
``stocks_daily.parquet``。首次运行会自动建立按年份组织的轻量缓存，
后续回测无需反复扫描整个行情文件。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """数据格式或时间规则不符合严格回测要求。"""


@dataclass
class DataPaths:
    """因子源、Parquet 行情、缓存和输出路径。

    ``factor_dir`` 为兼容旧入口保留原名称，既可以是逐日 CSV 目录，
    也可以是包含 ``date/code/因子列`` 的单个 Parquet 文件。
    """

    factor_dir: Path
    market_parquet: Path
    output_dir: Path
    market_cache_dir: Path
    flags_dir: Path | None = None


@dataclass
class FactorSpec:
    """因子的名称、数据列名、预期方向及预处理状态。"""

    name: str
    column: str
    direction: str = "positive"
    preprocessed: bool = False


@dataclass
class BacktestConfig:
    """回测参数；execution_lag 至少为 1，防止同日信号同日成交。"""

    start_date: str | None = None
    end_date: str | None = None
    horizon: int = 1
    execution_lag: int = 1
    rebalance_step: int | None = None
    groups: int = 10
    cost_bps: float = 10.0
    min_stocks: int = 100
    tradability_mode: str = "basic"
    missing_return_policy: str = "zero"
    max_missing_return_ratio: float = 0.05
    mad_n: float = 5.0
    standardize: bool = True
    neutralize: bool = False
    neutralize_columns: tuple[str, ...] = ("log_market_cap",)
    market_exchanges: tuple[str, ...] = ("SH", "SZ")
    save_holdings: bool = False
    allow_exit_after_end: bool = False

    @property
    def effective_rebalance_step(self) -> int:
        return self.horizon if self.rebalance_step is None else self.rebalance_step

    def validate(self, paths: DataPaths, factor: FactorSpec) -> None:
        factor_source = Path(paths.factor_dir)
        if not factor_source.exists():
            raise FileNotFoundError(f"因子数据源不存在：{factor_source}")
        if factor_source.is_file() and factor_source.suffix.lower() != ".parquet":
            raise DataContractError("单文件因子源目前只支持 .parquet")
        if not Path(paths.market_parquet).is_file():
            raise FileNotFoundError(f"Parquet 行情不存在：{paths.market_parquet}")
        if self.execution_lag < 1:
            raise DataContractError("execution_lag 必须 >= 1，否则会产生同日成交偏差")
        if self.horizon < 1:
            raise DataContractError("horizon 必须 >= 1")
        if self.effective_rebalance_step < self.horizon:
            raise DataContractError("调仓间隔不能小于收益期限，否则持有期会重叠")
        if self.groups < 2:
            raise DataContractError("groups 必须 >= 2")
        if self.min_stocks < self.groups:
            raise DataContractError("min_stocks 不能小于 groups")
        if factor.direction not in {"positive", "negative"}:
            raise DataContractError("direction 只能是 positive 或 negative")
        if factor.preprocessed and self.neutralize:
            raise DataContractError(
                "preprocessed=True 时不能再次中性化；请输入原始因子或关闭 neutralize"
            )
        if self.tradability_mode not in {"basic", "none", "off"}:
            raise DataContractError("tradability_mode 目前只支持 basic/none/off")
        if self.missing_return_policy not in {"zero", "raise"}:
            raise DataContractError("missing_return_policy 只能是 zero 或 raise")
        if not 0 <= self.max_missing_return_ratio <= 1:
            raise DataContractError("max_missing_return_ratio 必须介于 0 和 1")
        supported_exposures = {"log_market_cap"}
        unsupported = set(self.neutralize_columns) - supported_exposures
        if self.neutralize and unsupported:
            raise DataContractError(
                f"Parquet 当前只支持市值中性化 log_market_cap，不支持：{sorted(unsupported)}"
            )
        supported_exchanges = {"SH", "SZ", "BJ"}
        unknown_exchanges = set(self.market_exchanges) - supported_exchanges
        if unknown_exchanges:
            raise DataContractError(f"未知交易所：{sorted(unknown_exchanges)}")


def available_dates(folder: Path) -> list[pd.Timestamp]:
    """只识别 YYYY-MM-DD.csv，并按交易日排序。"""

    dates: list[pd.Timestamp] = []
    for file in Path(folder).glob("*.csv"):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", file.stem):
            dates.append(pd.Timestamp(file.stem))
    return sorted(set(dates))


def trading_date_offset(
    calendar: list[pd.Timestamp], date: pd.Timestamp, offset: int
) -> pd.Timestamp | None:
    """使用 Parquet 实际交易日历计算 T+N，不按自然日推算。"""

    try:
        position = calendar.index(pd.Timestamp(date))
    except ValueError:
        return None
    target = position + offset
    return calendar[target] if 0 <= target < len(calendar) else None


def _convert_security_code(values: pd.Series) -> pd.Series:
    """统一为聚宽风格代码，兼容 .SH/.SZ/.BJ 和已转换代码。"""

    result = values.astype("string").str.strip()
    result = result.str.replace(r"\.SH$", ".XSHG", regex=True)
    result = result.str.replace(r"\.SZ$", ".XSHE", regex=True)
    result = result.str.replace(r"\.BJ$", ".XBEI", regex=True)
    return result


def _normalise_code(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "code" not in frame.columns:
        raise DataContractError(f"{label}缺少 code 列")
    result = frame.copy()
    result["code"] = _convert_security_code(result["code"])
    if (
        result["code"].isna().any()
        or result["code"].fillna("").eq("").any()
        or result["code"].duplicated().any()
    ):
        raise DataContractError(f"{label}的 code 存在空值或重复")
    return result


def read_date_csv(folder: Path, date: pd.Timestamp, label: str) -> pd.DataFrame:
    file = Path(folder) / f"{pd.Timestamp(date):%Y-%m-%d}.csv"
    if not file.exists():
        raise FileNotFoundError(f"{label}文件不存在：{file}")
    return _normalise_code(pd.read_csv(file, dtype={"code": str}), label)


class FactorDataSource:
    """统一读取逐日 CSV 或单个 Parquet，并按年份控制内存占用。"""

    OPTIONAL_ASOF_COLUMNS = ("asof_date", "factor_date")
    CACHE_VERSION = 1

    def __init__(
        self,
        source: Path,
        factor_column: str,
        start_date: str | None,
        end_date: str | None,
        cache_root: Path | None = None,
    ) -> None:
        self.source = Path(source)
        self.factor_column = factor_column
        self.mode = "daily_csv" if self.source.is_dir() else "parquet"
        self.cache_dir: Path | None = None

        if self.mode == "daily_csv":
            self.dates = available_dates(self.source)
        else:
            factor_cache_root = (
                Path(cache_root)
                if cache_root is not None
                else self.source.parent / ".factor_cache"
            )
            self._prepare_parquet_cache(factor_cache_root)

        if start_date:
            start = pd.Timestamp(start_date)
            self.dates = [date for date in self.dates if date >= start]
        if end_date:
            end = pd.Timestamp(end_date)
            self.dates = [date for date in self.dates if date <= end]
        if not self.dates:
            raise DataContractError("因子数据源在目标区间内没有有效日期")

    def _prepare_parquet_cache(self, cache_root: Path) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("读取 Parquet 因子需要 pyarrow") from error

        parquet_file = pq.ParquetFile(self.source)
        schema = parquet_file.schema_arrow
        schema_names = schema.names
        required = ["date", "code", self.factor_column]
        missing = [column for column in required if column not in schema_names]
        if missing:
            raise DataContractError(f"Parquet 因子缺少字段：{missing}")

        self.parquet_columns = required + [
            column for column in self.OPTIONAL_ASOF_COLUMNS if column in schema_names
        ]
        source_stat = self.source.stat()
        fingerprint_text = (
            f"v{self.CACHE_VERSION}|{self.source.resolve()}|{source_stat.st_size}|"
            f"{source_stat.st_mtime_ns}|{self.factor_column}"
        )
        fingerprint = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:16]
        self.cache_dir = cache_root / f"{self.source.stem}_{fingerprint}"
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            self._build_parquet_cache(source_stat)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.dates = [pd.Timestamp(date) for date in manifest["dates"]]

    def _build_parquet_cache(self, source_stat) -> None:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        assert self.cache_dir is not None
        self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
        building_dir = self.cache_dir.with_name(f"{self.cache_dir.name}.building")
        if building_dir.exists():
            shutil.rmtree(building_dir)
        building_dir.mkdir(parents=True)

        optional_columns = [
            column
            for column in self.OPTIONAL_ASOF_COLUMNS
            if column in self.parquet_columns
        ]
        cache_schema = pa.schema(
            [
                ("date", pa.timestamp("ns")),
                ("code", pa.string()),
                (self.factor_column, pa.float64()),
                *[(column, pa.timestamp("ns")) for column in optional_columns],
            ]
        )
        scanner = ds.dataset(self.source, format="parquet").scanner(
            columns=self.parquet_columns,
            batch_size=262_144,
            use_threads=True,
        )
        writers: dict[int, pq.ParquetWriter] = {}
        factor_dates: set[str] = set()
        total_rows = 0

        print(f"首次运行：正在为 {self.source.name} 建立年度因子缓存……")
        try:
            for batch_number, batch in enumerate(scanner.to_batches(), start=1):
                frame = batch.to_pandas()
                if pd.api.types.is_numeric_dtype(frame["date"]):
                    frame["date"] = pd.to_datetime(
                        frame["date"].astype("Int64").astype(str),
                        format="%Y%m%d",
                        errors="coerce",
                    )
                else:
                    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame = frame.dropna(subset=["date", "code"])
                frame["code"] = _convert_security_code(frame["code"])
                frame[self.factor_column] = pd.to_numeric(
                    frame[self.factor_column], errors="coerce"
                )
                for column in optional_columns:
                    frame[column] = pd.to_datetime(frame[column], errors="coerce")
                if frame["code"].fillna("").eq("").any():
                    raise DataContractError("Parquet 因子的 code 存在空值")

                factor_dates.update(frame["date"].dt.strftime("%Y-%m-%d"))
                for year, year_frame in frame.groupby(frame["date"].dt.year, sort=False):
                    year = int(year)
                    writer = writers.get(year)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            building_dir / f"{year}.parquet",
                            cache_schema,
                            compression="snappy",
                            use_dictionary=["code"],
                            write_statistics=True,
                        )
                        writers[year] = writer
                    table = pa.Table.from_pandas(
                        year_frame[cache_schema.names],
                        schema=cache_schema,
                        preserve_index=False,
                    )
                    writer.write_table(table)
                    total_rows += len(year_frame)

                if batch_number % 10 == 0:
                    print(f"  已缓存 {total_rows:,} 行因子记录")
        finally:
            for writer in writers.values():
                writer.close()

        if not writers:
            raise DataContractError("Parquet 因子没有有效记录")
        manifest = {
            "cache_version": self.CACHE_VERSION,
            "source": str(self.source.resolve()),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "factor_column": self.factor_column,
            "cached_rows": total_rows,
            "years": sorted(writers),
            "dates": sorted(factor_dates),
        }
        (building_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        building_dir.replace(self.cache_dir)
        print(f"年度因子缓存建立完成：{self.cache_dir}")

    @lru_cache(maxsize=2)
    def _load_year(self, year: int) -> pd.DataFrame:
        assert self.cache_dir is not None
        path = self.cache_dir / f"{year}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["date", "code", self.factor_column]).set_index(
                "date"
            )
        frame = pd.read_parquet(path)
        if frame.duplicated(["date", "code"]).any():
            raise DataContractError(f"年度因子缓存 {year} 存在重复 date-code")
        return frame.set_index("date").sort_index()

    def read_date(self, date: pd.Timestamp) -> pd.DataFrame:
        date = pd.Timestamp(date)
        if self.mode == "daily_csv":
            return read_date_csv(self.source, date, "因子")
        frame = self._load_year(date.year)
        try:
            daily = frame.loc[[date]].reset_index(drop=True)
        except KeyError:
            raise FileNotFoundError(f"Parquet 因子缺少日期：{date:%Y-%m-%d}")
        return _normalise_code(daily, f"Parquet 因子 {date:%Y-%m-%d}")


def assert_entry_after_signal(signal_date: pd.Timestamp, entry_date: pd.Timestamp) -> None:
    if pd.Timestamp(entry_date) <= pd.Timestamp(signal_date):
        raise DataContractError(
            f"建仓日 {entry_date:%Y-%m-%d} 必须晚于信号日 {signal_date:%Y-%m-%d}"
        )


def validate_factor_asof(frame: pd.DataFrame, signal_date: pd.Timestamp) -> None:
    """如果因子表提供了数据截止日，禁止其晚于信号日。"""

    for column in ("asof_date", "factor_date"):
        if column in frame.columns:
            asof = pd.to_datetime(frame[column], errors="coerce")
            if asof.isna().any():
                raise DataContractError(f"{column} 中存在无法解析的日期")
            if (asof > signal_date).any():
                raise DataContractError(
                    f"{signal_date:%Y-%m-%d} 因子文件使用了未来的 {column}"
                )


class ParquetMarketData:
    """Parquet 行情访问层：首次建年度缓存，之后按年份懒加载。"""

    SOURCE_COLUMNS = [
        "股票代码",
        "交易日",
        "开盘价",
        "最高价",
        "最低价",
        "收盘价",
        "成交量（手）",
        "成交额（千元）",
        "总市值（万元）",
        "复权因子",
    ]

    CACHE_VERSION = 1

    def __init__(
        self,
        source: Path,
        cache_root: Path,
        exchanges: tuple[str, ...],
    ) -> None:
        try:
            import pyarrow  # noqa: F401
        except ImportError as error:
            raise ImportError("读取 Parquet 需要 pyarrow，请在当前 Python 环境安装") from error

        self.source = Path(source).resolve()
        self.exchanges = tuple(sorted(set(exchanges)))
        source_stat = self.source.stat()
        fingerprint_text = (
            f"v{self.CACHE_VERSION}|{self.source}|{source_stat.st_size}|"
            f"{source_stat.st_mtime_ns}|{','.join(self.exchanges)}"
        )
        fingerprint = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:16]
        self.cache_dir = Path(cache_root) / f"{self.source.stem}_{fingerprint}"
        self.manifest_path = self.cache_dir / "manifest.json"
        if not self.manifest_path.exists():
            self._build_cache(source_stat)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.date_counts = {
            int(date): int(count)
            for date, count in self.manifest["date_counts"].items()
        }

    @staticmethod
    def _cache_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("date", pa.int32()),
                ("code", pa.string()),
                ("open", pa.float64()),
                ("close", pa.float64()),
                ("high", pa.float64()),
                ("low", pa.float64()),
                ("volume", pa.float64()),
                ("adjusted_vwap", pa.float64()),
                ("log_market_cap", pa.float64()),
            ]
        )

    def _build_cache(self, source_stat) -> None:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
        building_dir = self.cache_dir.with_name(f"{self.cache_dir.name}.building")
        if building_dir.exists():
            shutil.rmtree(building_dir)
        building_dir.mkdir(parents=True)

        print(f"首次运行：正在为 {self.source.name} 建立年度回测缓存……")
        dataset = ds.dataset(self.source, format="parquet")
        missing = [name for name in self.SOURCE_COLUMNS if name not in dataset.schema.names]
        if missing:
            raise DataContractError(f"Parquet 缺少字段：{missing}")

        scanner = dataset.scanner(
            columns=self.SOURCE_COLUMNS,
            batch_size=262_144,
            use_threads=True,
        )
        schema = self._cache_schema()
        writers: dict[int, pq.ParquetWriter] = {}
        date_counts: Counter[int] = Counter()
        total_rows = 0
        exchange_pattern = tuple(f".{exchange}" for exchange in self.exchanges)

        try:
            for batch_number, batch in enumerate(scanner.to_batches(), start=1):
                raw = batch.to_pandas()
                raw = raw[raw["股票代码"].astype(str).str.endswith(exchange_pattern)].copy()
                if raw.empty:
                    continue

                date_values = pd.to_numeric(raw["交易日"], errors="coerce")
                volume = pd.to_numeric(raw["成交量（手）"], errors="coerce")
                money = pd.to_numeric(raw["成交额（千元）"], errors="coerce")
                adjustment = pd.to_numeric(raw["复权因子"], errors="coerce")
                market_cap = pd.to_numeric(raw["总市值（万元）"], errors="coerce")

                with np.errstate(divide="ignore", invalid="ignore"):
                    adjusted_vwap = money * 10.0 / volume * adjustment
                    log_market_cap = np.log(market_cap.where(market_cap > 0))

                frame = pd.DataFrame(
                    {
                        "date": date_values,
                        "code": _convert_security_code(raw["股票代码"]),
                        "open": pd.to_numeric(raw["开盘价"], errors="coerce"),
                        "close": pd.to_numeric(raw["收盘价"], errors="coerce"),
                        "high": pd.to_numeric(raw["最高价"], errors="coerce"),
                        "low": pd.to_numeric(raw["最低价"], errors="coerce"),
                        "volume": volume,
                        "adjusted_vwap": adjusted_vwap,
                        "log_market_cap": log_market_cap,
                    }
                ).dropna(subset=["date", "code"])
                frame["date"] = frame["date"].astype("int32")
                frame = frame.replace([np.inf, -np.inf], np.nan)
                date_counts.update(frame["date"].value_counts().to_dict())

                for year, year_frame in frame.groupby(frame["date"] // 10_000, sort=False):
                    year = int(year)
                    writer = writers.get(year)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            building_dir / f"{year}.parquet",
                            schema,
                            compression="snappy",
                            use_dictionary=["code"],
                            write_statistics=True,
                        )
                        writers[year] = writer
                    table = pa.Table.from_pandas(
                        year_frame[schema.names], schema=schema, preserve_index=False
                    )
                    writer.write_table(table)
                    total_rows += len(year_frame)

                if batch_number % 10 == 0:
                    print(f"  已处理 {total_rows:,} 行沪深/北交所记录")
        finally:
            for writer in writers.values():
                writer.close()

        if not writers:
            raise DataContractError("Parquet 中没有选定交易所的数据")

        manifest = {
            "cache_version": self.CACHE_VERSION,
            "source": str(self.source),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "exchanges": list(self.exchanges),
            "cached_rows": total_rows,
            "years": sorted(writers),
            "date_counts": {
                str(int(key)): int(value)
                for key, value in sorted(date_counts.items())
            },
            "return_formula": (
                "adjusted_vwap = 成交额（千元）*10/成交量（手）*复权因子"
            ),
        }
        (building_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        building_dir.replace(self.cache_dir)
        print(f"年度缓存建立完成：{self.cache_dir}")

    def calendar(self, min_stocks: int) -> list[pd.Timestamp]:
        """剔除横截面过小的异常日期，例如节假日中的孤立记录。"""

        return [
            pd.Timestamp(str(date))
            for date, count in sorted(self.date_counts.items())
            if count >= min_stocks
        ]

    @lru_cache(maxsize=3)
    def _load_year(self, year: int) -> pd.DataFrame:
        path = self.cache_dir / f"{year}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=self._cache_schema().names).set_index(
                ["date", "code"]
            )
        frame = pd.read_parquet(path)
        if frame.duplicated(["date", "code"]).any():
            raise DataContractError(f"年度缓存 {year} 存在重复 date-code")
        return frame.set_index(["date", "code"]).sort_index()

    def read_date(self, date: pd.Timestamp) -> pd.DataFrame:
        date = pd.Timestamp(date)
        date_key = int(date.strftime("%Y%m%d"))
        frame = self._load_year(date.year)
        try:
            daily = frame.xs(date_key, level="date").reset_index()
        except KeyError:
            return pd.DataFrame(columns=["code", *self._cache_schema().names[2:]])
        return _normalise_code(daily, f"Parquet {date:%Y-%m-%d}")

    def read_tradability(
        self,
        mode: str,
        entry_date: pd.Timestamp,
        flags_dir: Path | None,
    ) -> pd.DataFrame | None:
        if mode in {"none", "off"}:
            return None
        daily = self.read_date(entry_date)
        required = ["open", "close", "high", "low", "volume", "adjusted_vwap"]
        numeric = daily[required].apply(pd.to_numeric, errors="coerce")
        filled = numeric.notna().all(axis=1) & (numeric > 0).all(axis=1)
        result = pd.DataFrame({"code": daily["code"], "filled": filled})
        if flags_dir is not None:
            flags = read_date_csv(flags_dir, entry_date, "交易标记")
            if "tradable" not in flags.columns:
                raise DataContractError("交易标记文件缺少 tradable 列")
            result = result.merge(flags[["code", "tradable"]], on="code", how="left")
            result["filled"] &= result["tradable"].fillna(False).astype(bool)
            result = result[["code", "filled"]]
        return result

    def read_exposures(
        self, date: pd.Timestamp, columns: tuple[str, ...]
    ) -> pd.DataFrame:
        daily = self.read_date(date)
        missing = [column for column in columns if column not in daily.columns]
        if missing:
            raise DataContractError(f"Parquet 暴露缺少列：{missing}")
        return daily[["code", *columns]].copy()

    def read_forward_returns(
        self,
        entry_date: pd.Timestamp,
        horizon: int,
        calendar: list[pd.Timestamp],
    ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
        exit_date = trading_date_offset(calendar, entry_date, horizon)
        if exit_date is None:
            return pd.DataFrame(columns=["code", "forward_return"]), None
        entry = self.read_date(entry_date)[["code", "adjusted_vwap"]].rename(
            columns={"adjusted_vwap": "entry_vwap"}
        )
        exit_frame = self.read_date(exit_date)[["code", "adjusted_vwap"]].rename(
            columns={"adjusted_vwap": "exit_vwap"}
        )
        returns = entry.merge(exit_frame, on="code", how="left", validate="one_to_one")
        valid = (
            returns["entry_vwap"].notna()
            & returns["exit_vwap"].notna()
            & (returns["entry_vwap"] > 0)
            & (returns["exit_vwap"] > 0)
        )
        returns["forward_return"] = np.nan
        returns.loc[valid, "forward_return"] = (
            returns.loc[valid, "exit_vwap"] / returns.loc[valid, "entry_vwap"] - 1.0
        )
        return returns[["code", "forward_return"]], exit_date


def process_factor_cross_section(
    factor_raw: pd.DataFrame,
    factor_column: str,
    mad_n: float,
    standardize: bool,
    exposures: pd.DataFrame | None,
    neutralize_columns: tuple[str, ...],
) -> pd.DataFrame:
    """横截面 MAD 去极值、可选市值中性化、标准化。"""

    frame = factor_raw[["code", factor_column]].copy()
    frame["factor_raw"] = pd.to_numeric(frame[factor_column], errors="coerce")
    frame = frame[["code", "factor_raw"]].replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["code", "factor_raw", "factor_processed"])

    median = frame["factor_raw"].median()
    mad = (frame["factor_raw"] - median).abs().median()
    if pd.notna(mad) and mad > 0:
        bound = 1.4826 * mad_n * mad
        frame["factor_processed"] = frame["factor_raw"].clip(
            median - bound, median + bound
        )
    else:
        frame["factor_processed"] = frame["factor_raw"]

    if exposures is not None:
        missing = [column for column in neutralize_columns if column not in exposures.columns]
        if missing:
            raise DataContractError(f"中性化暴露缺少列：{missing}")
        exposure_frame = exposures[["code", *neutralize_columns]].copy()
        exposure_frame[list(neutralize_columns)] = exposure_frame[
            list(neutralize_columns)
        ].apply(pd.to_numeric, errors="coerce")
        frame = frame.merge(
            exposure_frame, on="code", how="inner", validate="one_to_one"
        ).dropna()
        if len(frame) <= len(neutralize_columns) + 1:
            raise DataContractError("市值中性化有效样本不足")
        x = frame[list(neutralize_columns)].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        y = frame["factor_processed"].to_numpy(dtype=float)
        frame["factor_processed"] = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        frame = frame[["code", "factor_raw", "factor_processed"]]

    if standardize:
        std = frame["factor_processed"].std(ddof=0)
        if not np.isfinite(std) or std <= 0:
            return frame.iloc[0:0][["code", "factor_raw", "factor_processed"]]
        frame["factor_processed"] = (
            frame["factor_processed"] - frame["factor_processed"].mean()
        ) / std
    return frame[["code", "factor_raw", "factor_processed"]]


def use_preprocessed_factor(
    factor_raw: pd.DataFrame,
    factor_column: str,
) -> pd.DataFrame:
    """接收已完成横截面处理的因子，避免重复去极值和标准化。"""

    frame = factor_raw[["code", factor_column]].copy()
    frame["factor_raw"] = pd.to_numeric(frame[factor_column], errors="coerce")
    frame = frame[["code", "factor_raw"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    frame["factor_processed"] = frame["factor_raw"]
    return frame[["code", "factor_raw", "factor_processed"]]


def assign_quantile_groups(frame: pd.DataFrame, groups: int) -> pd.DataFrame:
    """先仅根据信号日因子分组，再读取未来收益。"""

    result = frame.sort_values(["factor_processed", "code"]).reset_index(drop=True).copy()
    if len(result) < groups:
        raise DataContractError("有效股票数小于分组数")
    result["group"] = np.floor(np.arange(len(result)) * groups / len(result)).astype(int) + 1
    counts = result.groupby("group")["code"].transform("size")
    result["target_weight"] = 1.0 / counts
    return result


def leg_turnover(
    history: list[tuple[pd.Timestamp, dict[str, float]]]
) -> pd.Series:
    """单边换手率：0.5 * 新旧权重绝对变化之和。"""

    values: dict[pd.Timestamp, float] = {}
    previous: dict[str, float] = {}
    for date, current in history:
        codes = set(previous) | set(current)
        values[pd.Timestamp(date)] = 0.5 * sum(
            abs(current.get(code, 0.0) - previous.get(code, 0.0)) for code in codes
        )
        previous = current
    return pd.Series(values, dtype=float).sort_index()


def performance_metrics(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
    """计算常用组合绩效指标。"""

    series = pd.to_numeric(returns, errors="coerce").dropna()
    if series.empty:
        return {
            name: np.nan
            for name in (
                "total_return",
                "annual_return",
                "annual_volatility",
                "sharpe",
                "max_drawdown",
                "win_rate",
            )
        }
    nav = (1.0 + series).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    annual_return = float(nav.iloc[-1] ** (periods_per_year / len(series)) - 1.0)
    annual_volatility = float(series.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    drawdown = nav / nav.cummax() - 1.0
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((series > 0).mean()),
    }


@dataclass
class BacktestResult:
    group_returns: pd.DataFrame
    ic_series: pd.DataFrame
    ic_summary: pd.DataFrame
    performance: pd.DataFrame
    audit: pd.DataFrame
    holdings: pd.DataFrame


class StrictFactorBacktester:
    """逐日 CSV/单文件 Parquet 因子 + Parquet 行情的通用回测器。"""

    def __init__(self, paths: DataPaths, factor: FactorSpec, config: BacktestConfig):
        config.validate(paths, factor)
        self.paths = paths
        self.factor = factor
        self.config = config
        self.factor_data = FactorDataSource(
            paths.factor_dir,
            factor.column,
            config.start_date,
            config.end_date,
            Path(paths.market_cache_dir).parent / "factor_data",
        )
        self.market = ParquetMarketData(
            paths.market_parquet,
            paths.market_cache_dir,
            config.market_exchanges,
        )

    def _signal_dates(self, calendar: list[pd.Timestamp]) -> list[pd.Timestamp]:
        calendar_set = set(calendar)
        dates = [date for date in self.factor_data.dates if date in calendar_set]
        return dates[:: self.config.effective_rebalance_step]

    def run(self, save: bool = True) -> BacktestResult:
        calendar = self.market.calendar(self.config.min_stocks)
        if not calendar:
            raise DataContractError("Parquet 没有满足最小股票数要求的交易日")

        daily_group_rows: list[dict[str, object]] = []
        ic_rows: list[dict[str, object]] = []
        audit_rows: list[dict[str, object]] = []
        holdings_frames: list[pd.DataFrame] = []
        weight_history: dict[int, list[tuple[pd.Timestamp, dict[str, float]]]] = {
            group: [] for group in range(1, self.config.groups + 1)
        }

        signal_dates = self._signal_dates(calendar)
        for signal_number, signal_date in enumerate(signal_dates, start=1):
            entry_date = trading_date_offset(calendar, signal_date, self.config.execution_lag)
            if entry_date is None:
                continue
            assert_entry_after_signal(signal_date, entry_date)

            # 先确认完整持有期没有越过样本边界，再读取因子、可交易标记或
            # 未来收益。这样验证期末应跳过的信号不会触碰测试期数据。
            expected_exit_date = trading_date_offset(
                calendar, entry_date, self.config.horizon
            )
            if expected_exit_date is None:
                continue
            if (
                not self.config.allow_exit_after_end
                and self.config.end_date
                and expected_exit_date > pd.Timestamp(self.config.end_date)
            ):
                continue

            factor_raw = self.factor_data.read_date(signal_date)
            validate_factor_asof(factor_raw, signal_date)
            if self.factor.column not in factor_raw.columns:
                raise DataContractError(
                    f"{signal_date.date()} 因子文件缺少列：{self.factor.column}"
                )

            exposures = None
            if self.config.neutralize:
                exposures = self.market.read_exposures(
                    signal_date, self.config.neutralize_columns
                )

            if self.factor.preprocessed:
                processed = use_preprocessed_factor(
                    factor_raw,
                    self.factor.column,
                )
            else:
                processed = process_factor_cross_section(
                    factor_raw,
                    self.factor.column,
                    self.config.mad_n,
                    self.config.standardize,
                    exposures,
                    self.config.neutralize_columns,
                )
            if len(processed) < self.config.min_stocks:
                continue

            # 关键：在读取未来收益之前完成分组。
            targets = assign_quantile_groups(processed, self.config.groups)
            targets["signal_date"] = signal_date
            targets["entry_date"] = entry_date

            tradability = self.market.read_tradability(
                self.config.tradability_mode,
                entry_date,
                self.paths.flags_dir,
            )
            if tradability is None:
                targets["filled"] = True
            else:
                targets = targets.merge(
                    tradability, on="code", how="left", validate="one_to_one"
                )
                targets["filled"] = targets["filled"].fillna(False).astype(bool)

            returns, exit_date = self.market.read_forward_returns(
                entry_date, self.config.horizon, calendar
            )
            if exit_date != expected_exit_date:
                raise DataContractError("未来收益退出日期与交易日历不一致")
            targets["exit_date"] = exit_date
            targets = targets.merge(returns, on="code", how="left", validate="one_to_one")

            filled = targets["filled"]
            missing_return = filled & targets["forward_return"].isna()
            missing_ratio = float(missing_return.sum() / max(filled.sum(), 1))
            if (
                self.config.missing_return_policy == "raise" and missing_ratio > 0
            ) or missing_ratio > self.config.max_missing_return_ratio:
                raise DataContractError(
                    f"{signal_date.date()} 建仓后缺失收益比例 {missing_ratio:.2%}，"
                    f"超过阈值 {self.config.max_missing_return_ratio:.2%}"
                )

            # 不可成交或缺失收益保留为现金收益 0，不使用未来信息重分配权重。
            targets["realized_return"] = 0.0
            valid_return = filled & targets["forward_return"].notna()
            targets.loc[valid_return, "realized_return"] = targets.loc[
                valid_return, "forward_return"
            ]
            targets["actual_weight"] = targets["target_weight"].where(filled, 0.0)
            targets["weighted_return"] = (
                targets["target_weight"] * targets["realized_return"]
            )

            row: dict[str, object] = {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
            }
            for group, group_frame in targets.groupby("group"):
                row[f"Q{group}"] = float(group_frame["weighted_return"].sum())
                weights = dict(
                    zip(
                        group_frame.loc[group_frame["filled"], "code"],
                        group_frame.loc[group_frame["filled"], "actual_weight"],
                    )
                )
                weight_history[int(group)].append((signal_date, weights))
            daily_group_rows.append(row)

            ic_sample = targets.loc[valid_return, ["factor_processed", "forward_return"]]
            ic_rows.append(
                {
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "pearson_ic": ic_sample["factor_processed"].corr(
                        ic_sample["forward_return"]
                    ),
                    "rank_ic": ic_sample["factor_processed"].rank().corr(
                        ic_sample["forward_return"].rank()
                    ),
                    "n_ic": len(ic_sample),
                }
            )
            audit_rows.append(
                {
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "factor_rows_raw": len(factor_raw),
                    "factor_rows_valid": len(processed),
                    "target_rows": len(targets),
                    "filled_rows": int(filled.sum()),
                    "fill_ratio": float(filled.mean()),
                    "missing_return_rows": int(missing_return.sum()),
                    "missing_return_ratio": missing_ratio,
                    "execution_lag": self.config.execution_lag,
                    "horizon": self.config.horizon,
                }
            )
            if self.config.save_holdings:
                holdings_frames.append(
                    targets[
                        [
                            "signal_date",
                            "entry_date",
                            "exit_date",
                            "code",
                            "factor_raw",
                            "factor_processed",
                            "group",
                            "target_weight",
                            "filled",
                            "actual_weight",
                            "forward_return",
                            "realized_return",
                        ]
                    ]
                )

            if signal_number % 250 == 0:
                print(f"已完成 {signal_number:,}/{len(signal_dates):,} 个信号日")

        if not daily_group_rows:
            raise DataContractError("没有产生有效回测日期，请检查区间和数据契约")

        group_returns = pd.DataFrame(daily_group_rows).set_index("signal_date").sort_index()
        ic_series = pd.DataFrame(ic_rows).set_index("signal_date").sort_index()
        audit = pd.DataFrame(audit_rows).set_index("signal_date").sort_index()
        holdings = (
            pd.concat(holdings_frames, ignore_index=True)
            if holdings_frames
            else pd.DataFrame()
        )

        if self.factor.direction == "negative":
            long_group, short_group = 1, self.config.groups
        else:
            long_group, short_group = self.config.groups, 1
        group_returns["long_short_gross"] = (
            group_returns[f"Q{long_group}"] - group_returns[f"Q{short_group}"]
        )
        long_turnover = (
            leg_turnover(weight_history[long_group])
            .reindex(group_returns.index)
            .fillna(0)
        )
        short_turnover = (
            leg_turnover(weight_history[short_group])
            .reindex(group_returns.index)
            .fillna(0)
        )
        group_returns["long_turnover"] = long_turnover
        group_returns["short_turnover"] = short_turnover
        group_returns["transaction_cost"] = (
            long_turnover + short_turnover
        ) * self.config.cost_bps / 10_000
        group_returns["long_short_net"] = (
            group_returns["long_short_gross"] - group_returns["transaction_cost"]
        )

        periods_per_year = 252 / self.config.effective_rebalance_step
        rank_ic_std = ic_series["rank_ic"].std(ddof=1)
        ic_summary = pd.DataFrame(
            {
                "value": {
                    "mean_pearson_ic": ic_series["pearson_ic"].mean(),
                    "mean_rank_ic": ic_series["rank_ic"].mean(),
                    "rank_ic_std": rank_ic_std,
                    "rank_ic_ir": (
                        ic_series["rank_ic"].mean()
                        / rank_ic_std
                        * np.sqrt(periods_per_year)
                        if rank_ic_std > 0
                        else np.nan
                    ),
                    "negative_rank_ic_ratio": (ic_series["rank_ic"] < 0).mean(),
                    "observations": ic_series["rank_ic"].count(),
                }
            }
        )
        metric_columns = [f"Q{i}" for i in range(1, self.config.groups + 1)]
        metric_columns += ["long_short_gross", "long_short_net"]
        performance = pd.DataFrame(
            {
                column: performance_metrics(group_returns[column], periods_per_year)
                for column in metric_columns
            }
        ).T
        performance.index.name = "portfolio"

        result = BacktestResult(
            group_returns, ic_series, ic_summary, performance, audit, holdings
        )
        if save:
            self.save(result)
        return result

    def save(self, result: BacktestResult) -> None:
        output = self.paths.output_dir
        output.mkdir(parents=True, exist_ok=True)
        result.group_returns.to_csv(output / "group_returns.csv")
        result.ic_series.to_csv(output / "ic_series.csv")
        result.ic_summary.to_csv(output / "ic_summary.csv")
        result.performance.to_csv(output / "performance.csv")
        result.audit.to_csv(output / "integrity_audit.csv")
        if not result.holdings.empty:
            result.holdings.to_csv(
                output / "holdings.csv.gz", index=False, compression="gzip"
            )
        self._save_chart(result.group_returns, output / "nav_curve.png")

        path_values = asdict(self.paths)
        path_values = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in path_values.items()
        }
        metadata = {
            "factor": asdict(self.factor),
            "config": asdict(self.config),
            "paths": path_values,
            "actual_sample": {
                "first_signal_date": f"{result.group_returns.index.min():%Y-%m-%d}",
                "last_signal_date": f"{result.group_returns.index.max():%Y-%m-%d}",
                "observations": len(result.group_returns),
            },
            "market_data": self.market.manifest,
            "time_contract": {
                "factor_available_at": "signal_date close",
                "entry_rule": (
                    f"signal_date + {self.config.execution_lag} trading day(s)"
                ),
                "return_rule": (
                    f"entry adjusted VWAP to entry + {self.config.horizon} trading day(s) adjusted VWAP"
                ),
                "grouping_before_loading_returns": True,
                "allow_exit_after_end": self.config.allow_exit_after_end,
            },
            "factor_data": {
                "source": str(self.factor_data.source),
                "mode": self.factor_data.mode,
                "preprocessed": self.factor.preprocessed,
            },
        }
        (output / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_chart(self, returns: pd.DataFrame, path: Path) -> None:
        import os

        cache = path.parent / ".matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache))
        # 命令行回测固定使用无界面后端，避免 macOS GUI 后端在子进程中崩溃。
        os.environ.setdefault("MPLBACKEND", "Agg")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("当前环境没有 matplotlib，已跳过净值图。")
            return

        columns = ["Q1", f"Q{self.config.groups}", "long_short_net"]
        nav = (1 + returns[columns].fillna(0)).cumprod()
        fig, ax = plt.subplots(figsize=(11, 6))
        nav.plot(ax=ax, linewidth=1.4)
        ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{self.factor.name} strict backtest")
        ax.set_xlabel("Signal date")
        ax.set_ylabel("Net asset value")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
