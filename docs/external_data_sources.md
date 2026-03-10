# 外部数据源字段说明（纯数据字典）

> 本文仅用于查看外部数据源本身，不涉及任何业务流程、策略、回测、实盘、项目实现。
> 字段类型来自 2026-03-09 的样本扫描结果。

## 1. 数据源清单

| 数据源 | 出处 |
|---|---|
| snapshot | NFS：`yinhe-data/snapshot/cbond/raw_data`；Redis：`asset_type=cbond, source=combiner, stage=raw` |
| kline | 源数据库 KLINE 表（按数据库配置直连读取） |
| metadata.trading_calendar | 源数据库：`metadata.trading_calendar` |
| market_cbond.daily_price | 源数据库：`market_cbond.daily_price` |
| market_cbond.daily_twap | 源数据库：`market_cbond.daily_twap` |
| market_cbond.daily_vwap | 源数据库：`market_cbond.daily_vwap` |
| market_cbond.daily_deriv | 源数据库：`market_cbond.daily_deriv` |
| market_cbond.daily_base | 源数据库：`market_cbond.daily_base` |
| market_cbond.daily_rating | 源数据库：`market_cbond.daily_rating` |

## 2. 字段字典

## 2.1 NFS/Redis snapshot（同构，37 列）

出处：
- NFS：`yinhe-data/snapshot/cbond/raw_data`
- Redis：`asset_type=cbond, source=combiner, stage=raw`

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| code | object | 证券代码（如 `110067.SH`） | 110070.SH |
| trade_time | datetime64[ns] | 时间戳 | 2026-03-09 01:30:02 |
| pre_close | float64 | 前收价 | 122.155 |
| last | float64 | 最新价 | 121.708 |
| open | float64 | 开盘价 | 121.71 |
| high | float64 | 最高价 | 121.992 |
| low | float64 | 最低价 | 121.21 |
| close | float64 | 收盘价 | 0.0 |
| volume | float64 | 成交量 | 4250.0 |
| amount | float64 | 成交额 | 517286.37 |
| num_trades | int64 | 成交笔数 | 34 |
| high_limited | float64 | 涨停价 | 146.586 |
| low_limited | float64 | 跌停价 | 97.724 |
| ask_price1 | float64 | 卖一价 | 121.707 |
| ask_volume1 | float64 | 卖一量 | 200.0 |
| bid_price1 | float64 | 买一价 | 121.676 |
| bid_volume1 | float64 | 买一量 | 20.0 |
| ask_price2 | float64 | 卖二价 | 121.71 |
| ask_volume2 | float64 | 卖二量 | 3630.0 |
| bid_price2 | float64 | 买二价 | 121.5 |
| bid_volume2 | float64 | 买二量 | 20.0 |
| ask_price3 | float64 | 卖三价 | 121.8 |
| ask_volume3 | float64 | 卖三量 | 660.0 |
| bid_price3 | float64 | 买三价 | 121.21 |
| bid_volume3 | float64 | 买三量 | 50.0 |
| ask_price4 | float64 | 卖四价 | 121.9 |
| ask_volume4 | float64 | 卖四量 | 810.0 |
| bid_price4 | float64 | 买四价 | 121.1 |
| bid_volume4 | float64 | 买四量 | 20.0 |
| ask_price5 | float64 | 卖五价 | 121.948 |
| ask_volume5 | float64 | 卖五量 | 90.0 |
| bid_price5 | float64 | 买五价 | 121.001 |
| bid_volume5 | float64 | 买五量 | 10.0 |
| iopv | float64 | iopv | 0.0 |
| trading_phase_code | object | 交易阶段码 | T |
| __index_level_0__ | float64 | 索引残留字段 | 1.0 |
| source | object | 来源标记 | itrade |

## 2.2 NFS kline

## 2.3 metadata.trading_calendar（15 列）

出处：`metadata.trading_calendar`

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| calendar_date | object | 自然日 | 2000-01-01 |
| is_open | bool | 是否交易日 | True |
| normal_date_count | int64 | 自然日序号 | 1 |
| trade_date_count | float64 | 交易日序号 | 1.0 |
| prev_trade_date | object | 前一交易日 | 2000-01-04 |
| next_trade_date | object | 后一交易日 | 2000-01-05 |
| week_start_date | object | 周起始日 | 1999-12-27 |
| week_end_date | object | 周结束日 | 1999-12-30 |
| month_start_date | object | 月起始日 | 2000-01-04 |
| month_end_date | object | 月结束日 | 2000-01-28 |
| quarter_start_date | object | 季起始日 | 2000-01-04 |
| quarter_end_date | object | 季结束日 | 2000-03-31 |
| year_start_date | object | 年起始日 | 2000-01-04 |
| year_end_date | object | 年结束日 | 2000-12-29 |
| update_time | datetime64[ns, UTC] | 更新时间 | - |

## 2.4 market_cbond.daily_price（13 列）

出处：`market_cbond.daily_price`

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| instrument_code | object | 债券代码 | 127501 |
| exchange_code | object | 交易所代码 | SH |
| trade_date | object | 交易日 | 2026-03-06 |
| prev_close_price | float64 | 前收价 | 33.05 |
| act_prev_close_price | float64 | 复权前收价 | 33.05 |
| close_price | float64 | 收盘价 | 33.05 |
| open_price | float64 | 开盘价 | 141.2 |
| high_price | float64 | 最高价 | 144.2 |
| low_price | float64 | 最低价 | 138.76 |
| volume | float64 | 日成交量 | 1257699.0 |
| amount | float64 | 日成交额 | 178982979.38 |
| deal | float64 | 日成交笔相关指标 | 17441.0 |
| update_time | datetime64[ns, UTC] | 更新时间 | - |

## 2.5 market_cbond.daily_twap（26 列）

出处：`market_cbond.daily_twap`

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| instrument_code | object | 债券代码 | 110067 |
| exchange_code | object | 交易所代码 | SH |
| trade_date | object | 交易日 | 2026-03-06 |
| twap_0930_1000 | float64 | 09:30-10:00 TWAP | 106.9906 |
| twap_0930_0945 | float64 | 09:30-09:45 TWAP | 106.9828 |
| twap_0930_1030 | float64 | 09:30-10:30 TWAP | 106.9951 |
| twap_0930_1015 | float64 | 09:30-10:15 TWAP | 106.9935 |
| twap_1442_1457 | float64 | 14:42-14:57 TWAP | 106.9942 |
| twap_1447_1457 | float64 | 14:47-14:57 TWAP | 106.994 |
| twap_1452_1457 | float64 | 14:52-14:57 TWAP | 106.9963 |
| twap_0935_1000 | float64 | 09:35-10:00 TWAP | 106.9926 |
| twap_0945_1015 | float64 | 09:45-10:15 TWAP | 106.9989 |
| twap_1000_1030 | float64 | 10:00-10:30 TWAP | 106.9996 |
| twap_1030_1100 | float64 | 10:30-11:00 TWAP | 106.9992 |
| twap_1100_1130 | float64 | 11:00-11:30 TWAP | 107.0002 |
| twap_1300_1330 | float64 | 13:00-13:30 TWAP | 106.9962 |
| twap_1330_1400 | float64 | 13:30-14:00 TWAP | 107.0211 |
| twap_1400_1430 | float64 | 14:00-14:30 TWAP | 107.0035 |
| twap_1430_1500 | float64 | 14:30-15:00 TWAP | 106.9963 |
| twap_0935_1005 | float64 | 09:35-10:05 TWAP | 106.9937 |
| twap_1430_1442 | float64 | 14:30-14:42 TWAP | 106.9992 |
| twap_0945_1000 | float64 | 09:45-10:00 TWAP | 106.9984 |
| twap_0935_0950 | float64 | 09:35-09:50 TWAP | 106.9882 |
| twap_0935_1030 | float64 | 09:35-10:30 TWAP | 106.9964 |
| twap_0930_0935 | float64 | 09:30-09:35 TWAP | 106.9805 |
| update_time | datetime64[ns, UTC] | 更新时间 | - |

## 2.6 market_cbond.daily_vwap（26 列）

出处：`market_cbond.daily_vwap`

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| instrument_code | object | 债券代码 | 110067 |
| exchange_code | object | 交易所代码 | SH |
| trade_date | object | 交易日 | 2026-03-06 |
| vwap_0930_1000 | float64 | 09:30-10:00 VWAP | 106.9914 |
| vwap_0930_0945 | float64 | 09:30-09:45 VWAP | 106.9843 |
| vwap_0930_1030 | float64 | 09:30-10:30 VWAP | 106.9962 |
| vwap_0930_1015 | float64 | 09:30-10:15 VWAP | 106.9939 |
| vwap_1442_1457 | float64 | 14:42-14:57 VWAP | 106.9951 |
| vwap_1447_1457 | float64 | 14:47-14:57 VWAP | 106.9953 |
| vwap_1452_1457 | float64 | 14:52-14:57 VWAP | 106.9966 |
| vwap_0935_1000 | float64 | 09:35-10:00 VWAP | 106.9949 |
| vwap_0945_1015 | float64 | 09:45-10:15 VWAP | 106.9995 |
| vwap_1000_1030 | float64 | 10:00-10:30 VWAP | 106.9998 |
| vwap_1030_1100 | float64 | 10:30-11:00 VWAP | 106.9998 |
| vwap_1100_1130 | float64 | 11:00-11:30 VWAP | 107.0002 |
| vwap_1300_1330 | float64 | 13:00-13:30 VWAP | 106.9978 |
| vwap_1330_1400 | float64 | 13:30-14:00 VWAP | 107.026 |
| vwap_1400_1430 | float64 | 14:00-14:30 VWAP | 107.0007 |
| vwap_1430_1500 | float64 | 14:30-15:00 VWAP | 106.9948 |
| vwap_0935_1005 | float64 | 09:35-10:05 VWAP | 106.9954 |
| vwap_1430_1442 | float64 | 14:30-14:42 VWAP | 106.9942 |
| vwap_0945_1000 | float64 | 09:45-10:00 VWAP | 106.9994 |
| vwap_0935_0950 | float64 | 09:35-09:50 VWAP | 106.9928 |
| vwap_0935_1030 | float64 | 09:35-10:30 VWAP | 106.9979 |
| vwap_0930_0935 | float64 | 09:30-09:35 VWAP | 106.9784 |
| update_time | datetime64[ns, UTC] | 更新时间 | - |

## 2.7 market_cbond.daily_deriv（25 列）

出处：`market_cbond.daily_deriv`

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| instrument_code | object | 债券代码 | 110067 |
| exchange_code | object | 交易所代码 | SH |
| trade_date | object | 交易日 | 2026-03-06 |
| year_to_mat | float64 | 距到期年限 | 0.016438 |
| remain_size | float64 | 剩余规模 | 14.8418 |
| current_yield | float64 | 当前收益率 | 1.869176 |
| cb_conv_price | float64 | 转股价 | 5.78 |
| cb_call_price | float64 | 强赎触发价 | 7.514 |
| cb_put_price | float64 | 回售触发价 | 7.826 |
| turnover_rate | float64 | 换手率 | 38.7419 |
| stock_code | object | 正股代码 | 600909 |
| stock_close_price | float64 | 正股收盘价 | 6.18 |
| bond_prem_ratio | float64 | 转股溢价率 | 0.073498 |
| debt_puredebt_ratio | object | 纯债相关字段 | - |
| puredebt_prem_ratio | object | 纯债相关字段 | - |
| conv_value | float64 | 转股价值 | 106.920415 |
| ytm | float64 | 到期收益率 | 0.0625 |
| duration | float64 | 久期 | 0.016438 |
| modify_duration | float64 | 修正久期 | 0.016428 |
| convexity | float64 | 凸性 | 0.016688 |
| base_rate | float64 | 基准利率 | 0.000275 |
| stock_volatility | float64 | 正股波动率 | 27.227723 |
| pure_redemption_value | float64 | 纯债价值 | 105.548703 |
| redemption_prem_ratio | float64 | 纯债溢价率 | 1.374055 |
| update_time | datetime64[ns, UTC] | 更新时间 | - |

## 2.8 market_cbond.daily_base（50 列）

出处：`market_cbond.daily_base`

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| instrument_code | object | 债券代码 | 110067 |
| exchange_code | object | 交易所代码 | SH |
| trade_date | object | 交易日 | 2026-03-06 |
| year_to_mat | float64 | 距到期年限 | 0.016438 |
| remain_size | float64 | 剩余规模 | 14.8418 |
| current_yield | float64 | 当前收益率 | 1.869176 |
| cb_conv_price | float64 | 转股价 | 5.78 |
| cb_put_price | float64 | 回售触发价 | 1.379 |
| turnover_rate | float64 | 换手率 | 38.7419 |
| stock_code | object | 正股代码 | 600909 |
| stock_close_price | float64 | 正股收盘价 | 6.18 |
| bond_prem_ratio | float64 | 转股溢价率 | 0.073498 |
| debt_puredebt_ratio | object | 纯债相关字段 | - |
| puredebt_prem_ratio | object | 纯债相关字段 | - |
| conv_value | float64 | 转股价值 | 106.920415 |
| ytm | float64 | 到期收益率 | 0.0625 |
| duration | float64 | 久期 | 0.016438 |
| modify_duration | float64 | 修正久期 | 0.016428 |
| convexity | float64 | 凸性 | 0.016688 |
| base_rate | float64 | 基准利率 | 0.000275 |
| stock_volatility | float64 | 正股波动率 | 27.227723 |
| pure_redemption_value | float64 | 纯债价值 | 105.548703 |
| redemption_prem_ratio | float64 | 纯债溢价率 | 1.374055 |
| cb_prev_close_price | float64 | 转债前收 | 107.01 |
| cb_act_prev_close_price | float64 | 转债复权前收 | 107.01 |
| cb_close_price | float64 | 转债收盘 | 107.0 |
| cb_volume | float64 | 转债成交量 | 575001.0 |
| cb_amount | float64 | 转债成交额 | 615263065.0 |
| cb_deal | float64 | 转债成交笔相关指标 | 16254.0 |
| stk_prev_close_price | float64 | 正股前收 | 6.15 |
| stk_act_prev_close_price | float64 | 正股复权前收 | 6.15 |
| stk_close_price | float64 | 正股收盘 | 6.18 |
| stk_volume | float64 | 正股成交量 | 88944042.0 |
| stk_amount | float64 | 正股成交额 | 549209995.0 |
| stk_deal | float64 | 正股成交笔相关指标 | 28383.0 |
| rating | object | 评级 | AAA |
| cb_call_price | float64 | 强赎触发价 | 7.514 |
| trigger_is_price | object | 触发类型标记 | 1 |
| trigger_cum_days | float64 | 触发累计天数 | 3.0 |
| trigger_reach_days | float64 | 触发阈值天数 | 15.0 |
| trigger_process | object | 触发进度 | 3/15 |
| trigger_date | object | 触发日期 | - |
| in_trigger_process | float64 | 是否处于触发过程 | 1.0 |
| trigger_price_revise | float64 | 修正触发价 | 4.624 |
| trigger_is_price_revise | object | 修正触发类型标记 | 1 |
| trigger_cum_days_revise | float64 | 修正累计天数 | 4.0 |
| trigger_reach_days_revise | float64 | 修正阈值天数 | 15.0 |
| trigger_process_revise | object | 修正触发进度 | 0/15 |
| trigger_date_revise | object | 修正触发日期 | - |
| update_time | datetime64[ns, UTC] | 更新时间 | - |

## 2.9 market_cbond.daily_rating（6 列）

出处：`market_cbond.daily_rating`

| 字段 | 类型 | 含义 | 示例值 |
|---|---|---|---|
| id | int64 | 记录主键 | 113043327 |
| instrument_code | object | 债券代码 | 100016 |
| exchange_code | object | 交易所代码 | SH |
| trade_date | object | 交易日 | 2026-03-06 |
| rating | object | 评级 | AA |
| update_time | datetime64[ns, UTC] | 更新时间 | - |

## 3. Real Non-zero Examples

- snapshot example:
  - `{ "code": "110070.SH", "trade_time": "2026-03-09 01:30:02", "last": "121.708", "volume": "4250.0", "amount": "517286.37", "trading_phase_code": "T" }`
- daily_price example:
  - `{ "instrument_code": "127501", "exchange_code": "SH", "trade_date": "2026-03-06", "close_price": "33.05", "volume": "0.0", "amount": "0.0" }`
- daily_twap example:
  - `{ "instrument_code": "110067", "exchange_code": "SH", "trade_date": "2026-03-06", "twap_1442_1457": "106.9942", "twap_0930_0945": "106.9828" }`
