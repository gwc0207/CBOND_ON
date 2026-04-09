# CBOND_ON 鍥犲瓙寮€鍙戞墜鍐岋紙Agent 浜ゆ帴鐗堬紝2026-03-25锛?
> 鐩爣锛氳鏂?Agent 鍙互鍦ㄤ笉鍙嶅闂汉鐨勫墠鎻愪笅锛岀嫭绔嬪畬鎴愬洜瀛愬紑鍙戙€佸洖娴嬨€佺瓫閫夈€佸叆妯″鎺ャ€?
## 1. 椤圭洰鍙ｅ緞锛堝厛缁熶竴锛?
- ON 宸茬粡鏄?DataHub 娑堣垂鏂癸細
  - `raw/clean` 鐢?DataHub 鐢熶骇涓庣淮鎶わ紱
  - ON 涓嶅啀璐熻矗鎷?raw / 鐢熸垚 clean銆?- ON 璐熻矗鐨勮鐢熼摼璺細
  - `panel -> label -> factor -> model_score -> backtest/live`銆?- 鍥犲瓙寮€鍙戦粯璁ゅ熀浜?panel锛堝綋鍓嶆槸 cbond panel锛夈€?
## 2. 褰撳墠鍏抽敭璺緞锛堟湰鍦伴厤缃級

- `raw_data_root`: `D:/cbond_data_hub/raw_data`
- `clean_data_root`: `D:/cbond_data_hub/clean_data`
- `panel_data_root`: `D:/cbond_on/panel_data`
- `label_data_root`: `D:/cbond_on/label_data`
- `factor_data_root`: `D:/cbond_on/factor_data`
- `results_root`: `D:/cbond_on/results`

閰嶇疆鏂囦欢锛?
- 璺緞閰嶇疆锛歚cbond_on/config/data/paths_config.json5`
- panel 閰嶇疆锛歚cbond_on/config/data/panel_config.json5`
- label 閰嶇疆锛歚cbond_on/config/data/label_config.json5`
- factor 閰嶇疆锛歚cbond_on/config/factor/factor_config.json5`

## 3. 浠ｇ爜鍏ュ彛涓庤亴璐?
- 鍥犲瓙瀹氫箟鐩綍锛歚cbond_on/domain/factors/defs/`
- 娉ㄥ唽鍏ュ彛锛歚cbond_on/domain/factors/defs/__init__.py`
- 鍥犲瓙鍩虹被锛歚cbond_on/domain/factors/base.py`
- 鍥犲瓙璁＄畻娴佹按绾匡細`cbond_on/infra/factors/pipeline.py`
- batch 涓绘祦绋嬶紙鏋勫缓 + 鍗曞洜瀛愬洖娴?+ 鎶ュ憡 + 杩囩瓫锛夛細`cbond_on/app/usecases/factor_batch_runtime.py`
- 杩愯鍏ュ彛锛歚cbond_on/run/factor_batch.py`

鍏抽敭浜嬪疄锛?
- `run/factor_batch.py` 宸叉樉寮忓鍏?`cbond_on.domain.factors.defs`锛岄伩鍏嶆湭娉ㄥ唽閿欒銆?- 鍥犲瓙鍒楀悕浠?`FactorSpec.name`锛堟垨 `output_col`锛変负鍑嗭紝涓嶅啀鑷姩鎷兼帴鍙傛暟銆?- 鍥犲瓙涓婁笅鏂囩幇鏀寔涓夌被杈撳叆锛?  - `ctx.panel`锛坈bond panel锛屼富杈撳叆锛?  - `ctx.stock_panel`锛坰tock panel锛屽彲涓虹┖锛?  - `ctx.bond_stock_map`锛堝€鸿偂鏄犲皠锛屾潵鑷?`market_cbond.daily_base`锛屽彲涓虹┖锛?
## 4. 鍙敤鏁版嵁璧勬簮涓庡瓧娈?
## 4.1 Panel锛堝洜瀛?compute 鐩存帴鍙敤锛?
褰撳墠鏍锋湰锛坄panel_data/panels/cbond/T1430/2026-03/20260324.parquet`锛夛細

- 绱㈠紩锛歚dt, code, seq`
- 鍒楋紙34锛夛細

```text
trade_time
pre_close
last
open
high
low
close
volume
amount
num_trades
high_limited
low_limited
ask_price1
ask_volume1
bid_price1
bid_volume1
ask_price2
ask_volume2
bid_price2
bid_volume2
ask_price3
ask_volume3
bid_price3
bid_volume3
ask_price4
ask_volume4
bid_price4
bid_volume4
ask_price5
ask_volume5
bid_price5
bid_volume5
iopv
trading_phase_code
```

## 4.2 Stock 璧勬簮锛堝彲鐢紝浣嗕笉浼氳嚜鍔ㄨ繘褰撳墠鍥犲瓙娴佹按绾匡級

- `clean_data/snapshot/stock/YYYY-MM/YYYYMMDD.parquet`锛堢洏涓揩鐓э紝瀛楁缁撴瀯涓?cbond snapshot 瀵归綈锛?- `raw_data/market_cbond__daily_base`锛堟棩棰戯紝甯哥敤瀛楁锛歚stock_code`, `stock_close_price`, `stock_volatility`, `stk_*`, `conv_value`, `bond_prem_ratio`, `puredebt_prem_ratio`锛?- `raw_data/market_cbond__daily_deriv`锛堟棩棰戯紝甯哥敤瀛楁锛歚stock_code`, `stock_close_price`, `stock_volatility`, `conv_value`, `bond_prem_ratio`, `puredebt_prem_ratio`锛?
褰撳墠 `stock panel` 鍙洿鎺ヤ娇鐢ㄥ瓧娈碉紙涓?cbond panel 瀵归綈锛?4 鍒楋級锛?
```text
trade_time
pre_close
last
open
high
low
close
volume
amount
num_trades
high_limited
low_limited
ask_price1
ask_volume1
bid_price1
bid_volume1
ask_price2
ask_volume2
bid_price2
bid_volume2
ask_price3
ask_volume3
bid_price3
bid_volume3
ask_price4
ask_volume4
bid_price4
bid_volume4
ask_price5
ask_volume5
bid_price5
bid_volume5
iopv
trading_phase_code
```

褰撳墠 `bond_stock_map` 杈撳嚭瀛楁锛?
- `code`锛堣浆鍊轰唬鐮侊紝甯︿氦鏄撴墍鍚庣紑锛?- `stock_code`锛堟鑲′唬鐮侊紝甯︿氦鏄撴墍鍚庣紑锛?- `trade_date`锛堟槧灏勬墍灞炰氦鏄撴棩锛?
娉ㄦ剰锛?
- 褰撳墠 `run_factor_pipeline` 璇诲彇鐨勬槸 `panel_data/panels/cbond/...`锛?- stock 淇℃伅宸插彲閫氳繃 `ctx.stock_panel` 涓?`ctx.bond_stock_map` 杩涘叆鍥犲瓙锛?- 闇€瑕佽仈鍔ㄦ椂锛屽洜瀛愬唴閮ㄦ樉寮忔寜鏄犲皠鍏崇郴涓庢椂闂村榻愶紝涓嶈闅愬紡鍋囪 code 涓€鑷淬€?
## 4.3 澶氳祫浜т笂涓嬫枃锛堝凡钀藉湴锛?
`factor_config.context` 鎺у埗鏄惁娉ㄥ叆 stock 璧勬簮锛?
```json5
context: {
  stock_panel: { enabled: true, strict: false },
  bond_stock_map: { enabled: true, strict: false, table: "market_cbond.daily_base" }
}
```

璇存槑锛?
- `stock_panel.enabled=true`锛氭寜鍚屼竴浜ゆ槗鏃ヨ鍙?`panel_data/panels/stock/<panel_name>/...`
- `bond_stock_map.enabled=true`锛氳鍙?`raw_data/market_cbond__daily_base` 褰㈡垚 `code -> stock_code` 鏄犲皠
- `strict=true`锛氱己澶卞嵆鎶ラ敊锛沗strict=false`锛氱己澶卞垯浼犵┖锛屽洜瀛愯嚜琛屽厹搴?- 鏄犲皠浼氳嚜鍔ㄨ鑼冧负甯︿氦鏄撴墍鍚庣紑锛堜緥濡?`113574.SH -> 603679.SH`锛?- 鏄犲皠璇诲彇鏀寔鈥滃悜鍓嶅洖閫€鈥濓細褰撳ぉ缂哄け鏃朵娇鐢ㄦ渶杩戝彲鐢ㄤ氦鏄撴棩鏄犲皠

鍥犲瓙涓鍙栨柟寮忕ず渚嬶細

```python
def compute(self, ctx: FactorComputeContext) -> pd.Series:
    bond = ctx.panel
    stock = ctx.stock_panel          # 鍙兘涓?None
    mapping = ctx.bond_stock_map     # 鍙兘涓?None
    ...
```

## 5. 鍥犳灉涓庢椂闂磋鍒欙紙蹇呴』閬靛畧锛?
- `factor_time`锛堥粯璁?14:30锛変箣鍓嶅彲瑙佷俊鎭墠鍏佽杩涘叆鍥犲瓙銆?- `label_time`锛堥粯璁?14:42锛変箣鍚庣殑淇℃伅涓ョ娉勯湶鍒板洜瀛愩€?- 鏃ラ姝ｈ偂瀛楁榛樿鎸夆€滀笂涓€浜ゆ槗鏃モ€濆榻愶紱涓嶈鐩存帴鎶婂悓鏃ユ敹鐩樺彛寰勫瓧娈电敤浜庣洏涓洜瀛愩€?- 椤圭洰浜ゆ槗鏃ュ彛寰勪互浜ゆ槗鏃ユ湇鍔′负鍑嗭紝涓嶆寜鑷劧鏃ユ帹鏂€?
## 6. 鍥犲瓙寮€鍙戣鑼冿紙纭害鏉燂級

1. 涓€鍥犲瓙涓€鏂囦欢锛歚cbond_on/domain/factors/defs/<factor>.py`
2. 绫荤户鎵?`Factor`
3. 蹇呴』娉ㄥ唽锛歚@FactorRegistry.register("<factor_key>")`
4. `compute(ctx)` 杩斿洖 `pd.Series`锛岀储寮曞榻?`(dt, code)`锛堟渶缁堣惤鐩樺甫 `seq` 缁撴瀯锛?5. 杈撳嚭鍒楀悕鐢?`self.output_name(self.name)`
6. 鏄惧紡澶勭悊绌哄€煎拰寮傚父锛堜笉寰?silent fail锛?7. 鏂板鍥犲瓙鍚庡繀椤诲湪 `cbond_on/domain/factors/defs/__init__.py` 瀵煎叆

## 7. 鍛藉悕瑙勮寖锛堥厤缃眰锛?
- `factor`锛氭敞鍐岄敭锛堜緥濡?`ret_window`锛?- `name`锛氳瀹炰緥钀界洏鍒楀悕/鎶ュ憡鐩綍鍚嶏紙蹇呴』鍞竴锛?- `output_col`锛氬彲閫夛紝瑕嗙洊鏈€缁堝垪鍚?
寤鸿锛?
- 鐢ㄥ皬鍐?snake_case
- 缁撴瀯寤鸿锛歚<factor_short>_<key_param>`
- 涓嶈鐢ㄧ┖鏍笺€佷腑鏂囥€佽矾寰勭

## 8. 褰撳墠 batch 琛屼负锛堥潪甯搁噸瑕侊級

杩愯鍛戒护锛?
```bash
python cbond_on/run/factor_batch.py
```

褰撳墠 `factor_batch` 鍖呭惈浠ヤ笅鐜妭锛?
1. 鍥犲瓙鏋勫缓锛堟敮鎸佸绾跨▼锛宍factor_config.workers`锛?2. 鍗曞洜瀛愬洖娴嬶紙鏀寔澶氱嚎绋嬶紝`factor_config.backtest.workers`锛?3. 姣忎釜鍥犲瓙杈撳嚭鎶ュ憡鐩綍
4. 鎸夌瓫閫夎鍒欑敓鎴愬叆鍥磋〃
5. 鎶婂叆鍥村洜瀛愮殑鍥剧墖澶嶅埗鍒?`screened/selected_reports`

姣忔 batch 杈撳嚭鐩綍锛?
`D:/cbond_on/results/<start>_<end>/Single_Factor/<batch_ts>/`

鍏朵腑鍏抽敭鏂囦欢锛?
- `<factor_name>/factor_report.png`
- `<factor_name>/factor_metrics.csv`
- `<factor_name>/summary.json`
- `screened/factor_screening_all.csv`
- `screened/factor_shortlist.csv`
- `screened/selected_reports/<factor_name>.png`锛堜粎鍥剧墖锛?- `screened/screening_config.json`

## 9. 绛涢€夐€昏緫锛堝綋鍓嶅疄鐜帮級

绛涢€夐厤缃湪 `factor_config.screening`锛屾牳蹇冨瓧娈碉細

- `enabled`
- `ic_metric`锛堜緥濡?`rank_ic_mean`锛?- `ir_metric`锛堜緥濡?`rank_ic_ir`锛?- `ic_abs_min`锛堟瘮杈?`abs(ic_metric_value)`锛?- `ir_abs_min`锛堟瘮杈?`abs(ir_metric_value)`锛?- `sharpe_min`锛堟瘮杈?`sharpe >= sharpe_min`锛?- `copy_reports`锛堟槸鍚﹁緭鍑?`selected_reports`锛?
璇存槑锛?
- 闃堝€间笉搴旂‖缂栫爜鍦ㄤ唬鐮侀噷锛岀粺涓€浠庨厤缃鍙栵紱
- 鏂?Agent 鏀归槇鍊煎彧鏀归厤缃紝涓嶆敼閫昏緫浠ｇ爜銆?
## 10. 鍏ユā瀵规帴瑕佹眰

鏂板洜瀛愯繘鍏ユā鍨嬭缁冨墠锛屽繀椤诲悓姝ヤ慨鏀规ā鍨嬮厤缃腑鐨?`factors` 鍒楄〃锛?
- `cbond_on/config/models/lgbm/lgbm_factor_MSE_config.json5`
- `cbond_on/config/models/lgbm_ranker/lgbm_factor_ranker_config.json5`
- `cbond_on/config/models/linear/linear_factor_default_config.json5`

涓嶆敼杩欓噷浼氬嚭鐜扳€滃洜瀛愬凡钀界洏浣嗘ā鍨嬫湭璇诲彇鈥濈殑鍋囪薄銆?
## 11. 鎺ㄨ崘寮€鍙戞祦绋嬶紙缁欐柊 Agent锛?
1. 閫変竴涓ā鏉垮洜瀛愬鍒跺紑鍙戯紙鎺ㄨ崘 `ret_window.py` 鎴?`depth_imbalance.py`锛?2. 瀹炵幇 `compute()`锛屽厛鍙敤 panel 瀛楁锛岀‘淇濇棤娉勯湶
3. 鍦?`defs/__init__.py` 娉ㄥ唽瀵煎叆
4. 鍦?`factor_config.factors` 澧炲姞閰嶇疆椤癸紙鍞竴 `name`锛?5. 鍏堝皬鍖洪棿璺?`factor_batch` 楠岃瘉
6. 鐪?`factor_metrics.csv`銆乣factor_report.png`銆乣screened/factor_shortlist.csv`
7. 閫氳繃鍚庡啀鍔犲叆妯″瀷閰嶇疆 `factors`

## 12. 甯歌閿欒涓庢帓鏌?
- `RegistryError: 鏈壘鍒版敞鍐岄」`
  - 娌℃敞鍐屾垨娌″湪 `defs/__init__.py` 瀵煎叆锛屾垨鍏ュ彛娌″姞杞?`defs`
- `KeyError: panel missing column`
  - 鍥犲瓙璇锋眰浜?panel 涓嶅瓨鍦ㄥ瓧娈碉紝鍏堟墦鍗?`ctx.panel.columns`
- 鍥犲瓙缁撴灉鍏?0 / 鍏?NaN
  - 绐楀彛鍙傛暟杩囧ぇ銆佽繃婊よ繃涓ャ€佸瓧娈靛彛寰勪笉瀵?- IC 寮傚父楂?  - 楂樻鐜囧彂鐢熶俊鎭硠闇诧紙灏ゅ叾璇敤鏀剁洏鍙ｅ緞瀛楁锛?
## 13. 鏂?Agent 浜や粯鏍囧噯

- 浠ｇ爜锛?  - 鍥犲瓙瀹炵幇鏂囦欢
  - `defs/__init__.py` 瀵煎叆
  - `factor_config` 鏂板椤?- 缁撴灉锛?  - 鑷冲皯涓€杞?`factor_batch` 鎴愬姛
  - 鎶ュ憡鍥剧墖鍙
  - 杩囩瓫琛ㄥ彲璇伙紙鍚?pass/fail锛?- 鏂囨。锛?  - 鍦ㄦ湰鏂囦欢琛ュ厖鈥滄柊鍥犲瓙璇存槑 + 瀛楁渚濊禆 + 娉勯湶椋庨櫓妫€鏌モ€?
---

## 14. 澶?Agent 鍗忎綔绾﹀畾锛圤N 缁存姢 Agent 涓庡洜瀛愮爺绌?Agent锛?
涓哄噺灏戞潵鍥炴矡閫氭垚鏈紝鍥犲瓙鐮旂┒ Agent 闇€瑕佹彁渚涗互涓嬪唴瀹癸細

1. 鍥犲瓙瑙勬牸鍗★紙蹇呴』锛?   - 鍥犲瓙鍚嶏紙鏈€缁?`name`锛宻nake_case锛屽敮涓€锛?   - 鍥犲瓙鍏紡锛堝瓧娈电骇琛ㄨ揪锛屼笉瑕佸彧缁欐蹇碉級
   - 渚濊禆瀛楁娓呭崟锛堟爣娉ㄦ潵鑷?panel / raw 鏃ラ / stock 蹇収锛?   - 鏃堕棿鍥犳灉澹版槑锛堟瘡涓瓧娈垫槸鍚﹀湪 `factor_time` 鍓嶅彲瑙侊級
   - 鍙傛暟娓呭崟锛堥粯璁ゅ€笺€佸彲璋冭寖鍥达級

2. 鍙樻洿璇存槑锛堟瘡娆¤凯浠ｅ繀椤伙級
   - 鏈鍙樻洿鍐呭锛堝叕寮忔敼鍔?/ 鍙傛暟鏀瑰姩 / 瀛楁鏀瑰姩锛?   - 鍏煎鎬у奖鍝嶏紙鏄惁闇€瑕侀噸璺?factor / model锛?   - 椋庨櫓鐐癸紙鍙兘娉勯湶銆佸彲鑳界█鐤忋€佸彲鑳戒笉绋冲畾锛?
3. 缁撴灉鍥炴姤妯℃澘锛堝缓璁級
   - 鍥犲瓙鐗堟湰鍙凤紙濡?`xxx_v2`锛?   - 鍏抽敭鎸囨爣鎽樿锛圛C/IR/Sharpe锛?   - 鏄惁寤鸿鍏ユā锛堟槸/鍚?+ 鐞嗙敱锛?
鑱岃矗杈圭晫锛堝繀椤伙級锛?
- 鍥犲瓙鐮旂┒ Agent锛?  - 鍙礋璐ｄ骇鍑哄叕寮忎笌瑙勬牸鍗★紙瀛楁銆佸弬鏁般€佸洜鏋滃０鏄庯級
  - 涓嶈礋璐ｈ窇鍥炴祴銆佷笉璐熻矗鏀圭瓫閫夐槇鍊笺€佷笉璐熻矗鍏ユā閰嶇疆鏀瑰姩
- ON 缁存姢 Agent锛?  - 璐熻矗浠ｇ爜钀藉湴銆乥atch 鍥炴祴銆佺瓫閫夐厤缃墽琛屻€佹姤鍛婁骇鐗╀笌鍏ユā瀵规帴

娉ㄦ剰锛?
- 鈥滃彲鎵ц楠岃瘉鍖呪€濓紙鏈€灏忓洖娴嬪尯闂淬€佺洰鏍囬槇鍊笺€佸け璐ュ垽鎹€佸鐓у熀绾匡級鐢遍」鐩?Owner 鍐冲畾銆?- 璇ラ儴鍒嗙敱 Owner 鐩存帴閰嶇疆鍒?`factor_config.screening`锛屾垨鐢?Owner 鏄庣‘鍛婄煡 ON 缁存姢 Agent 鍚庝唬涓洪厤缃€?- 鍥犲瓙鐮旂┒ Agent 涓嶈礋璐ｆ搮鑷慨鏀圭瓫閫夐槇鍊笺€?
---

鍙傝€冩ā鏉匡細

- `cbond_on/domain/factors/defs/ret_window.py`
- `cbond_on/domain/factors/defs/depth_imbalance.py`
- `cbond_on/domain/factors/defs/vwap_gap.py`

## Factor Formula Catalog
- See: docs/factor_expression_catalog.md (active signal formulas, params, required fields).

## 16. Factor Rebuild Switch Semantics
- `refresh = true`: full rebuild in selected date range. The factor day file is rebuilt from current `specs` only.
- `overwrite = true` and `refresh = false`: recompute selected factor columns and overwrite those columns only; keep all other existing columns unchanged.
- `overwrite = false` and `refresh = false`: incremental append mode. Only missing factor columns are computed; existing columns are kept untouched.

## 15. Per-Factor `windowsize` OHLC Rebuild Rule (2026-03-27)
- This is now the **active** intraday bar rebuild convention for ON factor development.
- No global switch is used. Rebuild is controlled **per factor** via params only.

### 15.1 Trigger condition
- Rebuild is enabled only when a factor params contains:
  - `windowsize: <int>`
- Alias `window_size` is also accepted for compatibility.
- If no `windowsize` is provided, the factor keeps legacy snapshot-row behavior.

### 15.2 Scope
- Rebuild affects factor fields with market-bar semantics:
  - price: `open`, `high`, `low`, `close`, `last`
  - flow: `volume`, `amount`, `num_trades`
  - previous close semantics: `prev_bar_close` (alias output `pre_close`)
- Non-bar microstructure fields (depth ladders, `ask_volume*`, `bid_volume*`, etc.) stay as last snapshot of each rebuilt bar.

### 15.3 Rebuild semantics
- Rebuild is **bar-sequence aggregation** by `(dt, code)`, with non-overlap bucket size `windowsize`.
- One bucket produces one rebuilt row (no broadcast back to original snapshot rows).
- Price fields are rebuilt from `last`:
  - `open`: first `last` in bucket
  - `high`: max `last` in bucket
  - `low`: min `last` in bucket
  - `close`: last `last` in bucket
  - `last`: equal to rebuilt `close`
- Flow fields are rebuilt from cumulative snapshot fields:
  - per-snapshot increment = `diff(cum)` with reset protection (`diff<0` fallback to current cumulative),
  - bucket value = sum of increments in bucket.
- `prev_bar_close` is rebuilt as previous bucket `close` in the same `(dt, code)`.
  - first bucket fallback uses same-day source `pre_close` (if available).
  - output keeps compatibility alias: `pre_close == prev_bar_close`.
- When rebuild is enabled, open-like semantics prioritize rebuilt `open`.

### 15.4 Config example
```json5
{
  name: "alpha026_volume_high_rank_corr_v1",
  factor: "alpha026_volume_high_rank_corr_v1",
  params: {
    ts_rank_window: 5,
    corr_window: 5,
    ts_max_window: 3,
    windowsize: 10
  }
}
```

### 15.5 Factors already switched to `windowsize`
- `volatility_scaled_return_v1`
- `volume_price_trend_v1`
- `trade_intensity_v1`
- `premium_momentum_proxy_v1`
- `alpha001_signed_power_v1`
- `alpha002_corr_volume_return_v1`
- `alpha003_corr_open_volume_v1`
- `alpha004_ts_rank_low_v1`
- `alpha005_vwap_gap_v1`
- `alpha006_corr_open_volume_neg_v1`
- `alpha008_open_return_momentum_v1`
- `alpha014_return_open_volume_v1`
- `alpha015_high_volume_corr_v1`
- `alpha016_cov_high_volume_v1`
- `alpha018_close_open_vol_v1`
- `alpha019_close_momentum_sign_v1`
- `alpha020_open_delay_range_v1`
- `alpha023_high_momentum_v1`
- `alpha025_return_volume_vwap_range_v1`
- `alpha026_volume_high_rank_corr_v1`
- `alpha028_adv_low_close_signal_v1`
- `alpha029_complex_rank_signal_v1`
- `alpha031_close_decay_momentum_v1`
- `alpha033_open_close_ratio_v1`
- `alpha034_return_volatility_rank_v1`
- `alpha035_volume_price_momentum_v1`
- `alpha036_complex_correlation_signal_v1`
- `alpha037_open_close_correlation_v1`
- `alpha038_close_rank_ratio_v1`
- `alpha039_volume_decay_momentum_v1`
- `alpha040_high_volatility_corr_v1`
- `alpha052_low_momentum_volume_v1`

### 15.6 Notes for new factor agents
- If your formula depends on OHLCV or previous-close semantics, set `windowsize` explicitly per factor.
- Different factors can use different `windowsize`; no global mandatory value.
- New formulas should use `prev_bar_close`; `pre_close` is kept as compatibility alias only.
- Current recommended starting value: `windowsize = 10` (then tune by factor).



