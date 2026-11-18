# SupplyDemandStructure: MT5 Expert Advisor

This Expert Advisor (EA) automates your XAUUSD strategy: H1 market structure for the trend,
M5 supply and demand zones for entries.

## Install

1. In MT5, open **File → Open Data Folder**, then go to `MQL5/Experts/`.
2. Copy `SupplyDemandStructure.mq5` into that folder.
3. In MT5, open **Tools → MetaQuotes Language Editor** (F4), open the file and press **Compile** (F7).
   You should see `0 errors`. If there are any, send me the error lines.
4. Back in MT5, the EA appears in **Navigator → Expert Advisors**.

## Backtest in the Strategy Tester

1. Open **View → Strategy Tester** (Ctrl+R).
2. Expert: `SupplyDemandStructure`. Symbol: `XAUUSD` (or your broker's gold symbol, e.g. `XAUUSDm`, `GOLD`).
3. Timeframe: **M5**. The EA reads H1 by itself.
4. Modelling: **Every tick based on real ticks**. This matters because entries happen the moment
   price touches a zone, and breakeven at 1R is also intrabar.
5. Date range: start with the last 1–2 years. Deposit: e.g. 10,000 USD.
6. Press **Start**. Afterwards check the **Backtest**, **Graph** and **Trades** tabs.
   Tick **Visual mode** to watch the zones being drawn and traded.

## How each rule is coded

| Your rule | How the EA does it | Setting |
|---|---|---|
| H1 trend by market structure | Swing highs/lows need 3 H1 candles on each side. The last two swing highs and lows give **HH + HL = bullish**, **LH + LL = bearish**, anything else = no trades. An H1 close below the last HL (or above the last LH) cancels the trend until structure re-forms. | `InpSwingStrength`, `InpBreakCancelsTrend` |
| M5 demand zone | The **last down candle before a strong move up** of 1–3 up candles that closes at least **1.5× ATR** above that candle's high. The zone runs from the candle's low to its high (wick to wick). | `InpImpulseMaxCandles`, `InpImpulseATRMult`, `InpZoneUseBody` |
| M5 supply zone | Mirror image: the last up candle before a strong move down. | same |
| Buy / sell setup | Buy only at demand zones while H1 is bullish; sell only at supply zones while H1 is bearish. | – |
| Entry | Market order the moment price **first touches** the zone. | – |
| First touch only | The zone is used up on its first touch, even if no trade was taken (e.g. the trend didn't agree). | – |
| Stop loss | **5 pips beyond the zone candle's wick**. 1 pip = $0.10 on gold, so the stop is $0.50 below the low (buys) or above the high (sells). | `InpSLBufferPips`, `InpPipSize` |
| Take profit | **2R**, closing the whole trade at once. | `InpRewardRisk` |
| Breakeven | At **1R**, the stop moves to the entry price. | `InpBreakevenAtR` |
| Risk | **1% of balance**. The lot size is calculated from the stop distance; if even the broker's minimum lot would risk more than 1%, the trade is skipped. | `InpRiskPercent` |
| Max trades | Unlimited. | `InpMaxOpenTrades` (0 = unlimited) |
| Session | All sessions. | `InpSessionStartHour`, `InpSessionEndHour` (broker server time) |
| Zone lifetime | A zone expires after 1 day (288 M5 bars) if it is never touched. | `InpZoneMaxAgeBars` |

Things to check against your own trading. Each is an input you can change:

* **Pip size:** most brokers count 1 pip on gold as $0.10. If yours uses $0.01, set `InpPipSize = 0.01`.
* **Zone definition:** if your zones are drawn body-to-wick, set `InpZoneUseBody = true`.
  If you want more zones, lower `InpImpulseATRMult` (e.g. 1.0); for fewer, cleaner ones, raise it.
* **Very tall zones** (more than 3× ATR) are skipped because the stop would be huge. Set `InpZoneMaxHeightATR = 0` to allow them.

## Before going live

Run it on a **demo account** for a few weeks first, and compare its trades with the ones you would
have taken by hand.
