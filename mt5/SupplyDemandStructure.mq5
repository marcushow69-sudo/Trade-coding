//+------------------------------------------------------------------+
//| SupplyDemandStructure.mq5                                        |
//| XAUUSD: H1 market-structure trend + M5 supply/demand zone entry  |
//|                                                                  |
//| Rules                                                            |
//|  Trend (H1, closed bars): last two confirmed swing highs/lows.   |
//|    HH + HL = bullish, LH + LL = bearish, anything else = no trade|
//|    A close beyond the last HL (bull) / LH (bear) cancels trend.  |
//|  Demand zone (M5): last down candle before a bullish impulse of  |
//|    1-3 up candles that closes >= X*ATR above that candle's high. |
//|  Supply zone (M5): mirror image.                                 |
//|  Entry: market order the moment price first touches the zone,    |
//|    only if the H1 trend agrees. First touch only - the zone is   |
//|    used up on that touch whether a trade was taken or not.       |
//|  SL: 5 pips beyond the zone candle's wick.  TP: 2R (full close). |
//|  Breakeven: SL moved to entry when price reaches 1R.             |
//|  Risk: 1% of balance per trade.                                  |
//+------------------------------------------------------------------+
#property copyright "Trade-coding"
#property version   "1.00"
#property description "H1 market structure + M5 supply/demand zones, 1:2 RR, breakeven at 1R"

#include <Trade/Trade.mqh>

input group "Timeframes"
input ENUM_TIMEFRAMES InpTrendTF            = PERIOD_H1;  // Trend timeframe
input ENUM_TIMEFRAMES InpEntryTF            = PERIOD_M5;  // Entry / zone timeframe

input group "H1 market structure"
input int    InpSwingStrength               = 3;     // Bars each side to confirm a swing high/low
input int    InpStructureLookback           = 300;   // H1 bars searched for swings
input bool   InpBreakCancelsTrend           = true;  // Close beyond last HL/LH -> no trend

input group "M5 supply / demand zones"
input int    InpImpulseMaxCandles           = 3;     // Max candles in the impulse leaving the zone
input double InpImpulseATRMult              = 1.5;   // Impulse must travel >= this x ATR beyond the zone
input int    InpATRPeriod                   = 14;    // ATR period (entry timeframe)
input bool   InpZoneUseBody                 = false; // false = wick-to-wick zone, true = body-to-wick
input double InpZoneMaxHeightATR            = 3.0;   // Skip zones taller than this x ATR (0 = off)
input int    InpZoneMaxAgeBars              = 288;   // Zone expires after N entry bars (288 x M5 = 1 day)

input group "Risk & exits"
input double InpRiskPercent                 = 1.0;   // Risk per trade (% of balance)
input double InpPipSize                     = 0.10;  // 1 pip in price (gold: usually 0.10)
input double InpSLBufferPips                = 5.0;   // SL distance beyond the zone candle's wick (pips)
input double InpRewardRisk                  = 2.0;   // Take profit in R (closes the whole trade)
input double InpBreakevenAtR                = 1.0;   // Move SL to entry at this R (0 = off)
input double InpBreakevenOffsetPips         = 0.0;   // Lock in this many pips at breakeven
input int    InpMaxOpenTrades               = 0;     // Max open trades (0 = unlimited)
input double InpMaxSpreadPips               = 0.0;   // Skip entries if spread is wider (0 = off)

input group "Session (broker server time)"
input int    InpSessionStartHour            = 0;     // Start hour (inclusive)
input int    InpSessionEndHour              = 24;    // End hour (exclusive); 0-24 = all sessions

input group "Other"
input ulong  InpMagic                       = 260518; // Magic number
input bool   InpDrawZones                   = true;   // Draw zones on the chart

struct Zone
  {
   int      dir;        // +1 demand (buy), -1 supply (sell)
   double   top;
   double   bottom;
   datetime baseTime;   // time of the zone-forming candle
   datetime created;
  };

CTrade   trade;
Zone     zones[];
int      atrHandle   = INVALID_HANDLE;
int      trendDir    = 0;
datetime lastTrendBar = 0;
datetime lastEntryBar = 0;
string   PREFIX      = "SDZ_";

//+------------------------------------------------------------------+
int OnInit()
  {
   if(InpPipSize <= 0 || InpRewardRisk <= 0 || InpRiskPercent <= 0)
     {
      Print("Invalid inputs: pip size, reward:risk and risk % must be > 0");
      return INIT_PARAMETERS_INCORRECT;
     }
   atrHandle = iATR(_Symbol, InpEntryTF, InpATRPeriod);
   if(atrHandle == INVALID_HANDLE)
      return INIT_FAILED;
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetTypeFillingBySymbol(_Symbol);
   trade.SetDeviationInPoints(50);
   ArrayResize(zones, 0);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(atrHandle != INVALID_HANDLE)
      IndicatorRelease(atrHandle);
   if(InpDrawZones)
      ObjectsDeleteAll(0, PREFIX);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   datetime trendBar = iTime(_Symbol, InpTrendTF, 0);
   if(trendBar != lastTrendBar)
     {
      trendDir = TrendDirection();
      lastTrendBar = trendBar;
     }
   if(InpBreakCancelsTrend && trendDir != 0)
      trendDir = StructureStillValid(trendDir) ? trendDir : 0;

   datetime entryBar = iTime(_Symbol, InpEntryTF, 0);
   if(entryBar != lastEntryBar)
     {
      lastEntryBar = entryBar;
      DetectZones();
     }

   ManageBreakeven();
   CheckZoneTouches();
  }

//+------------------------------------------------------------------+
//| H1 market structure                                              |
//+------------------------------------------------------------------+
bool IsSwingHigh(int i)
  {
   double h = iHigh(_Symbol, InpTrendTF, i);
   for(int k = 1; k <= InpSwingStrength; k++)
      if(iHigh(_Symbol, InpTrendTF, i - k) >= h || iHigh(_Symbol, InpTrendTF, i + k) > h)
         return false;
   return true;
  }

bool IsSwingLow(int i)
  {
   double l = iLow(_Symbol, InpTrendTF, i);
   for(int k = 1; k <= InpSwingStrength; k++)
      if(iLow(_Symbol, InpTrendTF, i - k) <= l || iLow(_Symbol, InpTrendTF, i + k) < l)
         return false;
   return true;
  }

double g_lastSwingHigh = 0, g_lastSwingLow = 0;

int TrendDirection()
  {
   double sh[2], sl[2];
   int nh = 0, nl = 0;
   int maxBar = MathMin(InpStructureLookback, iBars(_Symbol, InpTrendTF) - InpSwingStrength - 1);
   // start at strength+1 so every bar on the right of a swing is already closed
   for(int i = InpSwingStrength + 1; i < maxBar && (nh < 2 || nl < 2); i++)
     {
      if(nh < 2 && IsSwingHigh(i))
         sh[nh++] = iHigh(_Symbol, InpTrendTF, i);
      if(nl < 2 && IsSwingLow(i))
         sl[nl++] = iLow(_Symbol, InpTrendTF, i);
     }
   if(nh < 2 || nl < 2)
      return 0;
   g_lastSwingHigh = sh[0];
   g_lastSwingLow  = sl[0];
   if(sh[0] > sh[1] && sl[0] > sl[1])
      return 1;    // HH + HL
   if(sh[0] < sh[1] && sl[0] < sl[1])
      return -1;   // LH + LL
   return 0;
  }

// Uptrend is broken once an H1 candle closes below the last higher low (and vice versa).
bool StructureStillValid(int dir)
  {
   double lastClose = iClose(_Symbol, InpTrendTF, 1);
   if(dir == 1)
      return lastClose >= g_lastSwingLow;
   return lastClose <= g_lastSwingHigh;
  }

//+------------------------------------------------------------------+
//| M5 zone detection (runs once per new entry-timeframe bar)        |
//+------------------------------------------------------------------+
bool AllImpulse(int base, int dir)
  {
   for(int k = 1; k < base; k++)
     {
      double o = iOpen(_Symbol, InpEntryTF, k), c = iClose(_Symbol, InpEntryTF, k);
      if(dir == 1 && c <= o)
         return false;
      if(dir == -1 && c >= o)
         return false;
     }
   return true;
  }

void DetectZones()
  {
   double atr[];
   if(CopyBuffer(atrHandle, 0, 1, 1, atr) != 1 || atr[0] <= 0)
      return;
   double a = atr[0];
   double lastClose = iClose(_Symbol, InpEntryTF, 1);

   for(int b = 2; b <= InpImpulseMaxCandles + 1; b++)
     {
      double o = iOpen(_Symbol, InpEntryTF, b), c = iClose(_Symbol, InpEntryTF, b);
      double h = iHigh(_Symbol, InpEntryTF, b), l = iLow(_Symbol, InpEntryTF, b);
      datetime t = iTime(_Symbol, InpEntryTF, b);

      // Demand: down candle, then up-candle impulse closing well above its high
      if(c <= o && AllImpulse(b, 1) && lastClose - h >= InpImpulseATRMult * a)
         AddZone(1, InpZoneUseBody ? o : h, l, t, a);

      // Supply: up candle, then down-candle impulse closing well below its low
      if(c >= o && AllImpulse(b, -1) && l - lastClose >= InpImpulseATRMult * a)
         AddZone(-1, h, InpZoneUseBody ? o : l, t, a);
     }
   ExpireZones();
  }

void AddZone(int dir, double top, double bottom, datetime baseTime, double atr)
  {
   if(top <= bottom)
      return;
   if(InpZoneMaxHeightATR > 0 && top - bottom > InpZoneMaxHeightATR * atr)
      return;
   for(int i = 0; i < ArraySize(zones); i++)
      if(zones[i].baseTime == baseTime && zones[i].dir == dir)
         return;   // already known

   int n = ArraySize(zones);
   ArrayResize(zones, n + 1);
   zones[n].dir = dir;
   zones[n].top = top;
   zones[n].bottom = bottom;
   zones[n].baseTime = baseTime;
   zones[n].created = TimeCurrent();
   DrawZone(zones[n], dir == 1 ? clrSeaGreen : clrCrimson);
  }

void ExpireZones()
  {
   long maxAge = (long)InpZoneMaxAgeBars * PeriodSeconds(InpEntryTF);
   for(int i = ArraySize(zones) - 1; i >= 0; i--)
      if(TimeCurrent() - zones[i].created > maxAge)
        {
         DrawZone(zones[i], clrDimGray);
         ArrayRemove(zones, i, 1);
        }
  }

//+------------------------------------------------------------------+
//| Entries: first touch of a zone                                   |
//+------------------------------------------------------------------+
void CheckZoneTouches()
  {
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   for(int i = ArraySize(zones) - 1; i >= 0; i--)
     {
      Zone z = zones[i];
      bool touched = (z.dir == 1 && bid <= z.top) || (z.dir == -1 && bid >= z.bottom);
      if(!touched)
         continue;

      // First touch only: the zone is consumed now, trade or not.
      ArrayRemove(zones, i, 1);
      bool gappedThrough = (z.dir == 1 && bid < z.bottom) || (z.dir == -1 && bid > z.top);
      if(gappedThrough || z.dir != trendDir || !CanOpenTrade())
        {
         DrawZone(z, clrDimGray);
         continue;
        }
      DrawZone(z, clrGold);
      OpenTrade(z);
     }
  }

bool InSession()
  {
   if(InpSessionStartHour <= 0 && InpSessionEndHour >= 24)
      return true;
   MqlDateTime t;
   TimeToStruct(TimeCurrent(), t);
   if(InpSessionStartHour <= InpSessionEndHour)
      return t.hour >= InpSessionStartHour && t.hour < InpSessionEndHour;
   return t.hour >= InpSessionStartHour || t.hour < InpSessionEndHour;   // overnight session
  }

int CountMyPositions()
  {
   int n = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol &&
         (ulong)PositionGetInteger(POSITION_MAGIC) == InpMagic)
         n++;
     }
   return n;
  }

bool CanOpenTrade()
  {
   if(!InSession())
      return false;
   if(InpMaxOpenTrades > 0 && CountMyPositions() >= InpMaxOpenTrades)
      return false;
   if(InpMaxSpreadPips > 0)
     {
      double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
      if(spread > InpMaxSpreadPips * InpPipSize)
         return false;
     }
   return true;
  }

double LotsForRisk(ENUM_ORDER_TYPE type, double entry, double sl)
  {
   double money = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0;
   double pnl = 0;
   if(!OrderCalcProfit(type, _Symbol, 1.0, entry, sl, pnl) || pnl >= 0)
      return 0;
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lots = MathFloor(money / -pnl / step) * step;
   if(lots < vmin)
     {
      PrintFormat("Skip: 1%% risk needs %.3f lots, below broker minimum %.2f", money / -pnl, vmin);
      return 0;   // never risk more than the limit just to meet the minimum lot
     }
   lots = MathMin(lots, vmax);
   double margin = 0;
   while(lots >= vmin && OrderCalcMargin(type, _Symbol, lots, entry, margin) &&
         margin > AccountInfoDouble(ACCOUNT_MARGIN_FREE))
      lots -= step;
   if(lots < vmin)
      return 0;
   int digits = (int)MathMax(0, MathCeil(-MathLog10(step)));
   return NormalizeDouble(lots, digits);
  }

void OpenTrade(const Zone &z)
  {
   bool   buy   = (z.dir == 1);
   double price = buy ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double buf   = InpSLBufferPips * InpPipSize;
   double sl    = buy ? z.bottom - buf : z.top + buf;
   double risk  = MathAbs(price - sl);
   if(risk <= 0 || (buy && sl >= price) || (!buy && sl <= price))
      return;
   double tp = buy ? price + InpRewardRisk * risk : price - InpRewardRisk * risk;

   double minDist = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   if(risk < minDist)
      return;

   ENUM_ORDER_TYPE type = buy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   double lots = LotsForRisk(type, price, sl);
   if(lots <= 0)
      return;

   sl = NormalizeDouble(sl, _Digits);
   tp = NormalizeDouble(tp, _Digits);
   string comment = buy ? "SD demand" : "SD supply";
   bool ok = buy ? trade.Buy(lots, _Symbol, 0, sl, tp, comment)
                 : trade.Sell(lots, _Symbol, 0, sl, tp, comment);
   if(!ok)
      PrintFormat("Order failed: %d %s", trade.ResultRetcode(), trade.ResultRetcodeDescription());
  }

//+------------------------------------------------------------------+
//| Breakeven at 1R                                                  |
//+------------------------------------------------------------------+
void ManageBreakeven()
  {
   if(InpBreakevenAtR <= 0)
      return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double minDist = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || PositionGetString(POSITION_SYMBOL) != _Symbol ||
         (ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      double open = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl   = PositionGetDouble(POSITION_SL);
      double tp   = PositionGetDouble(POSITION_TP);
      if(tp == 0)
         continue;
      double R = MathAbs(tp - open) / InpRewardRisk;   // initial risk, recovered from the 2R target
      double offset = InpBreakevenOffsetPips * InpPipSize;

      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
        {
         double be = NormalizeDouble(open + offset, _Digits);
         if(bid >= open + InpBreakevenAtR * R && sl < be - _Point / 2 && bid - be >= minDist)
            trade.PositionModify(ticket, be, tp);
        }
      else
        {
         double be = NormalizeDouble(open - offset, _Digits);
         if(ask <= open - InpBreakevenAtR * R && (sl == 0 || sl > be + _Point / 2) && be - ask >= minDist)
            trade.PositionModify(ticket, be, tp);
        }
     }
  }

//+------------------------------------------------------------------+
void DrawZone(const Zone &z, color clr)
  {
   if(!InpDrawZones)
      return;
   string name = PREFIX + (z.dir == 1 ? "D_" : "S_") + IntegerToString((long)z.baseTime);
   datetime t2 = z.baseTime + (long)InpZoneMaxAgeBars * PeriodSeconds(InpEntryTF);
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_RECTANGLE, 0, z.baseTime, z.top, t2, z.bottom);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
  }
//+------------------------------------------------------------------+
