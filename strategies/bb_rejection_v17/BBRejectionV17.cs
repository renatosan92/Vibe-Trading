// BB Rejection v17 — NinjaTrader 8 strategy for MNQ on 15-minute bars.
// Candle A pierces a Bollinger Band with an extreme stochastic, Candle B confirms
// with an opposite-colour body, and the trade is entered at the open of Candle C.
//
// Install: NinjaTrader 8 → New → NinjaScript Editor → Strategies (right-click) →
// New Strategy… → name it BBRejectionV17 → replace the whole file with this code → F5 to compile.
#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class BBRejectionV17 : Strategy
    {
        private Bollinger bb;
        private Stochastics stoch;
        private TimeZoneInfo nyTz;
        private DateTime lastTradeDay = DateTime.MinValue;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                         = "BBRejectionV17";
                Description                  = "BB Rejection v17: Bollinger 20/2 + Stoch 14-1-3 reversal, MNQ 15m";
                Calculate                    = Calculate.OnBarClose;
                EntriesPerDirection          = 1;
                EntryHandling                = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;     // safety net; the 16:00 NY exit below fires first
                ExitOnSessionCloseSeconds    = 30;
                BarsRequiredToTrade          = 20;
                StartBehavior                = StartBehavior.WaitUntilFlat;
                TimeInForce                  = TimeInForce.Gtc;
                IsInstantiatedOnEachOptimizationIteration = true;

                Quantity    = 15;
                BBPeriod    = 20;
                BBStdDev    = 2.0;
                StochK      = 14;
                StochSmooth = 1;
                StochD      = 3;
                StochHigh   = 80;
                StochLow    = 10;
                TPTicks     = 135;
                SLTicks     = 265;
                StartHHMM   = 1000;   // NY time, Candle C open must be >= this
                EndHHMM     = 1600;   // NY time, Candle C open must be < this; forced exit
            }
            else if (State == State.Configure)
            {
                SetProfitTarget(CalculationMode.Ticks, TPTicks);
                SetStopLoss(CalculationMode.Ticks, SLTicks);
            }
            else if (State == State.DataLoaded)
            {
                bb    = Bollinger(BBStdDev, BBPeriod);
                stoch = Stochastics(StochD, StochK, StochSmooth);
                nyTz  = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                AddChartIndicator(bb);
                AddChartIndicator(stoch);
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0 || CurrentBar < BarsRequiredToTrade)
                return;

            // NinjaTrader stamps bars with their CLOSE time, so Time[0] — the close of
            // Candle B (this bar) — is exactly the open time of Candle C.
            DateTime cOpenNy = TimeZoneInfo.ConvertTime(Time[0], Core.Globals.GeneralOptions.TimeZoneInfo, nyTz);
            int hhmm = cOpenNy.Hour * 100 + cOpenNy.Minute;

            // Forced exit: market order at the open of the 16:00 NY bar.
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                if (hhmm >= EndHHMM)
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong("16:00 flat", "Long");
                    else
                        ExitShort("16:00 flat", "Short");
                }
                return;
            }

            if (hhmm < StartHHMM || hhmm >= EndHHMM || cOpenNy.Date == lastTradeDay)
                return;

            // Candle A = bar [1], Candle B = bar [0].
            bool shortA = High[1] >= bb.Upper[1] && stoch.K[1] > StochHigh;
            bool longA  = Low[1]  <= bb.Lower[1] && stoch.K[1] < StochLow;
            bool bRed   = Close[0] < Open[0];
            bool bGreen = Close[0] > Open[0];

            if (shortA && bRed)
            {
                EnterShort(Quantity, "Short");
                lastTradeDay = cOpenNy.Date;
                Draw.ArrowDown(this, "S" + CurrentBar, true, 0, High[0] + 4 * TickSize, Brushes.Red);
            }
            else if (longA && bGreen)
            {
                EnterLong(Quantity, "Long");
                lastTradeDay = cOpenNy.Date;
                Draw.ArrowUp(this, "L" + CurrentBar, true, 0, Low[0] - 4 * TickSize, Brushes.LimeGreen);
            }
        }

        #region Properties
        [NinjaScriptProperty, Range(1, int.MaxValue)]
        [Display(Name = "Contracts", Order = 1, GroupName = "Trade")]
        public int Quantity { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue)]
        [Display(Name = "Take profit (ticks)", Order = 2, GroupName = "Trade")]
        public int TPTicks { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue)]
        [Display(Name = "Stop loss (ticks)", Order = 3, GroupName = "Trade")]
        public int SLTicks { get; set; }

        [NinjaScriptProperty, Range(0, 2359)]
        [Display(Name = "Entry window start (HHMM NY)", Order = 4, GroupName = "Trade")]
        public int StartHHMM { get; set; }

        [NinjaScriptProperty, Range(0, 2359)]
        [Display(Name = "Entry window end / forced exit (HHMM NY)", Order = 5, GroupName = "Trade")]
        public int EndHHMM { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue)]
        [Display(Name = "BB period", Order = 1, GroupName = "Indicators")]
        public int BBPeriod { get; set; }

        [NinjaScriptProperty, Range(0.1, double.MaxValue)]
        [Display(Name = "BB std dev", Order = 2, GroupName = "Indicators")]
        public double BBStdDev { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue)]
        [Display(Name = "Stoch %K length", Order = 3, GroupName = "Indicators")]
        public int StochK { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue)]
        [Display(Name = "Stoch %K smoothing", Order = 4, GroupName = "Indicators")]
        public int StochSmooth { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue)]
        [Display(Name = "Stoch %D", Order = 5, GroupName = "Indicators")]
        public int StochD { get; set; }

        [NinjaScriptProperty, Range(0, 100)]
        [Display(Name = "Short: stoch above", Order = 6, GroupName = "Indicators")]
        public double StochHigh { get; set; }

        [NinjaScriptProperty, Range(0, 100)]
        [Display(Name = "Long: stoch below", Order = 7, GroupName = "Indicators")]
        public double StochLow { get; set; }
        #endregion
    }
}
