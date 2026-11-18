from pathlib import Path

code = r'''import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Stock Hunter", page_icon="🎯", layout="wide")
st.title("🎯 Stock Hunter")
st.caption("ماسح الأسهم الأمريكية | أفضل 3 للاستثمار وأفضل 3 للصفقات القصيرة")
st.warning("البيانات من Yahoo Finance وقد تتأخر أو تنقص. النتائج فرز آلي أولي وليست توصية استثمارية.")

with st.sidebar:
    st.header("إعدادات الفحص")
    max_price = st.number_input("أقصى سعر للسهم ($)", 1.0, 100.0, 20.0, 1.0)
    capital = st.number_input("رأس المال ($)", 1.0, 10000.0, 75.0, 5.0)
    max_symbols = st.slider("عدد الأسهم المراد فحصها", 20, 200, 80, 20)
    refresh = st.button("🔄 تحديث البيانات الآن", use_container_width=True)
    st.caption("يتم التحديث عند فتح الصفحة أو الضغط على الزر. التشغيل اليومي التلقائي يحتاج استضافة وجدولة.")

UNIVERSE = list(dict.fromkeys("""
AAL AAPL ABEV ACHR ACHV ADMA AEHR AEO AG AGNC AI AIG ALAB ALGM AMCR AMD AMTX AMZN APLD APPS AR ARCC ASTS AUR AVGO AVTR BBAI BB BBD BBIO BCRX BHC BITF BKR BMY BTG C CAN CCL CHPT CLF CLSK CMCSA CLOV CMRX CNC CNEY COTY CROX CRSP CSCO CTRA CVS CX D DAL DB DKNG DNA DOC DVN EBS EDIT ET ETSY EVGO F FCX FIS FITB FOLD FRO FUBO GEVO GGB GILD GLW GM GME GOLD GOOGL GRAB GRPN HAL HBI HL HPE HPQ HUT IBRX INTC IONQ IQ JOBY JBLU JD JKS KGC KHC KIM KMI LCID LUMN LUNR LYFT M MARA MAT MBLY MCHP MGM MNKD MPW MRNA MRO MSFT MSTR NCLH NIO NOK NU NVAX NVDA NVTS O OCGN OPEN ORCL PBR PFE PLUG PLTR PSEC PTON QBTS RBLX RIG RIVN RKLB RUN S SIRI SOFI SOUN SPCE T TEVA TFC TGT TMC TSLA TU UAL UBER U VALE VFC VIPS VZ WBD WFC WULF X XOM YMM
""".split()))[:max_symbols]

def num(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def row_values(frame, names):
    if frame is None or frame.empty:
        return []
    for name in names:
        if name in frame.index:
            try:
                return [num(v) for v in frame.loc[name].tolist() if num(v) is not None]
            except Exception:
                pass
    return []

def increasing_3(values):
    # Yahoo annual statements are normally newest-first.
    return len(values) >= 3 and values[2] < values[1] < values[0] and all(v > 0 for v in values[:3])

def analyze(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y", interval="1d", auto_adjust=True)
        if hist is None or len(hist) < 60:
            return None
        price = num(hist["Close"].iloc[-1])
        if price is None or price <= 0 or price > max_price:
            return None
        info = t.get_info()
        close, vol = hist["Close"], hist["Volume"]
        ma20, ma50 = num(close.tail(20).mean()), num(close.tail(50).mean())
        ma200 = num(close.tail(200).mean()) if len(close) >= 200 else None
        avgvol = num(vol.tail(20).mean()) or 0
        dollarvol = avgvol * price
        ret20 = num((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 21 else None
        ret60 = num((close.iloc[-1] / close.iloc[-61] - 1) * 100) if len(close) > 61 else None
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = -delta.clip(upper=0).rolling(14).mean().iloc[-1]
        rsi = num(100 - 100 / (1 + gain / loss)) if pd.notna(loss) and loss > 0 else None

        fin, bal, cf = t.financials, t.balance_sheet, t.cashflow
        revs = row_values(fin, ["Total Revenue"])
        profits = row_values(fin, ["Net Income", "Net Income Common Stockholders"])
        eps_values = row_values(fin, ["Diluted EPS", "Basic EPS"])
        revenue = revs[0] if revs else None
        net = profits[0] if profits else None
        eps = num(info.get("trailingEps"))
        rev_growth = ((revs[0] / revs[1]) - 1) * 100 if len(revs) >= 2 and revs[1] else None
        eps_growth = ((eps_values[0] / eps_values[1]) - 1) * 100 if len(eps_values) >= 2 and eps_values[1] else None
        profit_growth = ((profits[0] / profits[1]) - 1) * 100 if len(profits) >= 2 and profits[1] else None
        profit_3y = increasing_3(profits)
        revenue_3y = increasing_3(revs)
        eps_3y = increasing_3(eps_values)

        gross = None
        operating = None
        op_cash = None
        try:
            gross = num(fin.loc["Gross Profit"].iloc[0])
            operating = num(fin.loc["Operating Income"].iloc[0])
        except Exception:
            pass
        try:
            op_cash = num(cf.loc["Operating Cash Flow"].iloc[0])
        except Exception:
            pass
        debt = num(info.get("totalDebt"))
        equity = num(info.get("totalStockholderEquity"))
        de = debt / equity if debt is not None and equity and equity > 0 else None
        pe = num(info.get("trailingPE"))
        margin = num(info.get("grossMargins"))
        current_ratio = num(info.get("currentRatio"))

        fundamentals = [
            revenue is not None and revenue > 0,
            gross is not None and gross > 0,
            operating is not None and operating > 0,
            net is not None and net > 0,
            eps is not None and eps > 0,
            op_cash is not None and op_cash > 0,
            rev_growth is not None and rev_growth > 0,
            profit_growth is not None and profit_growth > 0,
            current_ratio is not None and current_ratio >= 1.2,
            de is not None and de < 1.5,
            margin is not None and margin > 0.2,
            pe is not None and 0 < pe < 30,
        ]
        fscore = 60 * sum(fundamentals) / len(fundamentals)
        technicals = [
            ma20 is not None and price > ma20,
            ma50 is not None and price > ma50,
            ma200 is not None and price > ma200,
            ret20 is not None and ret20 > 0,
            ret60 is not None and ret60 > 0,
            dollarvol >= 1_000_000,
            rsi is not None and 45 <= rsi <= 70,
        ]
        tscore = 40 * sum(technicals) / len(technicals)
        fair_value = eps * min(max(pe, 8), 25) if eps and eps > 0 and pe and pe > 0 else None
        return {
            "Ticker": symbol, "Company": info.get("shortName", symbol),
            "Price": round(price, 2), "Long score": round(fscore + tscore, 1),
            "Short score": round(0.45 * fscore + 0.55 * tscore, 1),
            "Revenue": revenue, "Net income": net, "EPS": eps,
            "Revenue growth %": rev_growth, "EPS growth %": eps_growth,
            "Net income growth %": profit_growth, "Profit increasing 3Y": profit_3y,
            "Revenue increasing 3Y": revenue_3y, "EPS increasing 3Y": eps_3y,
            "P/E": pe, "Fair value (rough)": fair_value,
            "Gross margin": margin, "Current ratio": current_ratio,
            "Debt/Equity": de, "Operating cash flow": op_cash,
            "Avg daily $ volume": dollarvol, "20d return %": ret20,
            "60d return %": ret60, "RSI": rsi, "MA20": ma20, "MA50": ma50, "MA200": ma200
        }
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def scan(symbols):
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(analyze, s) for s in symbols]
        for future in as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)
    return pd.DataFrame(rows)

if refresh:
    st.cache_data.clear()
with st.spinner("جاري فحص الأسهم وجلب البيانات..."):
    data = scan(UNIVERSE)

if data.empty:
    st.error("لم تصل بيانات كافية. جرّب التحديث لاحقًا.")
    st.stop()

data["Affordable shares"] = (capital / data["Price"]).apply(lambda x: max(0, int(x)))
data["Entry zone"] = (data["Price"] * .985).round(2).astype(str) + " – " + (data["Price"] * 1.01).round(2).astype(str)
data["Stop loss (reference)"] = (data["Price"] * .92).round(2)
data["Target 1 (reference)"] = (data["Price"] * 1.10).round(2)
data["Target 2 (reference)"] = (data["Price"] * 1.20).round(2)

c1, c2, c3 = st.columns(3)
c1.metric("الشركات التي اجتازت فحص السعر", len(data))
c2.metric("رأس المال", f"${capital:,.0f}")
c3.metric("آخر تحديث", datetime.now().strftime("%Y-%m-%d %H:%M"))

long_candidates = data[data["Profit increasing 3Y"] == True].copy()
tabs = st.tabs(["🏆 أفضل 3 للاستثمار", "⚡ أفضل 3 للصفقات القصيرة", "🔎 كل النتائج", "📘 المنهجية"])

with tabs[0]:
    st.subheader("أفضل 3 للاستثمار طويل الأجل")
    st.caption("شرط إلزامي: صافي الأرباح موجب ومتزايد في آخر 3 سنوات مالية. إذا كانت بيانات السنوات الثلاث غير مكتملة، لا يجتاز السهم الشرط.")
    if long_candidates.empty:
        st.info("لا توجد شركات اجتازت شرط نمو صافي الأرباح لثلاث سنوات ببيانات مكتملة في هذه الجولة.")
    else:
        for _, r in long_candidates.sort_values("Long score", ascending=False).head(3).iterrows():
            with st.container(border=True):
                a, b, c = st.columns([2, 1, 1])
                a.subheader(f"{r['Ticker']} — {r['Company']}")
                a.write(f"السعر: **${r['Price']:.2f}** | منطقة متابعة: **{r['Entry zone']}**")
                b.metric("التقييم", f"{r['Long score']}/100")
                c.metric("عدد الأسهم النظري", r["Affordable shares"])
                st.write(f"وقف مرجعي: ${r['Stop loss (reference)']:.2f} | هدف 1: ${r['Target 1 (reference)']:.2f} | هدف 2: ${r['Target 2 (reference)']:.2f}")
                st.write(f"نمو الإيرادات: {r['Revenue growth %']:.1f}%" if pd.notna(r["Revenue growth %"]) else "نمو الإيرادات: غير متاح")
                st.write(f"نمو EPS: {r['EPS growth %']:.1f}%" if pd.notna(r["EPS growth %"]) else "نمو EPS: غير متاح")
                st.caption("مستويات السعر آلية مرجعية وليست دعمًا أو مقاومة مؤكدة. تحقق من دعم وسيطك للأسهم الكسرية.")

with tabs[1]:
    st.subheader("أفضل 3 للصفقات القصيرة")
    st.caption("شرط نمو صافي الأرباح لثلاث سنوات لا يُطبق على المضاربة قصيرة الأجل.")
    for _, r in data.sort_values("Short score", ascending=False).head(3).iterrows():
        st.write(f"**{r['Ticker']} — {r['Company']}** | ${r['Price']:.2f} | التقييم {r['Short score']}/100 | عائد 20 يوم: {r['20d return %'] if pd.notna(r['20d return %']) else 'غير متاح'}%")

with tabs[2]:
    cols = ["Ticker", "Company", "Price", "Long score", "Short score", "Profit increasing 3Y", "Revenue growth %", "EPS growth %", "Net income growth %", "P/E", "Fair value (rough)", "Avg daily $ volume", "RSI"]
    st.dataframe(data[cols].sort_values("Long score", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("تحميل النتائج CSV", data.to_csv(index=False).encode("utf-8-sig"), "stock_hunter_scan.csv", "text/csv")

with tabs[3]:
    st.markdown("""
### طريقة الفحص
- **الاستثمار طويل الأجل:** لا يدخل القائمة إلا السهم الذي لديه صافي أرباح موجب ومتزايد سنة بعد سنة خلال آخر 3 سنوات مالية، مع توافر السنوات الثلاث.
- **المضاربة القصيرة:** ترتيب فني وزخم، دون اشتراط نمو الأرباح لثلاث سنوات.
- **التحليل المالي:** الإيرادات، صافي الربح، EPS، التدفق النقدي، السيولة، الديون، هامش الربح ومضاعف الربحية.
- **التحليل الفني:** متوسطات 20/50/200 يوم، عائد 20 و60 يوم، حجم التداول وRSI.
- **القيمة العادلة التقريبية:** تقدير مبسط جدًا وليس نموذج تقييم مالي مستقل.

### حدود النسخة
1. قائمة الأسهم الأمريكية الحالية محدودة وليست كل السوق.
2. Yahoo Finance قد يعرض بيانات ناقصة أو متأخرة؛ لذلك لا تُعد البيانات الناقصة اجتيازًا لشرط السنوات الثلاث.
3. لا يوجد حتى الآن فلتر شرعي موثوق أو تحقق آلي من نشاط الشركة ونسبها المالية.
4. التحديث عند فتح الصفحة أو الضغط على الزر، وليس تشغيلًا يوميًا تلقائيًا.
5. ليست توصية شراء أو بيع؛ راجع التقارير المالية ومخاطر كل شركة بنفسك.
""")
st.caption("Stock Hunter • نسخة تطويرية")
'''

path = Path("/mnt/data/app.py")
path.write_text(code, encoding="utf-8")
print(f"تم تجهيز الملف: {path.name} — الحجم {path.stat().st_size:,} بايت")
