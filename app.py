import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Stock Hunter", page_icon="🎯", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.4rem}
.metric-card{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:14px}
.small{color:#888;font-size:.85rem}
</style>
""", unsafe_allow_html=True)

st.title("🎯 Stock Hunter")
st.caption("ماسح الأسهم الأمريكية | تحليل استثماري + فرص قصيرة الأجل")
st.warning("نسخة تشغيل أولية: البيانات من Yahoo Finance عبر yfinance وقد تتأخر أو تنقص. ليست توصية شراء. لا تنفّذ صفقة اعتمادًا على نتيجة آلية وحدها.")

with st.sidebar:
    st.header("إعدادات الفحص")
    max_price = st.number_input("أقصى سعر للسهم ($)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
    capital = st.number_input("رأس المال ($)", min_value=1.0, max_value=100.0, value=75.0, step=5.0)
    max_symbols = st.slider("عدد الأسهم المراد فحصها", 20, 200, 80, 20)
    refresh = st.button("🔄 تحديث البيانات الآن", use_container_width=True)
    st.caption("التحديث يحدث عند فتح الصفحة أو الضغط على الزر. الاستضافة الدائمة مطلوبة للتشغيل التلقائي يوميًا.")

# Large, liquid US-listed universe; screening universe is deliberately transparent and editable.
UNIVERSE = [
"AAL","AAPL","ABEV","ACHR","ACHV","ADMA","AEHR","AEO","AFCG","AG","AGNC","AI","AIG","AKRO","ALAB",
"ALGM","ALLR","AMCR","AMD","AMTX","AMZN","APLD","APPS","AR","ARCC","ASTS","AUR","AVGO","AVTR","BBAI",
"BB","BBD","BBIO","BCRX","BHC","BITF","BKR","BMY","BTG","C","CAN","CARA","CCL","CHPT","CLF","CLSK",
"CMCSA","CLOV","CMRX","CNC","CNEY","COTY","CPB","CROX","CRSP","CSCO","CTRA","CTSH","CURI","CVS","CX",
"D","DAL","DB","DKNG","DNA","DOC","DVN","EBS","EDIT","EEM","ET","ETSY","EVGO","F","FCX","FIS","FITB",
"FOLD","FRO","FUBO","GEVO","GGB","GILD","GLW","GM","GME","GOLD","GOOGL","GRAB","GRPN","HAL","HBI",
"HL","HPE","HPQ","HUT","IBRX","INTC","IONQ","IQ","JOBY","JBLU","JD","JKS","KGC","KHC","KIM","KMI",
"LCID","LUMN","LUNR","LYFT","M","MARA","MAT","MBLY","MCHP","MGM","MNKD","MPW","MRNA","MRO","MSFT",
"MSTR","NCLH","NIO","NOK","NU","NVAX","NVDA","NVTS","O","OCGN","OPEN","ORCL","PBR","PFE","PLUG",
"PLTR","PSEC","PTON","QBTS","RBLX","RIG","RIVN","RKLB","RUN","S","SIRI","SOFI","SOUN","SPCE","T",
"TEVA","TFC","TGT","TMC","TNA","TSLA","TU","UAL","UBER","U","VALE","VFC","VIPS","VZ","WBD","WFC",
"WULF","X","XOM","YMM"
]
UNIVERSE = list(dict.fromkeys(UNIVERSE))[:max_symbols]

def safe_num(x):
    try:
        if x is None or pd.isna(x): return None
        return float(x)
    except Exception: return None

def get_statement_value(df, candidates, col=0):
    if df is None or df.empty: return None
    for name in candidates:
        if name in df.index:
            try: return safe_num(df.loc[name].iloc[col])
            except Exception: pass
    return None

def analyze(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y", interval="1d", auto_adjust=True)
        if hist is None or len(hist) < 60: return None
        info = t.get_info()
        price = safe_num(hist["Close"].iloc[-1])
        if price is None or price <= 0 or price > max_price: return None
        close = hist["Close"]
        vol = hist["Volume"]
        ma20 = safe_num(close.tail(20).mean())
        ma50 = safe_num(close.tail(50).mean())
        ma200 = safe_num(close.tail(200).mean()) if len(close)>=200 else None
        avgvol = safe_num(vol.tail(20).mean())
        dollarvol = (avgvol or 0) * price
        ret20 = safe_num((close.iloc[-1]/close.iloc[-21]-1)*100) if len(close)>21 else None
        ret60 = safe_num((close.iloc[-1]/close.iloc[-61]-1)*100) if len(close)>61 else None
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] and loss.iloc[-1] > 0 else np.nan
        rsi = safe_num(100-(100/(1+rs))) if pd.notna(rs) else None

        fin = t.financials
        bal = t.balance_sheet
        cf = t.cashflow
        revenue = get_statement_value(fin, ["Total Revenue"])
        gross = get_statement_value(fin, ["Gross Profit"])
        operating = get_statement_value(fin, ["Operating Income"])
        net = get_statement_value(fin, ["Net Income"])
        eps = safe_num(info.get("trailingEps"))
        pe = safe_num(info.get("trailingPE"))
        book = safe_num(info.get("bookValue"))
        pb = safe_num(info.get("priceToBook"))
        margin = safe_num(info.get("grossMargins"))
        current_ratio = safe_num(info.get("currentRatio"))
        beta = safe_num(info.get("beta"))
        op_cash = get_statement_value(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        debt = safe_num(info.get("totalDebt"))
        equity = safe_num(info.get("totalStockholderEquity"))
        de = debt/equity if debt is not None and equity and equity > 0 else None

        # Revenue and net income trend based on up to 3 reported fiscal years
        rev_growth = None
        ni_growth = None
        try:
            revs = fin.loc["Total Revenue"].dropna().iloc[:3].tolist()
            if len(revs)>=2 and revs[1] != 0: rev_growth=(revs[0]/revs[1]-1)*100
        except: pass
        try:
            nis = fin.loc["Net Income"].dropna().iloc[:3].tolist()
            if len(nis)>=2 and nis[1] != 0: ni_growth=(nis[0]/nis[1]-1)*100
        except: pass

        # transparent score: fundamental 60, technical 40. Missing values are not treated as passes.
        fund = 0; fcount=0
        checks = [
            (revenue is not None and revenue>0, 1),
            (gross is not None and gross>0, 1),
            (operating is not None and operating>0, 1),
            (net is not None and net>0, 2),
            (eps is not None and eps>0, 2),
            (op_cash is not None and op_cash>0, 2),
            (rev_growth is not None and rev_growth>0, 1),
            (ni_growth is not None and ni_growth>0, 1),
            (current_ratio is not None and current_ratio>=1.2, 1),
            (de is not None and de<1.5, 1),
            (margin is not None and margin>0.2, 1),
            (pe is not None and 0<pe<30, 1),
        ]
        for passed, weight in checks:
            fcount += weight
            if passed: fund += weight
        fscore = 60*fund/fcount if fcount else 0

        tech_checks = [
            (ma20 is not None and price>ma20, 1),
            (ma50 is not None and price>ma50, 1),
            (ma200 is not None and price>ma200, 1),
            (ret20 is not None and ret20>0, 1),
            (ret60 is not None and ret60>0, 1),
            (dollarvol>=1_000_000, 1),
            (rsi is not None and 45<=rsi<=70, 1),
        ]
        tscore=40*sum(w for ok,w in tech_checks if ok)/sum(w for _,w in tech_checks)
        long_score=round(fscore+tscore,1)
        short_score=round(0.45*fscore + 0.55*tscore,1)
        fair_value = None
        # Simple screening estimate only; not a valuation model.
        if eps and eps>0 and pe and pe>0:
            fair_value = eps * min(max(pe,8),25)
        return {
            "Ticker":ticker,"Company":info.get("shortName",ticker),"Price":round(price,2),
            "Long score":long_score,"Short score":short_score,"Revenue":revenue,
            "Gross profit":gross,"Operating income":operating,"Net income":net,"EPS":eps,
            "P/E":pe,"Book value":book,"P/B":pb,"Fair value (rough)":fair_value,
            "Gross margin":margin,"Current ratio":current_ratio,"Beta":beta,
            "Operating cash flow":op_cash,"Debt/Equity":de,"Revenue growth %":rev_growth,
            "Net income growth %":ni_growth,"Avg daily $ volume":dollarvol,
            "20d return %":ret20,"60d return %":ret60,"RSI":rsi,
            "MA20":ma20,"MA50":ma50,"MA200":ma200
        }
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def scan(symbols, price_limit):
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures={ex.submit(analyze, s):s for s in symbols}
        for f in as_completed(futures):
            r=f.result()
            if r: rows.append(r)
    return pd.DataFrame(rows)

if refresh:
    st.cache_data.clear()

with st.spinner("جاري فحص الأسهم وجلب أحدث البيانات المتاحة..."):
    data=scan(UNIVERSE, max_price)

if data.empty:
    st.error("لم تصل بيانات كافية في هذه المحاولة. جرّب التحديث لاحقًا أو قلّل عدد الأسهم.")
    st.stop()

data["Affordable shares"]=(capital/data["Price"]).apply(lambda x: max(0,int(x)) if pd.notna(x) else 0)
data["Entry zone"]=(data["Price"]*0.985).round(2).astype(str)+" – "+(data["Price"]*1.01).round(2).astype(str)
data["Stop loss (reference)"]=(data["Price"]*0.92).round(2)
data["Target 1 (reference)"]=(data["Price"]*1.10).round(2)
data["Target 2 (reference)"]=(data["Price"]*1.20).round(2)

a,b,c=st.columns(3)
a.metric("الشركات التي اجتازت فحص السعر",len(data))
b.metric("رأس المال",f"${capital:,.0f}")
c.metric("آخر تحديث",datetime.now().strftime("%Y-%m-%d %H:%M"))

tab1,tab2,tab3,tab4=st.tabs(["🏆 أفضل 3 للاستثمار","⚡ أفضل 3 للصفقات القصيرة","🔎 كل النتائج","📘 منهجية التحليل"])

with tab1:
    top=data.sort_values("Long score",ascending=False).head(3)
    st.subheader("أفضل المرشحين للاستثمار طويل الأجل")
    st.caption("الترتيب حسب نموذج نقاط مبدئي؛ راجع التقارير والبيانات قبل اتخاذ القرار.")
    for _,r in top.iterrows():
        with st.container(border=True):
            x,y,z=st.columns([2,1,1])
            x.subheader(f"{r['Ticker']} — {r['Company']}")
            x.write(f"السعر: **${r['Price']:.2f}** | منطقة متابعة: **{r['Entry zone']}**")
            y.metric("التقييم",f"{r['Long score']}/100")
            z.metric("عدد الأسهم الصحيحة نظريًا",r["Affordable shares"])
            st.write(f"وقف مرجعي: ${r['Stop loss (reference)']:.2f} | هدف 1: ${r['Target 1 (reference)']:.2f} | هدف 2: ${r['Target 2 (reference)']:.2f}")
            st.caption("المستويات آلية أولية وليست مستويات دعم/مقاومة مؤكدة. قد لا تسمح ميزانية أقل من 100$ بشراء سهم كامل؛ تحقق من دعم الوسيط للأسهم الكسرية.")
with tab2:
    top=data.sort_values("Short score",ascending=False).head(3)
    st.subheader("أفضل المرشحين للصفقات القصيرة")
    for _,r in top.iterrows():
        st.write(f"**{r['Ticker']} — {r['Company']}** | ${r['Price']:.2f} | التقييم {r['Short score']}/100 | عائد 20 يوم: {r['20d return %'] if pd.notna(r['20d return %']) else 'غير متاح'}%")
    st.caption("هذا ترتيب زخم وفحص فني أولي، وليس إشارة تنفيذ لحظية. لا توجد بيانات دفتر أوامر أو أسعار لحظية مضمونة.")
with tab3:
    cols=["Ticker","Company","Price","Long score","Short score","P/E","P/B","EPS","Revenue growth %","Net income growth %","Gross margin","Current ratio","Beta","Operating cash flow","Debt/Equity","Avg daily $ volume","RSI"]
    st.dataframe(data[cols].sort_values("Long score",ascending=False),use_container_width=True,hide_index=True)
    st.download_button("تحميل النتائج CSV",data.to_csv(index=False).encode("utf-8-sig"),"stock_hunter_scan.csv","text/csv")
with tab4:
    st.markdown("""
### كيف يحسب البرنامج التقييم؟
- **المالي (60 نقطة):** الإيرادات، الربح الإجمالي، دخل التشغيل، صافي الدخل، EPS، التدفق النقدي التشغيلي، نمو الإيرادات والأرباح، السيولة، الديون، هامش الربح، ومضاعف الربحية.
- **الفني (40 نقطة):** السعر مقابل متوسطات 20/50/200 يوم، عائد 20 و60 يوم، السيولة بالدولار، وRSI.
- **Fair Value التقريبي:** لا يُستخدم في الترتيب حاليًا؛ معروض فقط كتقدير توضيحي مبسط عند توفر EPS ومضاعف الربحية. لا يصلح لاتخاذ قرار منفرد.
- **القيمة الدفترية وP/B وBeta** تظهر للمقارنة ولا تُعامل كقواعد قبول مطلقة.

### حدود مهمة
1. Universe الحالية قائمة أولية وليست جميع الأسهم الأمريكية. وسّعها أو اربط مزود بيانات مدفوع قبل الاعتماد التشغيلي.
2. بيانات Yahoo Finance قد تتأخر أو تكون ناقصة، كما أن بعض القوائم المالية تختلف في مسمياتها.
3. لا يوجد حتى الآن تحقق شرعي آلي موثوق. لا تعتبر أي سهم متوافقًا شرعيًا دون مراجعة نشاطه ونسب الديون/النقد/الإيرادات وفق معيار تختاره.
4. لا يوجد تنبيه تلقائي أو استضافة دائمة في هذه النسخة. التحديث يتم عند فتح الموقع/الضغط على تحديث، مع تخزين مؤقت لمدة ساعة.
5. الأسهم تحت 100 دولار قد لا تناسب شراء سهم كامل؛ استخدم أسهمًا كسرية فقط إذا كان وسيطك يدعمها.
""")
st.caption("Stock Hunter prototype • Not investment advice")
