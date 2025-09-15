import { useEffect, useMemo, useRef, useState } from "react";

/** ---- Universes ---- */
const UNIVERSES = {
    all: [],
    majors: ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"],
    minors: [
        "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY", "EURCHF", "EURAUD", "EURNZD",
        "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD", "AUDNZD", "AUDCAD", "CADJPY", "CADCHF",
        "NZDJPY", "NZDCHF", "NZDCAD", "USDSGD", "USDSEK", "USDNOK", "USDTRY", "USDZAR",
        "USDPLN", "USDHUF", "USDILS", "USDHKD", "USDKRW", "USDCNH", "USDINR", "USDIDR",
        "USDTHB", "USDMXN",
    ],
    exotics: ["USDCLP", "USDARS", "USDBRL", "USDTRY", "USDRUB", "USDCZK", "USDDKK", "USDHUF", "USDPLN", "USDTWD"],
    metals: ["XAUUSD", "XAGUSD"],
};

function universeList(u) {
    if (u === "all") {
        const s = Object.entries(UNIVERSES)
            .filter(([k]) => k !== "all")
            .flatMap(([, a]) => a);
        return Array.from(new Set(s)).sort();
    }
    return UNIVERSES[u] || [];
}

/** ---- Helpers ---- */
function normalizePreds(p) {
    if (!p || typeof p !== "object") return null;
    return {
        daily: p.daily,
        w3_12: p.w3_12,
        "2_10": p["2_10"] ?? p["w2_10"],
    };
}
function fmtProb(v) { return typeof v === "number" ? v.toFixed(3) : "-"; }
function fmtThr(v) { return typeof v === "number" ? v.toFixed(3) : "-"; }

/** ---- Simple 2-pane canvas chart (price + proba overlays) ---- */
function drawChart(canvas, pricePts, seriesMap, domain) {
    const DPR = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 600;
    const h = canvas.clientHeight || 260;
    canvas.width = Math.floor(w * DPR);
    canvas.height = Math.floor(h * DPR);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // layout
    const pad = 8;
    const midGap = 6;
    const topH = Math.floor((h - pad * 2 - midGap) * 0.62);
    const botH = (h - pad * 2 - midGap) - topH;

    // x-domain
    let tMin, tMax;
    if (domain?.from && domain?.to) {
        tMin = new Date(domain.from).getTime();
        tMax = new Date(domain.to).getTime();
    } else {
        const allTimes = [];
        (pricePts || []).forEach(p => { if (p?.t) allTimes.push(new Date(p.t).getTime()); });
        ["daily", "w3_12", "2_10"].forEach(k => {
            (seriesMap?.[k] || []).forEach(p => { if (p?.t) allTimes.push(new Date(p.t).getTime()); });
        });
        if (allTimes.length < 2) {
            ctx.fillStyle = "#6b7280";
            ctx.font = "12px sans-serif";
            ctx.fillText("No chart data.", pad, pad + 14);
            return;
        }
        tMin = Math.min(...allTimes);
        tMax = Math.max(...allTimes);
    }

    const xAt = (tISO) => {
        const tt = new Date(tISO).getTime();
        return pad + (tt - tMin) * (w - pad * 2) / Math.max(1, (tMax - tMin));
    };
    const clipByDomain = (arr = []) => arr.filter(p => {
        const tt = new Date(p.t).getTime();
        return tt >= tMin && tt <= tMax;
    });

    const priceClipped = clipByDomain(pricePts || []);
    const dailyClipped = clipByDomain(seriesMap?.daily || []);
    const w312Clipped = clipByDomain(seriesMap?.w3_12 || []);
    const w210Clipped = clipByDomain(seriesMap?.["2_10"] || []);

    // Price scale
    const prices = priceClipped.map(p => p.mid ?? p.v).filter(v => Number.isFinite(v));
    const havePrice = prices.length >= 2;
    const pMin = havePrice ? Math.min(...prices) : 0;
    const pMax = havePrice ? Math.max(...prices) : 1;
    const topY = (v) => {
        if (!Number.isFinite(v) || pMax === pMin) return pad + topH / 2;
        return pad + (pMax - v) * (topH - 1) / Math.max(1e-9, (pMax - pMin));
    };

    // Price line
    if (havePrice) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#111827";
        ctx.beginPath();
        let started = false;
        priceClipped.forEach(p => {
            const v = p.mid ?? p.v;
            if (!Number.isFinite(v)) return;
            const x = xAt(p.t);
            const y = topY(v);
            if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }
    // Frame
    ctx.strokeStyle = "#e5e7eb";
    ctx.strokeRect(pad, pad, w - pad * 2, topH);

    // Bottom: probabilities 0..1
    const botTop = pad + topH + midGap;
    const probY = (p) => botTop + (1 - p) * (botH - 1);

    // gridlines
    ctx.strokeStyle = "#f3f4f6";
    ctx.lineWidth = 1;
    [0, 0.5, 1].forEach(v => {
        const y = probY(v);
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
    });
    ctx.fillStyle = "#6B7280";
    ctx.font = "10px sans-serif";
    ctx.fillText("1.0", pad + 2, probY(1) - 2);
    ctx.fillText("0.5", pad + 2, probY(0.5) - 2);
    ctx.fillText("0.0", pad + 2, probY(0) - 2);

    function strokeSeries(arr, color) {
        if (!arr || arr.length < 2) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        let started = false;
        arr.forEach(p => {
            const x = xAt(p.t);
            const y = probY(Math.max(0, Math.min(1, p.proba)));
            if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    // overlays
    strokeSeries(dailyClipped, "#2563eb"); // blue
    strokeSeries(w312Clipped, "#16a34a"); // green
    strokeSeries(w210Clipped, "#f59e0b"); // amber

    // frame
    ctx.strokeStyle = "#e5e7eb";
    ctx.strokeRect(pad, botTop, w - pad * 2, botH);
}

/** ---- Page ---- */
export default function FigmaPage() {
    const [universe, setUniverse] = useState("majors");
    const [ticker, setTicker] = useState("EURUSD");
    const [objective, setObjective] = useState("acc"); // acc | mcc | pnl
    const [hours, setHours] = useState(24);
    const [status, setStatus] = useState("");

    const [loadingPreds, setLoadingPreds] = useState(false);
    const [predictionsByTicker, setPredictionsByTicker] = useState({});

    const [pricePoints, setPricePoints] = useState([]);
    const [series, setSeries] = useState(null);
    const [xDomain, setXDomain] = useState(null); // { from, to }

    const tickers = useMemo(() => universeList(universe), [universe]);

    const inFlightPred = useRef(null);
    const predSeq = useRef(0);
    const inFlightChart = useRef(null);
    const chartSeq = useRef(0);

    const canvasRef = useRef(null);

    // pick first ticker on universe change
    useEffect(() => {
        const first = (universeList(universe)[0] || "EURUSD").toUpperCase();
        setTicker(first);
    }, [universe]);

    // ---- Predictions (snapshot for table) ----
    useEffect(() => {
        const sym = (ticker || "EURUSD").toUpperCase();
        const thisReq = ++predSeq.current;

        setPredictionsByTicker(prev => ({ ...prev, [sym]: null }));
        setLoadingPreds(true);

        if (inFlightPred.current) inFlightPred.current.abort();
        const ctl = new AbortController();
        inFlightPred.current = ctl;

        fetch(`/api/predict?objective=${objective}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symbol: sym }),
            signal: ctl.signal,
        })
            .then(r => r.json())
            .then(data => {
                if (thisReq !== predSeq.current) return;
                const key = (data.symbol || sym).toUpperCase();
                const preds = normalizePreds(data.predictions);
                setPredictionsByTicker(prev => ({ ...prev, [key]: preds }));
            })
            .catch(e => { if (e.name !== "AbortError") setStatus(`Predict error: ${String(e)}`); })
            .finally(() => { if (thisReq === predSeq.current) setLoadingPreds(false); });

        return () => ctl.abort();
    }, [ticker, objective]);

    // ---- Chart data (price + prediction series) ----
    useEffect(() => {
        const sym = (ticker || "EURUSD").toUpperCase();
        const thisReq = ++chartSeq.current;

        if (inFlightChart.current) inFlightChart.current.abort();
        const ctl = new AbortController();
        inFlightChart.current = ctl;

        const urlTicks = `/api/ticks/recent?symbol=${encodeURIComponent(sym)}&hours=${hours}&_=${Date.now()}`;
        const urlSeries = `/api/predict/series?symbol=${encodeURIComponent(sym)}&hours=${hours}&objective=${objective}&align_to_ticks=1&_=${Date.now()}`;

        Promise.all([
            fetch(urlTicks, { signal: ctl.signal })
                .then(r => r.json())
                .then(d => {
                    let pts = Array.isArray(d.points)
                        ? d.points.map(p => ({ t: p.t, mid: p.mid ?? p.v }))
                        : [];
                    if (d?.from && d?.to) {
                        setXDomain({ from: d.from, to: d.to });
                    } else if (pts.length > 1) {
                        const tms = pts.map(p => new Date(p.t).getTime());
                        const minT = new Date(Math.min(...tms)).toISOString();
                        const maxT = new Date(Math.max(...tms)).toISOString();
                        setXDomain({ from: minT, to: maxT });
                    } else {
                        setXDomain(null);
                    }
                    return pts;
                }),
            fetch(urlSeries, { signal: ctl.signal })
                .then(r => r.json())
                .then(d => {
                    const S = d?.series || {};
                    return {
                        daily: (S.daily || []).map(x => ({ t: x.t, proba: x.proba })),
                        w3_12: (S.w3_12 || []).map(x => ({ t: x.t, proba: x.proba })),
                        "2_10": (S["2_10"] || S["w2_10"] || []).map(x => ({ t: x.t, proba: x.proba })),
                    };
                }),
        ])
            .then(([price, ser]) => {
                if (thisReq !== chartSeq.current) return;
                setPricePoints(price);
                setSeries(ser);
            })
            .catch(e => {
                if (e.name !== "AbortError") setStatus(`Chart fetch error: ${String(e)}`);
            });

        return () => ctl.abort();
    }, [ticker, hours, objective]);

    // ---- Draw chart whenever data or domain changes ----
    useEffect(() => {
        const c = canvasRef.current;
        if (!c) return;
        drawChart(c, pricePoints, series, xDomain);
    }, [pricePoints, series, xDomain]);

    async function post(path, body = {}) {
        setStatus(`Running ${path}...`);
        try {
            const res = await fetch(`/api${path}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const data = await res.json().catch(() => ({}));
            setStatus(JSON.stringify(data, null, 2));
            return data;
        } catch (e) {
            setStatus(`❌ ${String(e)}`);
            throw e;
        }
    }

    // Buttons
    function setPolygonKey() {
        const prev = localStorage.getItem("POLY_KEY") || "";
        const k = prompt("Enter Polygon.io API Key:", prev) || "";
        const key = k.trim();
        if (!key) return;
        localStorage.setItem("POLY_KEY", key);
        // persist server-side (your app.py has /api/polygon/credentials)
        post("/polygon/credentials", { api_key: key }).catch(() => { });
        setStatus("Polygon API key saved (local + server).");
    }
    async function downloadDataAll() {
        const api_key = localStorage.getItem("POLY_KEY") || undefined;
        const body = { incremental: true, rpm: 5, timespan: "minute", multiplier: 1, api_key };
        if (universe === "all") body.all = true; else body.universe = universe;
        await post("/polygon/backfill", body);
    }
    async function fetchMt5Ticks() { await post("/mt5/fetch_ticks", { symbol: ticker }); }
    async function buildSelected() { await post("/build", { symbol: ticker }); }
    async function trainSelected() { await post("/train", { symbol: ticker }); }

    const preds = predictionsByTicker[(ticker || "").toUpperCase()] || null;

    return (
        <div className="w-full max-w-2xl mx-auto p-4 font-sans">
            <h1 className="text-2xl font-bold text-emerald-900 mb-4">Forex Prediction System</h1>

            {/* Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
                <div>
                    <label className="text-sm text-gray-700 block mb-1">Universe</label>
                    <select
                        value={universe}
                        onChange={(e) => setUniverse(e.target.value)}
                        className="w-full py-2 pl-3 pr-10 border rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                        <option value="all">All</option>
                        <option value="majors">Majors</option>
                        <option value="minors">Minors</option>
                        <option value="exotics">Exotics</option>
                        <option value="metals">Metals</option>
                    </select>
                </div>

                <div>
                    <label className="text-sm text-gray-700 block mb-1">Ticker</label>
                    <select
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                        className="w-full py-2 pl-3 pr-10 border rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                        {tickers.map((s) => (<option key={s} value={s}>{s}</option>))}
                    </select>
                </div>

                <div>
                    <label className="text-sm text-gray-700 block mb-1">Objective</label>
                    <select
                        value={objective}
                        onChange={(e) => setObjective(e.target.value)}
                        className="w-full py-2 pl-3 pr-10 border rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                        <option value="acc">Accuracy</option>
                        <option value="mcc">MCC</option>
                        <option value="pnl">PnL Proxy</option>
                    </select>
                </div>

                <div>
                    <label className="text-sm text-gray-700 block mb-1">Window</label>
                    <select
                        value={hours}
                        onChange={(e) => setHours(parseInt(e.target.value, 10))}
                        className="w-full py-2 pl-3 pr-10 border rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                        <option value={6}>6h</option>
                        <option value={12}>12h</option>
                        <option value={24}>24h</option>
                        <option value={36}>36h</option>
                        <option value={48}>48h</option>
                        <option value={72}>72h</option>
                    </select>
                </div>
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-1 gap-2 mb-4">
                <button onClick={downloadDataAll} className="w-full px-4 py-2 rounded-md bg-emerald-700 text-white hover:bg-emerald-800">
                    Download Data ({universe === "all" ? "All universes" : universe})
                </button>
                <button onClick={fetchMt5Ticks} className="w-full px-4 py-2 rounded-md bg-emerald-700 text-white hover:bg-emerald-800">
                    Fetch MT5 Ticks (24h) for {ticker}
                </button>
                <button onClick={setPolygonKey} className="w-full px-4 py-2 rounded-md bg-white border border-emerald-700 text-emerald-700 hover:bg-emerald-50">
                    Set Polygon.io API Key
                </button>
                <div className="grid grid-cols-2 gap-2">
                    <button onClick={buildSelected} className="px-4 py-2 rounded-md bg-emerald-600 text-white hover:bg-emerald-700">
                        Build ({ticker})
                    </button>
                    <button onClick={trainSelected} className="px-4 py-2 rounded-md bg-emerald-600 text-white hover:bg-emerald-700">
                        Train ({ticker})
                    </button>
                </div>
            </div>

            {/* Chart */}
            <section className="mb-6">
                <div className="flex items-center justify-between mb-2">
                    <h2 className="text-sm font-medium text-gray-700">
                        {ticker} — last {hours}h (price + prediction overlays)
                    </h2>
                </div>
                <div className="h-64 border rounded-md p-2 bg-white">
                    <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
                </div>
            </section>

            {/* Predictions Table */}
            <PredictionsTable preds={preds} loading={loadingPreds} ticker={ticker} />

            {/* Status / debug */}
            <pre className="mt-4 text-xs whitespace-pre-wrap border rounded p-3 bg-gray-50">
                {status || "Ready."}
            </pre>
        </div>
    );
}

/** ---- Table ---- */
function PredictionsTable({ preds, loading, ticker }) {
    if (loading) {
        return <div className="mt-2 text-sm text-emerald-700">Loading predictions for {ticker}…</div>;
    }
    if (!preds) {
        return <div className="mt-2 text-sm text-gray-600">No predictions yet for {ticker}.</div>;
    }
    const rows = [
        { label: "Daily", key: "daily" },
        { label: "3–12 Window", key: "w3_12" },
        { label: "2–10 Window", key: "2_10" },
    ];
    return (
        <div className="overflow-x-auto">
            <table className="w-full text-sm border rounded-md">
                <thead className="bg-gray-100">
                    <tr>
                        <th className="text-left p-2">Window</th>
                        <th className="text-left p-2">Dir</th>
                        <th className="text-left p-2">Prob (eff/raw)</th>
                        <th className="text-left p-2">Thr</th>
                        <th className="text-left p-2">Meets Thr?</th>
                        <th className="text-left p-2">Flip</th>
                        <th className="text-left p-2">Pips Pred</th>
                        <th className="text-left p-2">Active</th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map(({ label, key }) => {
                        const r = preds?.[key];
                        const meets =
                            r && typeof r.proba_eff === "number" && typeof r.threshold === "number"
                                ? r.proba_eff >= r.threshold
                                : undefined;
                        return (
                            <tr key={key} className="border-t">
                                <td className="p-2">{label}</td>
                                <td className="p-2">{r?.dir ?? "-"}</td>
                                <td className="p-2">{fmtProb(r?.proba_eff)} / {fmtProb(r?.proba_raw)}</td>
                                <td className="p-2">{fmtThr(r?.threshold)}</td>
                                <td className="p-2">{meets === undefined ? "-" : meets ? "Yes" : "No"}</td>
                                <td className="p-2">{r?.flip ? "Yes" : "No"}</td>
                                <td className="p-2">{typeof r?.pips_pred === "number" ? r.pips_pred.toFixed(2) : "-"}</td>
                                <td className="p-2">{r?.active ? "Yes" : "No"}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}
