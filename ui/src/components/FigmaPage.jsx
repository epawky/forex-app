import { useMemo, useRef, useState } from 'react';

export default function FigmaPage() {
    const [tickers, setTickers] = useState(['EURUSD']);
    const [ticker, setTicker] = useState('EURUSD');
    const [action, setAction] = useState('Build');     // Build | Train | Predict
    const [objective, setObjective] = useState('acc'); // acc | mcc | pnl
    const [status, setStatus] = useState('');
    const [lastUpdated, setLastUpdated] = useState(() =>
        new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
    );

    // hidden file input for “Add New Ticker” → upload CSV
    const fileInputRef = useRef(null);
    const [pendingSymbol, setPendingSymbol] = useState(null); // symbol to pair with the upload

    // NEW: typed symbol input state
    const [symbolInput, setSymbolInput] = useState(ticker);

    const ACTIONS = ['Build', 'Train', 'Predict'];

    const actionText = useMemo(
        () => `${ticker} - ${action}`,
        [ticker, action]
    );

    function resetFileInput() {
        if (fileInputRef.current) {
            // Ensure change fires even for same file chosen again
            fileInputRef.current.value = '';
        }
    }

    async function uploadCsv(file, sym) {
        if (!file || !sym) return;

        try {
            setStatus(`Uploading ${file.name} for ${sym}…`);
            const fd = new FormData();
            fd.append('symbol', sym);
            fd.append('file', file);

            const res = await fetch('/api/upload_ticks', {
                method: 'POST',
                body: fd,
            });
            const data = await res.json().catch(() => ({}));
            setStatus(JSON.stringify(data, null, 2));

            if (data.ok) {
                setTickers(prev => (prev.includes(sym) ? prev : [...prev, sym]));
                setTicker(sym);
                setLastUpdated(new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }));
            }
        } catch (e) {
            setStatus(String(e));
        } finally {
            setPendingSymbol(null);
            resetFileInput();
        }
    }

    function onFileChosen(e) {
        const file = e.target.files?.[0];
        if (!file) {
            // user canceled; revert state
            setPendingSymbol(null);
            return;
        }
        uploadCsv(file, pendingSymbol);
    }

    function onTickerChange(e) {
        const val = e.target.value;
        if (val === '__add__') {
            // Ask for the symbol first
            const name = prompt('Add ticker symbol (e.g., GBPUSD):');
            const sym = (name || '').trim().toUpperCase();
            if (!sym) {
                // no symbol, reset select to current ticker
                e.target.value = ticker;
                return;
            }
            setPendingSymbol(sym);
            // open file chooser
            resetFileInput();
            fileInputRef.current?.click();
            // keep UI selection on current ticker until upload finishes
            e.target.value = ticker;
            return;
        }
        setTicker(val);
    }

    async function post(path, body) {
        setStatus(`Running ${path}…`);
        try {
            const res = await fetch(`/api${path}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: body ? JSON.stringify(body) : undefined,
            });
            const data = await res.json().catch(() => ({}));
            setStatus(JSON.stringify(data, null, 2));
            setLastUpdated(new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }));
        } catch (e) {
            setStatus(String(e));
        }
    }

    // Clicking the “Selected:” line commits the current selection
    async function commitSelection() {
        if (action === 'Build') {
            await post('/build', { glob: 'data/ticks/*.csv', symbol: ticker });
        } else if (action === 'Train') {
            await post('/train', { symbol: ticker });
        } else if (action === 'Predict') {
            await post(`/predict?objective=${objective}`, { symbol: ticker });
        }
    }

    // --- MT5 config prompt + call ---
    async function openMt5ConfigDialog() {
        const login = prompt('MT5 Login (account number):', '') || '';
        const password = prompt('MT5 Password:', '') || '';
        const server = prompt('MT5 Server (e.g., MetaQuotes-Demo):', '') || '';
        const path = prompt('MT5 Terminal Path (optional, leave blank to auto-detect):', '') || '';
        const daysStr = prompt('How many days of ticks to fetch when requested? (default 1):', '1') || '1';
        const days = Math.max(1, parseInt(daysStr, 10) || 1);

        await post('/mt5/config', { login, password, server, path, symbol: ticker, days });
    }

    return (
        <div className="min-h-screen bg-white text-gray-900">
            {/* hidden file input for Add New Ticker */}
            <input
                ref={fileInputRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={onFileChosen}
            />

            <div className="max-w-4xl mx-auto p-6">
                {/* Header */}
                <header className="mb-8">
                    <h1 className="text-3xl font-bold text-emerald-900 mb-3">Forex Prediction System</h1>
                    <p className="text-gray-700 leading-relaxed">
                        This app lets you ingest daily ticks, rebuild the dataset, retrain models, and generate next-day
                        predictions—without touching code. Use the controls below to fetch data from MetaTrader 5, build,
                        train, and predict for your chosen ticker.
                    </p>
                </header>

                {/* Controls */}
                <section className="mb-6">
                    <div className="grid sm:grid-cols-3 gap-4">
                        {/* REPLACED: Dropdown -> Text Input + Apply */}
                        <div className="flex flex-col">
                            <label className="text-sm font-medium text-gray-700 mb-1">Symbol</label>
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    inputMode="text"
                                    autoComplete="off"
                                    spellCheck="false"
                                    value={symbolInput}
                                    onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter') {
                                            if (/^[A-Z0-9._-]{3,15}$/.test(symbolInput)) {
                                                setTicker(symbolInput);
                                            }
                                        }
                                    }}
                                    placeholder="e.g., EURUSD"
                                    className="w-full py-2 px-3 border rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                                />
                                <button
                                    type="button"
                                    onClick={() => {
                                        if (/^[A-Z0-9._-]{3,15}$/.test(symbolInput)) {
                                            setTicker(symbolInput);
                                        }
                                    }}
                                    className="px-3 py-2 rounded-md bg-emerald-700 text-white hover:bg-emerald-800 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                                >
                                    Apply
                                </button>
                            </div>
                            <p className="mt-1 text-xs text-gray-500">
                                We’ll use this symbol for MT5 fetch, build, train, and predict.
                            </p>
                        </div>

                        <div className="flex flex-col">
                            <label className="text-sm font-medium text-gray-700 mb-1">Action</label>
                            <div className="relative">
                                <select
                                    className="appearance-none w-full py-2 pl-3 pr-10 border rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                                    value={action}
                                    onChange={(e) => setAction(e.target.value)}
                                >
                                    {ACTIONS.map(a => <option key={a}>{a}</option>)}
                                </select>
                                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2">
                                    <svg className="h-5 w-5 text-emerald-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                                        <path fillRule="evenodd" d="M5.29 7.29a1 1 0 011.42 0L10 10.59l3.29-3.3a1 1 0 111.42 1.42l-4 4a1 1 0 01-1.42 0l-4-4a1 1 0 010-1.42z" clipRule="evenodd" />
                                    </svg>
                                </div>
                            </div>
                        </div>

                        <div className="flex flex-col">
                            <label className="text-sm font-medium text-gray-700 mb-1">Maximize Prediction</label>
                            <div className="relative">
                                <select
                                    className={`appearance-none w-full py-2 pl-3 pr-10 border rounded-md focus:outline-none ${action === 'Predict'
                                        ? 'bg-white focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500'
                                        : 'bg-gray-100 text-gray-500 cursor-not-allowed'
                                        }`}
                                    value={objective}
                                    onChange={(e) => setObjective(e.target.value)}
                                    disabled={action !== 'Predict'}
                                >
                                    <option value="acc">Accuracy</option>
                                    <option value="mcc">MCC</option>
                                    <option value="pnl">PnL Proxy</option>
                                </select>
                                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2">
                                    <svg className="h-5 w-5 text-emerald-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                                        <path fillRule="evenodd" d="M5.29 7.29a1 1 0 011.42 0L10 10.59l3.29-3.3a1 1 0 111.42 1.42l-4 4a1 1 0 01-1.42 0l-4-4a1 1 0 010-1.42z" clipRule="evenodd" />
                                    </svg>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Clickable commit */}
                    <p className="mt-3 text-sm">
                        <span className="text-emerald-700">Selected: </span>
                        <span
                            className="cursor-pointer underline underline-offset-2 decoration-emerald-600 hover:text-emerald-800 select-text"
                            onClick={commitSelection}
                            title="Click to run with current selections"
                        >
                            {ticker} - {action}{action === 'Predict' ? ` [${objective}]` : ''}
                        </span>
                    </p>
                </section>

                {/* Actions */}
                <section className="mb-8">
                    <div className="grid sm:grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <h2 className="text-sm font-medium text-gray-700">Pull Daily Data</h2>
                            <button
                                onClick={() => post('/mt5/fetch_ticks', { symbol: ticker })}
                                className="w-full px-4 py-2 rounded-md bg-emerald-700 text-white shadow hover:bg-emerald-800 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                            >
                                Fetch MT5 Ticks
                            </button>
                            <p className="text-xs text-gray-600">
                                Downloads the last 24h of {ticker} ticks from your local MetaTrader 5 and saves to <code>data/ticks/</code>.
                            </p>
                        </div>

                        <div className="space-y-2">
                            <h2 className="text-xl font-bold text-center">Load Docker</h2>
                            <div className="grid grid-cols-1 gap-2">
                                <button
                                    onClick={() => post('/build', { glob: 'data/ticks/*.csv' })}
                                    className="px-4 py-2 rounded bg-emerald-700 text-white hover:bg-emerald-800"
                                >
                                    Load Docker for Daily Data (Build)
                                </button>
                                <button
                                    onClick={() => post('/train')}
                                    className="px-4 py-2 rounded bg-emerald-700 text-white hover:bg-emerald-800"
                                >
                                    Load Docker for Model (Train)
                                </button>
                                <button
                                    onClick={() => post(`/predict?objective=${objective}`)}
                                    className="px-4 py-2 rounded bg-emerald-700 text-white hover:bg-emerald-800 disabled:opacity-60"
                                    disabled={action !== 'Predict'}
                                >
                                    Load Docker for Predictions
                                </button>
                                <button
                                    onClick={openMt5ConfigDialog}
                                    className="px-4 py-2 rounded bg-emerald-700 text-white hover:bg-emerald-800"
                                >
                                    Set MetaTrader5 Info
                                </button>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Output */}
                <section className="mb-8">
                    <pre className="text-xs whitespace-pre-wrap border rounded p-3 bg-gray-50">{status || 'Ready.'}</pre>
                </section>

                {/* Explainers (unchanged) */}
                <section className="mb-8 space-y-6">
                    <div>
                        <h2 className="text-2xl font-bold mb-2">Quick start</h2>
                        <ol className="list-decimal list-inside space-y-2 text-gray-700">
                            <li><span className="font-semibold">Fetch ticks (optional):</span> Pull latest 24h ticks from MT5.</li>
                            <li><span className="font-semibold">Build dataset:</span> Convert ticks → features &amp; labels.</li>
                            <li><span className="font-semibold">Train models:</span> Fit models for Daily, 2–10, and 3–12 windows.</li>
                            <li><span className="font-semibold">Predict:</span> Choose an objective (Accuracy/MCC/PnL) and run predictions.</li>
                        </ol>
                    </div>

                    <div>
                        <h2 className="text-2xl font-bold mb-2">Prediction objectives</h2>
                        <ul className="list-disc list-inside space-y-2 text-gray-700">
                            <li><span className="font-semibold">Accuracy (acc):</span> Maximize % correct direction.</li>
                            <li><span className="font-semibold">MCC (mcc):</span> Balanced correctness even when classes are skewed.</li>
                            <li><span className="font-semibold">PnL (pnl):</span> Simple PnL proxy to choose the decision threshold.</li>
                        </ul>
                    </div>

                    <div>
                        <h2 className="text-2xl font-bold mb-2">Troubleshooting</h2>
                        <ol className="list-decimal list-inside space-y-2 text-gray-700">
                            <li>If Predict says columns/models missing, run Build then Train first.</li>
                            <li>MT5 fetch runs on the host. Make sure MT5 is installed and logged in; set env vars if needed.</li>
                            <li>If a Docker job hangs, restart Docker Desktop and try again.</li>
                        </ol>
                    </div>
                </section>

                <footer className="pt-4 text-sm text-gray-600">
                    <p>Last updated: {lastUpdated}</p>
                </footer>
            </div>
        </div>
    );
}
