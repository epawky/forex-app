import { useState } from "react";

export default function FXDashboard() {
    const [ticker, setTicker] = useState("EURUSD");
    const [action, setAction] = useState("Build");
    const [objective, setObjective] = useState("acc"); // acc | mcc | pnl
    const [status, setStatus] = useState("");

    const callApi = async (path, body = {}) => {
        setStatus(`Running ${path}...`);
        try {
            const res = await fetch(`/api/${path}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ticker, objective, ...body }),
            });
            const data = await res.json().catch(() => ({}));
            setStatus(res.ok ? `✅ ${path} done` : `❌ ${path} failed`);
            if (!res.ok) console.error(data);
        } catch (e) {
            console.error(e);
            setStatus(`❌ ${path} error`);
        }
    };

    return (
        <div className="w-full max-w-md mx-auto p-4 font-sans">
            <header className="mb-6">
                <h1 className="text-2xl font-bold text-emerald-900">
                    Forex Prediction System
                </h1>
            </header>

            {/* Controls */}
            <div className="flex flex-col space-y-4">
                <div>
                    <label className="text-sm font-medium text-gray-700 mb-1 block">
                        Select Ticker
                    </label>
                    <select
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value)}
                        className="w-full py-2 pl-3 pr-10 border rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500 border-emerald-500"
                    >
                        <option value="EURUSD">EURUSD</option>
                        <option value="Add New Ticker">Add New Ticker</option>
                    </select>
                </div>

                <div>
                    <label className="text-sm font-medium text-gray-700 mb-1 block">
                        Action
                    </label>
                    <select
                        value={action}
                        onChange={(e) => setAction(e.target.value)}
                        className="w-full py-2 pl-3 pr-10 border rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500 border-emerald-500"
                    >
                        <option>Build</option>
                        <option>Train</option>
                        <option>Predict</option>
                    </select>
                </div>

                <div>
                    <label className="text-sm font-medium text-gray-700 mb-1 block">
                        Maximize Prediction
                    </label>
                    <select
                        value={objective}
                        onChange={(e) => setObjective(e.target.value)}
                        className="w-full py-2 pl-3 pr-10 border rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500 border-emerald-500"
                    >
                        <option value="acc">Accuracy</option>
                        <option value="mcc">MCC</option>
                        <option value="pnl">PnL Proxy</option>
                    </select>
                </div>

                <p className="text-sm text-emerald-700">
                    Selected: {ticker} – {action} ({objective})
                </p>
            </div>

            {/* MT5 Daily Data */}
            <div className="mt-6 p-3 border rounded-md bg-white">
                <div className="text-sm font-medium text-gray-700 mb-2">Pull Daily Data</div>
                <button
                    onClick={() => callApi("mt5/fetch")}
                    className="w-full px-4 py-2 text-sm font-medium text-white rounded-md focus:outline-none focus:ring-2"
                    style={{ backgroundColor: "rgb(1,106,58)" }}
                >
                    Fetch MT5 ticks (24h)
                </button>
            </div>

            {/* Docker buttons */}
            <div className="mt-6 flex flex-col items-center">
                <h2 className="text-lg font-bold mb-3">Load Docker</h2>
                <button
                    onClick={() => callApi("docker/build")}
                    className="my-1 w-60 bg-green-800 text-emerald-100 border border-black px-3 py-2 rounded font-bold"
                >
                    Load Docker for Daily Data (Build)
                </button>
                <button
                    onClick={() => callApi("docker/train")}
                    className="my-1 w-60 bg-green-800 text-emerald-100 border border-black px-3 py-2 rounded font-bold"
                >
                    Load Docker for Model (Train)
                </button>
                <button
                    onClick={() => callApi("docker/predict")}
                    className="my-1 w-60 bg-green-800 text-emerald-100 border border-black px-3 py-2 rounded font-bold"
                >
                    Load Docker for Predictions
                </button>
                <button
                    onClick={() => callApi("mt5/config")}
                    className="my-1 w-60 bg-green-800 text-emerald-100 border border-black px-3 py-2 rounded font-bold"
                >
                    Set MetaTrader5 Info
                </button>
            </div>

            {/* Run selected Action directly */}
            <div className="mt-6">
                <button
                    onClick={() =>
                        action === "Build"
                            ? callApi("build")
                            : action === "Train"
                                ? callApi("train")
                                : callApi("predict")
                    }
                    className="w-full bg-emerald-600 text-white px-4 py-2 rounded-md font-semibold"
                >
                    Run: {action}
                </button>
            </div>

            <div className="mt-4 text-sm text-gray-700 min-h-[1.5rem]">{status}</div>
        </div>
    );
}

