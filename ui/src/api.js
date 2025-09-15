export async function build(ticksGlob = "data/ticks/*.csv") {
    const r = await fetch('/api/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticks_glob: ticksGlob }),
    });
    return r.json();
}

export async function train() {
    const r = await fetch('/api/train', { method: 'POST' });
    return r.json();
}

export async function predict(objective = 'acc') {
    const r = await fetch(`/api/predict?objective=${objective}`);
    return r.json();
}

export async function holdoutLastDays(days = 90) {
    const r = await fetch('/api/holdout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ last_days: days }),
    });
    return r.json();
}

export async function mt5SaveConfig(cfg) {
    const r = await fetch('/api/mt5/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cfg),
    });
    return r.json();
}

export async function mt5FetchTicks() {
    const r = await fetch('/api/mt5/fetch_ticks', { method: 'POST' });
    return r.json();
}

// --- Polygon backfill (full or universe/all) -------------------------------
export async function polygonBackfill(body = {}) {
    // body can include: { symbol, symbols, universe, all, years, timespan, multiplier, rpm }
    const r = await fetch('/api/polygon/backfill', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    return r.json();
}

// --- Polygon incremental backfill -----------------------------------------
// Looks at existing CSVs on the server, finds last timestamp per symbol,
// pulls from there forward, and de-dups on write.
export async function polygonBackfillIncremental(body = {}) {
    // same fields as polygonBackfill, plus { incremental: true }
    const r = await fetch('/api/polygon/backfill', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ incremental: true, ...body }),
    });
    return r.json();
}

// (your existing API functions)
export async function build(ticksGlob = "data/ticks/*.csv") {
    const r = await fetch('/api/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticks_glob: ticksGlob }),
    });
    return r.json();
}

export async function train() {
    const r = await fetch('/api/train', { method: 'POST' });
    return r.json();
}

export async function predict(objective = 'acc') {
    const r = await fetch(`/api/predict?objective=${objective}`);
    return r.json();
}

export async function holdoutLastDays(days = 90) {
    const r = await fetch('/api/holdout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ last_days: days }),
    });
    return r.json();
}

export async function mt5SaveConfig(cfg) {
    const r = await fetch('/api/mt5/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cfg),
    });
    return r.json();
}

export async function mt5FetchTicks() {
    const r = await fetch('/api/mt5/fetch_ticks', { method: 'POST' });
    return r.json();
}
