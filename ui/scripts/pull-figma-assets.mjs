import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import https from "node:https";
import http from "node:http";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function fetchToFile(url, outPath) {
    return new Promise((resolve, reject) => {
        const client = url.startsWith("https") ? https : http;
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        const file = fs.createWriteStream(outPath);
        client.get(url, (res) => {
            if (res.statusCode && res.statusCode >= 400) {
                file.close(); fs.unlinkSync(outPath);
                return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
            }
            res.pipe(file);
            file.on("finish", () => file.close(resolve));
        }).on("error", (err) => {
            try { file.close(); fs.unlinkSync(outPath); } catch { }
            reject(err);
        });
    });
}

async function main() {
    const [, , manifestPath = "../assets.json", outDir = "../public/figma"] = process.argv;
    const manifestAbs = path.resolve(__dirname, manifestPath);
    const outAbs = path.resolve(__dirname, outDir);

    const manifest = JSON.parse(fs.readFileSync(manifestAbs, "utf-8"));
    // Expecting something like: { "assets": [ { "name": "hero.png", "url": "https://..." }, ... ] }
    const list = manifest.assets || manifest.images || manifest; // be lenient about shape

    for (const item of list) {
        const url = item.url || item.src || item.href;
        if (!url) continue;
        const name = item.name || path.basename(new URL(url).pathname);
        const outPath = path.join(outAbs, name);
        console.log("→", name);
        await fetchToFile(url, outPath);
    }
    console.log("Done.");
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});

