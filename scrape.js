// scrape.js
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

(async () => {
    const url = "https://wrap-patch-65350245.figma.site/";
    const outDir = path.resolve(__dirname, "scrape_out");
    const htmlPath = path.join(outDir, "page.html");

    const browser = await puppeteer.launch({
        headless: "new", // Chrome >=115
        defaultViewport: { width: 1440, height: 900 }
    });
    const page = await browser.newPage();

    await page.goto(url, { waitUntil: "networkidle0", timeout: 120000 });

    // Inline computed HTML (DOM) so you can copy sections into JSX
    const html = await page.evaluate(() => document.documentElement.outerHTML);

    // Make sure output dir exists
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(htmlPath, html, "utf8");

    console.log(`Saved DOM to ${htmlPath}`);

    // Also list external stylesheet and image URLs so you can download them
    const assets = await page.evaluate(() => {
        const css = [...document.querySelectorAll('link[rel="stylesheet"]')].map(l => l.href);
        const imgs = [...document.images].map(i => i.src);
        const fonts = [...document.querySelectorAll('link[rel="preload"][as="font"], style')]
            .map(n => n.href).filter(Boolean);
        return { css, imgs, fonts };
    });

    fs.writeFileSync(path.join(outDir, "assets.json"), JSON.stringify(assets, null, 2));
    console.log("Saved asset list to scrape_out/assets.json");

    await browser.close();
})();

