#!/usr/bin/env node
// Export an .excalidraw file to PNG with the scene embedded, so the PNG
// can be dragged back into Excalidraw and edited.
//
// Usage: node export_png.mjs <input.excalidraw> <output.png>
//
// Rendering uses Excalidraw's own export code inside a headless browser.
// Browser lookup order: CHROME_PATH env var, Playwright's Chromium,
// installed Google Chrome, installed Microsoft Edge.

import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { deflateSync } from "node:zlib";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

const [, , input, output] = process.argv;
if (!input || !output) {
  console.error("Usage: node export_png.mjs <input.excalidraw> <output.png>");
  process.exit(1);
}

let chromium;
try {
  ({ chromium } = require("playwright"));
} catch {
  console.error("Missing dependencies. Run: bash " + join(here, "setup.sh"));
  process.exit(2);
}

async function launch() {
  const attempts = [];
  if (process.env.CHROME_PATH) attempts.push({ executablePath: process.env.CHROME_PATH });
  attempts.push({}, { channel: "chrome" }, { channel: "msedge" });
  const errors = [];
  for (const opts of attempts) {
    try {
      return await chromium.launch({ headless: true, ...opts });
    } catch (e) {
      errors.push(`${JSON.stringify(opts)}: ${String(e.message).split("\n")[0]}`);
    }
  }
  console.error("Could not start a browser. Tried:\n  " + errors.join("\n  "));
  console.error("Fix: run setup.sh, install Google Chrome, or set CHROME_PATH.");
  process.exit(3);
}

// Embed the scene in a PNG tEXt chunk using Excalidraw's own format
// (keyword "application/vnd.excalidraw+json", deflated JSON as a byte string),
// so the PNG opens as an editable drawing in Excalidraw.
function crc32(buf) {
  let c, crc = 0xffffffff;
  for (let n = 0; n < buf.length; n++) {
    c = (crc ^ buf[n]) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crc = (crc >>> 8) ^ c;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function embedScene(png, scene) {
  const json = JSON.stringify({
    type: "excalidraw", version: 2, source: "ml-system-summary skill",
    elements: scene.elements, appState: scene.appState || {}, files: scene.files || {},
  });
  const encoded = deflateSync(Buffer.from(json, "utf8")).toString("latin1");
  const payload = JSON.stringify({ version: "1", encoding: "bstring", compressed: true, encoded });
  const data = Buffer.concat([
    Buffer.from("application/vnd.excalidraw+json", "latin1"), Buffer.from([0]),
    Buffer.from(payload, "latin1"),
  ]);
  const type = Buffer.from("tEXt", "latin1");
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([type, data])));
  const chunk = Buffer.concat([len, type, data, crc]);
  const iend = png.lastIndexOf(Buffer.from("IEND", "latin1")) - 4;
  return Buffer.concat([png.subarray(0, iend), chunk, png.subarray(iend)]);
}

const scene = JSON.parse(readFileSync(input, "utf8"));
const utilsPath = join(here, "node_modules", "@excalidraw", "utils", "dist", "excalidraw-utils.min.js");

const browser = await launch();
try {
  const page = await browser.newPage();
  await page.setContent("<!doctype html><html><body></body></html>");
  await page.addScriptTag({ path: utilsPath });
  const b64 = await page.evaluate(async (scene) => {
    const blob = await window.ExcalidrawUtils.exportToBlob({
      elements: scene.elements,
      appState: {
        ...(scene.appState || {}),
        exportBackground: true,
        viewBackgroundColor: "#ffffff",
        exportEmbedScene: true,
      },
      files: scene.files || {},
      mimeType: "image/png",
      exportPadding: 40,
      getDimensions: (w, h) => ({ width: w * 2, height: h * 2, scale: 2 }),
    });
    const buf = new Uint8Array(await blob.arrayBuffer());
    let s = "";
    for (let i = 0; i < buf.length; i += 0x8000) {
      s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    }
    return btoa(s);
  }, scene);
  writeFileSync(output, embedScene(Buffer.from(b64, "base64"), scene));
  console.log(`Wrote ${output}`);
} finally {
  await browser.close();
}
