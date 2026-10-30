#!/usr/bin/env bash
# One time setup for PNG export. Safe to rerun.
set -u
cd "$(dirname "$0")"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is not installed. Install Node 18 or newer (for example: brew install node), then rerun this script."
  exit 1
fi

echo "Installing Node packages into $(pwd)/node_modules ..."
npm install --no-audit --no-fund --loglevel=error || { echo "npm install failed."; exit 1; }

echo "Installing Playwright Chromium (optional if Google Chrome is installed) ..."
if npx playwright install chromium >/dev/null 2>&1; then
  echo "Chromium installed."
else
  echo "Could not download Chromium. The exporter will fall back to an installed Google Chrome or Microsoft Edge."
  echo "You can also point it at any Chrome binary with CHROME_PATH=/path/to/chrome."
fi
echo "Setup complete."
