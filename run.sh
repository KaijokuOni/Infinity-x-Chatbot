#!/usr/bin/env bash
# Launch the Bangla S2S chatbot and expose it over the internet via ngrok.
#
#   ./run.sh           # server + public ngrok URL
#   ./run.sh --local   # server only (http://127.0.0.1:8000)
#
# Requires: uv, ffmpeg, ngrok (authenticated). Needs GROQ_API_KEY +
# GEMINI_API_KEY in .env. TTS (edge-tts) needs no key.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
LOCAL_ONLY=0
[[ "${1:-}" == "--local" ]] && LOCAL_ONLY=1

cleanup() {
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  [[ -n "${NGROK_PID:-}" ]] && kill "$NGROK_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting server on port $PORT…"
PORT="$PORT" uv run python app.py &
SERVER_PID=$!
sleep 4

if [[ "$LOCAL_ONLY" == "1" ]]; then
  echo "Local: http://127.0.0.1:$PORT"
  wait "$SERVER_PID"; exit 0
fi

echo "Starting ngrok…"
ngrok http "$PORT" --log=stdout > /tmp/s2s_ngrok.log 2>&1 &
NGROK_PID=$!
sleep 5
PUBLIC_URL=$(curl -s http://127.0.0.1:4040/api/tunnels \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null || true)

echo
echo "=================================================="
echo "  বাংলা ভয়েস চ্যাটবট চালু হয়েছে"
echo "  Local : http://127.0.0.1:$PORT"
echo "  Public: ${PUBLIC_URL:-<see /tmp/s2s_ngrok.log>}"
echo "  Open the Public URL, allow the mic, hold the"
echo "  button (or Spacebar) and speak Bangla."
echo "=================================================="
echo "  Ctrl-C to stop."
echo
wait "$SERVER_PID"
