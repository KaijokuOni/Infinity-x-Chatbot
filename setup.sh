#!/usr/bin/env bash
set -e

echo "═══════════════════════════════════════════"
echo "  OmniCare Bangla S2S Chatbot — Setup"
echo "═══════════════════════════════════════════"

# Create virtualenv
if [ ! -d ".venv" ]; then
  echo "→ Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "→ Installing dependencies..."
pip install -q --upgrade pip
pip install -q fastapi uvicorn[standard] python-multipart httpx edge-tts \
            openai anthropic python-dotenv aiofiles pydantic pydantic-settings

# Optional: local Apple Silicon Whisper (recommended for offline STT)
read -p "Install mlx-whisper for local offline STT? (Apple Silicon only) [y/N]: " mlx
if [[ "$mlx" == "y" || "$mlx" == "Y" ]]; then
  pip install -q mlx-whisper
  echo "→ mlx-whisper installed. Set STT_PROVIDER=mlx in your .env"
fi

# Copy .env if it doesn't exist
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "→ Created .env from template. Edit it to add API keys."
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  1. Edit .env and add your API keys"
echo "  2. Make sure Ollama is running:  ollama serve"
echo "  3. Start the server:             source .venv/bin/activate && python app.py"
echo "  4. Open:                         http://localhost:8000"
echo "═══════════════════════════════════════════"
