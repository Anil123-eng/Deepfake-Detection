"""
Entry point for the Deepfake Detection web application.

Run with:
    python run.py
"""

from app import app, config

if __name__ == "__main__":
    print(f"Deepfake Detection Server running at http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
