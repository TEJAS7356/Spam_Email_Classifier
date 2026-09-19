from pathlib import Path

from flask import Flask


def create_app():
    """Application factory: creates and configures the Flask app."""
    app = Flask(__name__)
    app.config["MODEL_DIR"] = Path(app.root_path).parent / "model"

    from .routes import main
    app.register_blueprint(main)

    return app
