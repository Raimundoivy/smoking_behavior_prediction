import logging
from flask import Flask, render_template
import config

if not config.SECRET_KEY:
    raise ValueError("SECRET_KEY not set in environment. Please set it in your .env file.")

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.config.from_object(config)

@app.route('/')
def home():
    """Renders the single page that will host our React component."""
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=config.FLASK_DEBUG)