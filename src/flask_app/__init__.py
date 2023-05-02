"""Initialize Flask app."""
from flask import Flask


def init_app():
    """Construct core Flask application."""
    UPLOAD_FOLDER = 'upload/'

    app = Flask(__name__, template_folder='templates')

    with app.app_context():
        # import Flask app
        from . import routes

        # Import Dash application
        from .dashboard import init_dashboard
        app = init_dashboard(app)

        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

        return app
