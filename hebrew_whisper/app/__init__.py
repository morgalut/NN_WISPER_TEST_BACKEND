import os
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
from app.views import main_blueprint
from app.socket_handlers import SocketHandler
from app.welcome_handler import WelcomeHandler  # Import the WelcomeHandler


class FlaskAppFactory:
    """Factory class for creating and configuring the Flask application."""

    def __init__(self, config_class: str = 'app.config.Config', cors_enabled: bool = True):
        self.config_class = config_class
        self.cors_enabled = cors_enabled

    def create_app(self) -> Flask:
        """Create and configure the Flask application."""
        try:
            app = Flask(__name__)
            app.config.from_object(self.config_class)  # Load configuration from the provided config class

            # Register blueprints
            self.register_blueprints(app)

            # Enable CORS if required
            if self.cors_enabled:
                CORS(app)

            return app
        except Exception as e:
            raise RuntimeError(f"Error creating the Flask application: {e}")

    @staticmethod
    def register_blueprints(app: Flask) -> None:
        """Register all Flask blueprints."""
        try:
            app.register_blueprint(main_blueprint, url_prefix='/api')  # Prefix all blueprint routes with `/api`
        except Exception as e:
            raise RuntimeError(f"Error registering blueprints: {e}")


class SocketIOFactory:
    """Factory class for creating and configuring the SocketIO instance."""

    def __init__(self, async_mode: str = 'eventlet', logger: bool = True, cors_allowed_origins: str = '*'):
        self.async_mode = async_mode
        self.logger = logger
        self.cors_allowed_origins = cors_allowed_origins

    def create_socketio(self, app: Flask) -> SocketIO:
        """Create and configure the SocketIO instance."""
        try:
            return SocketIO(app, cors_allowed_origins=self.cors_allowed_origins, logger=self.logger, async_mode=self.async_mode)
        except Exception as e:
            raise RuntimeError(f"Error creating SocketIO instance: {e}")


def create_app() -> tuple[Flask, SocketIO]:
    """
    High-level function to create and return the Flask application and SocketIO instance.

    Returns:
        A tuple containing the Flask application and SocketIO instance.
    """
    try:
        # Create Flask app and SocketIO instance
        app_factory = FlaskAppFactory()
        socketio_factory = SocketIOFactory()

        app = app_factory.create_app()
        socketio = socketio_factory.create_socketio(app)

        # Initialize SocketHandler to register WebSocket events
        SocketHandler(socketio, app)

        # Initialize WelcomeHandler to register the root ("/") route
        WelcomeHandler(app)

        return app, socketio
    except Exception as e:
        raise RuntimeError(f"Error initializing the Flask app and SocketIO instance: {e}")
