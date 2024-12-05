from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
from app.views import main_blueprint
from app.socket_handlers import SocketHandler
from app.welcome_handler import WelcomeHandler  # Import the WelcomeHandler
import logging


class FlaskAppFactory:
    """Factory class for creating and configuring the Flask application."""

    def __init__(self, config_class: str = 'app.config.Config', cors_enabled: bool = True):
        self.config_class = config_class
        self.cors_enabled = cors_enabled

    def create_app(self) -> Flask:
        """
        Create and configure the Flask application.
        
        Returns:
            Flask: Configured Flask application instance.
        """
        app = Flask(__name__)
        try:
            # Load configuration from the provided config class
            app.config.from_object(self.config_class)

            # Register blueprints
            self.register_blueprints(app)

            # Enable CORS
            if self.cors_enabled:
                CORS(app, resources={r"/api/*": {"origins": "*"}})

            self._initialize_logging(app)
        except Exception as e:
            raise RuntimeError(f"Error creating the Flask application: {e}")
        return app

    @staticmethod
    def register_blueprints(app: Flask) -> None:
        """Register all Flask blueprints."""
        try:
            app.register_blueprint(main_blueprint, url_prefix='/api')
        except Exception as e:
            raise RuntimeError(f"Error registering blueprints: {e}")

    @staticmethod
    def _initialize_logging(app: Flask) -> None:
        """Set up logging for the application."""
        if not app.debug:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            app.logger.addHandler(handler)
        app.logger.info("Flask application initialized successfully.")


class SocketIOFactory:
    """Factory class for creating and configuring the SocketIO instance."""

    def __init__(self, async_mode: str = 'eventlet', logger: bool = True, cors_allowed_origins: str = '*'):
        self.async_mode = async_mode
        self.logger = logger
        self.cors_allowed_origins = cors_allowed_origins

    def create_socketio(self, app: Flask) -> SocketIO:
        """
        Create and configure the SocketIO instance.
        
        Args:
            app (Flask): Flask application instance.
        
        Returns:
            SocketIO: Configured SocketIO instance.
        """
        try:
            socketio = SocketIO(app, cors_allowed_origins=self.cors_allowed_origins, 
                                logger=self.logger, async_mode=self.async_mode)
            app.logger.info("SocketIO initialized successfully.")
            return socketio
        except Exception as e:
            raise RuntimeError(f"Error creating SocketIO instance: {e}")


def create_app(config_class='app.config.Config', cors_enabled=True) -> tuple[Flask, SocketIO]:
    """
    High-level function to create and return the Flask application and SocketIO instance.

    Args:
        config_class (str): The configuration class to use for the Flask app.
        cors_enabled (bool): Whether to enable CORS.

    Returns:
        tuple[Flask, SocketIO]: A tuple containing the Flask application and SocketIO instance.
    """
    try:
        # Create Flask app and SocketIO instance
        app_factory = FlaskAppFactory(config_class=config_class, cors_enabled=cors_enabled)
        socketio_factory = SocketIOFactory()

        app = app_factory.create_app()
        socketio = socketio_factory.create_socketio(app)

        # Initialize SocketHandler to register WebSocket events
        SocketHandler(socketio, app)

        # Initialize WelcomeHandler to register the root ("/") route
        WelcomeHandler(app)

        app.logger.info("Application and SocketIO instances successfully created.")
        return app, socketio
    except Exception as e:
        logging.error(f"Error initializing the Flask app and SocketIO instance: {e}")
        raise RuntimeError(f"Error initializing the Flask app and SocketIO instance: {e}")
