from flask import Flask, jsonify, request


class WelcomeHandler:
    """
    A class to handle the root ("/") route and provide a welcome message in JSON format.
    """

    def __init__(self, app: Flask):
        """
        Initialize the WelcomeHandler and register the route.

        Args:
            app: The Flask application instance.
        """
        self.app = app
        self.register_routes()

    def register_routes(self) -> None:
        """Register the root ("/") route."""
        @self.app.route("/", methods=["GET"])
        def welcome():
            """Return a JSON welcome message."""
            client_ip = request.remote_addr or "unknown"
            return jsonify({
                "message": "Welcome to the Flask SocketIO server!",
                "client_ip": client_ip,
                "documentation_url": "/api"
            }), 200
