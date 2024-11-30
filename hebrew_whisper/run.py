import os
import eventlet

# Apply monkey patching before importing anything else
eventlet.monkey_patch()

from app import create_app


class Server:
    """
    A class to encapsulate the server logic.
    Ensures a clean and maintainable structure for managing the server lifecycle.
    """

    def __init__(self, host="0.0.0.0", default_port=10000):
        """
        Initialize the server with host and default port.
        """
        self.host = host
        self.default_port = default_port
        self.app, self.socketio = self._initialize_app()

    def _initialize_app(self):
        """
        Create and initialize the Flask app and SocketIO instances.
        """
        return create_app()

    def get_port(self):
        """
        Retrieve the port from the environment or use the default port (for local development).
        """
        return int(os.getenv('PORT', self.default_port))


    def run(self):
        """
        Start the server and handle errors gracefully.
        """
        try:
            port = self.get_port()
            # Determine the link based on the host
            link = f"http://{'127.0.0.1' if self.host == '0.0.0.0' else self.host}:{port}/"
            
            print(f"Starting server on {self.host}:{port}...")
            print(f"Access the server at {link}")
            self.socketio.run(self.app, host=self.host, port=port, log_output=True, debug=False)

            print(f"Server running on {link}")
            print(f"Blueprint URLs: {self.app.url_map}")
        except Exception as e:
            print(f"Error starting the server: {e}")


if __name__ == "__main__":
    server = Server()
    server.run()
