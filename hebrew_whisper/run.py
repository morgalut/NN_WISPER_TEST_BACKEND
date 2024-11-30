import eventlet
eventlet.monkey_patch()  # Must be the first line of the file

import os
from app import create_app

# Create app and socketio instances
app, socketio = create_app()

def run_server():
    """
    Run the server locally for development purposes.
    """
    host = "0.0.0.0"
    default_port = 10000
    port = int(os.getenv('PORT', default_port))

    link = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/"
    print(f"Starting server on {host}:{port}...")
    print(f"Access the server at {link}")

    # Run the app with SocketIO
    socketio.run(app, host=host, port=port, log_output=True, debug=False)


if __name__ == "__main__":
    run_server()
