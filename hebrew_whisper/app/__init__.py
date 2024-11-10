from flask import Flask
from .views import main_blueprint
from utilities.file_handler import FileHandler
from flask_apscheduler import APScheduler

# Configure the Flask APScheduler
scheduler = APScheduler()

def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    
    # Register blueprints
    app.register_blueprint(main_blueprint)

    # Initialize and start the scheduler
    initialize_scheduler(app)

    return app

def initialize_scheduler(app):
    """Initialize and start the APScheduler."""
    scheduler.init_app(app)
    scheduler.start()

    # Define the job to check for new training files, passing `app` as an argument
    scheduler.add_job(id='Check Perfect Training Files',
                      func=check_perfect_training_files,
                      args=[app],  # Pass the app instance to the function
                      trigger='interval',
                      minutes=5)  # changed from seconds to minutes for practical usage

def check_perfect_training_files(app):
    """Check for new training files in the perfect training folder."""
    with app.app_context():  # Use app instead of current_app
        try:
            perfect_training_path = app.config['PERFECT_TRAINING_FOLDER']
            new_files = FileHandler.check_new_files(perfect_training_path, ('.wav',))
            if new_files:
                print("Scheduled check found new perfect training files:", new_files)
            else:
                print("No new files found during scheduled check.")
        except Exception as e:
            print(f"Error checking for new files: {e}")
