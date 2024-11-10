import os
import shutil
import logging

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileHandler:
    @staticmethod
    def ensure_directory_exists(directory):
        """
        Ensure the directory exists. If not, create it.
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")

    @staticmethod
    def save_file(file, path):
        """
        Save a file to the specified path with error handling.
        Ensures the directory for the path exists before saving the file.
        """
        try:
            FileHandler.ensure_directory_exists(os.path.dirname(path))
            file.save(path)
            logging.info(f"File saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save file to {path}: {e}")
            raise IOError(f"Could not save file to {path}: {e}")

    @staticmethod
    def check_new_files(directory, extensions):
        """
        Check for new files with specified extensions in the given directory.
        Ensures the directory exists before attempting to list files.
        """
        FileHandler.ensure_directory_exists(directory)
        try:
            new_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(extensions)]
            logging.info(f"New files checked: {new_files}")
            return new_files
        except Exception as e:
            logging.error(f"Failed to check new files in {directory}: {e}")
            raise IOError(f"Failed to check new files in {directory}: {e}")

    @staticmethod
    def copy_file(source, destination):
        """
        Copy a file from source to destination with logging.
        Ensures the destination directory exists before copying.
        """
        try:
            FileHandler.ensure_directory_exists(os.path.dirname(destination))
            shutil.copy2(source, destination)
            logging.info(f"File copied from {source} to {destination}")
        except Exception as e:
            logging.error(f"Failed to copy file from {source} to {destination}: {e}")
            raise IOError(f"Could not copy file from {source} to {destination}: {e}")

    @staticmethod
    def delete_file(file_path):
        """
        Delete a file and log the action.
        Checks if the file exists before attempting to delete.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"File deleted: {file_path}")
            else:
                logging.error(f"File does not exist: {file_path}")
                raise IOError(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Failed to delete file: {file_path}: {e}")
            raise IOError(f"Could not delete file: {file_path}: {e}")
