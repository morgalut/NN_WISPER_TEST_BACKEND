import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileHandler:
    @staticmethod
    def ensure_directory_exists(directory):
        """Ensure that the directory exists, and if not, create it."""
        try:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Directory checked/created: {directory}")
        except Exception as e:
            logging.error(f"Failed to ensure directory exists: {directory}: {e}")
            raise OSError(f"Failed to ensure directory exists: {directory}: {e}")

    @staticmethod
    def check_new_files(directory, extensions):
        """
        Check for new files with specified extensions in a directory.
        Extensions should be a tuple of strings.
        """
        FileHandler.ensure_directory_exists(directory)
        try:
            new_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(extensions)]
            logging.info(f"Checked for new files in {directory}: {len(new_files)} files found")
            return new_files
        except Exception as e:
            logging.error(f"Failed to check new files in {directory}: {e}")
            raise IOError(f"Failed to check new files in {directory}: {e}")

    @staticmethod
    def delete_file(file_path):
        """Delete a file if it exists, with error handling and logging."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"File deleted: {file_path}")
            else:
                logging.warning(f"Attempted to delete non-existing file: {file_path}")
        except Exception as e:
            logging.error(f"Failed to delete file: {file_path}: {e}")
            raise OSError(f"Failed to delete file: {file_path}: {e}")
