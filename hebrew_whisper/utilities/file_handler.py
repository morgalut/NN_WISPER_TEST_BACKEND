import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileHandler:
    @staticmethod
    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def check_new_files(directory, extensions):
        FileHandler.ensure_directory_exists(directory)
        try:
            return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(extensions)]
        except Exception as e:
            logging.error(f"Failed to check new files in {directory}: {e}")
            raise IOError(f"Failed to check new files in {directory}: {e}")

    @staticmethod
    def delete_file(file_path):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"File deleted: {file_path}")
        except Exception as e:
            logging.error(f"Failed to delete file: {file_path}: {e}")
