from django.apps import AppConfig
import logging
import os
from django.conf import settings
from .stegomarkov import build_model  # Use your custom function

# Set up basic logging
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)

class FmsappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fmsApp'

    def ready(self):
        print("========== FmsAppConfig ready() method called! ==========")  # Debug statement
        logger.info("FmsAppConfig ready() method called!")
        global markov_model
        try:
            logging.info("Loading Markov model during server startup...")

            # Correct path to the model file
            model_path = os.path.join(settings.BASE_DIR, "markov_models", "legal_corpus.json")
            
            if not os.path.isfile(model_path):
                logging.error(f"Markov model file not found at {model_path}")
                return
            
            # Use the build_model function from stegomarkov
            markov_model = build_model(model_path)
            
        except Exception as e:
            logging.error(f"Failed to load Markov model: {str(e)}")
