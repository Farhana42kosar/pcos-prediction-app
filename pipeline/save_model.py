import os
import joblib

class SaveArtifacts:
    def __init__(self, save_dir="artifacts"):
        """
        Initialize the saver with a directory to save files.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Create folder if it doesn't exist

    def save_model(self, model, model_name="model.pkl"):
        """
        Save the trained ML model.
        """
        path = os.path.join(self.save_dir, model_name)
        joblib.dump(model, path)
        print(f"✅ Model saved at: {path}")
        return path

    def save_preprocessor(self, preprocessor, preprocessor_name="preprocessor.pkl"):
        """
        Save the preprocessing object (scaler, encoder, pipeline, etc.)
        """
        path = os.path.join(self.save_dir, preprocessor_name)
        joblib.dump(preprocessor, path)
        print(f"✅ Preprocessor saved at: {path}")
        return path

    def save_object(self, obj, filename="object.pkl"):
        """
        Save any generic Python object.
        """
        path = os.path.join(self.save_dir, filename)
        joblib.dump(obj, path)
        print(f"✅ Object saved at: {path}")
        return path
