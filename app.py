from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import json
import logging
import traceback
import sys

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
models = {}
label_encoder = None
feature_names = []
is_ready = False

# Model mapping
MODEL_MAPPING = {
    'Rice_model.pkl': 'rice_kg',
    'Dhal (lentils)_model.pkl': 'dhal_kg',
    'Cooking oil_model.pkl': 'oil_l',
    'Salt_model.pkl': 'salt_kg',
    'Sugar_model.pkl': 'sugar_kg',
    'Tea_model.pkl': 'tea_kg',
    'Biscuits_model.pkl': 'biscuits_packs',
    'Canned Tuna_model.pkl': 'canned_tuna_cans',
    'Water_model.pkl': 'water_l',
}

def load_models():
    """Load models from local files with detailed error handling"""
    global models, label_encoder, feature_names, is_ready
    
    logger.info("🔄 Loading models from local files...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in models directory: {os.listdir('models') if os.path.exists('models') else 'models folder not found'}")
    
    try:
        # Load label encoder
        encoder_path = 'models/label_encoder.pkl'
        if os.path.exists(encoder_path):
            try:
                label_encoder = joblib.load(encoder_path)
                logger.info("✅ Label encoder loaded successfully")
                logger.info(f"Label encoder classes: {label_encoder.classes_}")
            except Exception as e:
                logger.error(f"❌ Failed to load label_encoder: {e}")
                logger.error(traceback.format_exc())
                return
        else:
            logger.error(f"❌ label_encoder.pkl not found at {encoder_path}")
            return
        
        # Load feature names
        if os.path.exists('models/feature_names.txt'):
            with open('models/feature_names.txt', 'r') as f:
                feature_names = f.read().split(',')
            logger.info(f"✅ Features loaded: {feature_names}")
        else:
            feature_names = ['emergency_type_encoded', 'total_people', 'cooking_available', 
                           'children', 'elderly', 'pregnant']
            logger.warning("⚠️ feature_names.txt not found, using defaults")
        
        # Load each model
        loaded_count = 0
        for model_file, target in MODEL_MAPPING.items():
            file_path = f'models/{model_file}'
            if os.path.exists(file_path):
                try:
                    logger.info(f"📥 Loading {model_file}...")
                    models[target] = joblib.load(file_path)
                    logger.info(f"  ✅ Loaded: {target} (from {model_file})")
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"  ❌ Error loading {model_file}: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning(f"  ⚠️ File not found: {model_file}")
        
        is_ready = loaded_count > 0
        logger.info(f"📊 Models loaded: {loaded_count}/{len(MODEL_MAPPING)}")
        logger.info(f"Ready status: {is_ready}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        logger.error(traceback.format_exc())
        is_ready = False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if is_ready else 'loading',
        'models_loaded': len(models),
        'total_models': len(MODEL_MAPPING),
        'is_ready': is_ready,
        'python_version': sys.version,
        'numpy_version': np.__version__
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint for Render health checks"""
    return jsonify({
        'status': 'running',
        'service': 'RescueNet ML API',
        'health': '/health',
        'predict': '/predict'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    if not is_ready:
        return jsonify({
            'success': False,
            'error': 'Models not ready. Please try again.',
            'status': 'loading'
        }), 503
    
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        emergency_type = data.get('emergency_type', 'Flood')
        total_people = int(data.get('total_people', 1))
        cooking_available = 1 if data.get('cooking_available', True) else 0
        children = int(data.get('children', 0))
        elderly = int(data.get('elderly', 0))
        pregnant = int(data.get('pregnant', 0))
        
        logger.info(f"📊 Prediction request: {emergency_type}, {total_people} people")
        logger.info(f"   cooking: {cooking_available}, children: {children}, elderly: {elderly}, pregnant: {pregnant}")
        
        # Encode emergency type
        try:
            encoded_type = label_encoder.transform([emergency_type])[0]
            logger.info(f"   Encoded emergency type: {encoded_type}")
        except Exception as e:
            logger.warning(f"⚠️ Emergency type '{emergency_type}' not in training, using Flood")
            encoded_type = label_encoder.transform(['Flood'])[0]
        
        # Create feature vector
        test_df = pd.DataFrame([{
            'emergency_type_encoded': encoded_type,
            'total_people': total_people,
            'cooking_available': cooking_available,
            'children': children,
            'elderly': elderly,
            'pregnant': pregnant
        }])
        
        logger.info(f"   Feature vector: {test_df.iloc[0].to_dict()}")
        
        # Make predictions using all loaded models
        predictions = {}
        for target, model in models.items():
            try:
                pred = model.predict(test_df)[0]
                pred = max(0, pred)
                predictions[target] = round(float(pred), 2)
            except Exception as e:
                logger.error(f"Error predicting {target}: {e}")
                predictions[target] = 0.0
        
        logger.info(f"✅ Prediction successful for emergency: {emergency_type}")
        logger.info(f"   Predictions: {predictions}")
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'metadata': {
                'emergency_type': emergency_type,
                'total_people': total_people,
                'cooking_available': cooking_available == 1,
                'children': children,
                'elderly': elderly,
                'pregnant': pregnant,
                'models_used': len(models),
                'source': 'ML Model (Random Forest)'
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint to check model loading"""
    return jsonify({
        'is_ready': is_ready,
        'models_loaded': list(models.keys()),
        'models_count': len(models),
        'total_models': len(MODEL_MAPPING),
        'feature_names': feature_names,
        'label_encoder_classes': label_encoder.classes_.tolist() if label_encoder else None,
        'cwd': os.getcwd(),
        'models_dir_exists': os.path.exists('models'),
        'models_dir_files': os.listdir('models') if os.path.exists('models') else []
    })

if __name__ == '__main__':
    # Load models before starting server
    load_models()
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)