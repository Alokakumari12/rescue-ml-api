from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import json
import logging
import traceback

app = Flask(__name__)
CORS(app)

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
    """Load models from local files"""
    global models, label_encoder, feature_names, is_ready
    
    logger.info("🔄 Loading models from local files...")
    
    try:
        # Load label encoder
        if os.path.exists('models/label_encoder.pkl'):
            label_encoder = joblib.load('models/label_encoder.pkl')
            logger.info("✅ Label encoder loaded")
        else:
            logger.error("❌ label_encoder.pkl not found")
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
                    models[target] = joblib.load(file_path)
                    logger.info(f"  ✅ Loaded: {target} (from {model_file})")
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"  ❌ Error loading {model_file}: {e}")
            else:
                logger.warning(f"  ⚠️ File not found: {model_file}")
        
        is_ready = loaded_count > 0
        logger.info(f"📊 Models loaded: {loaded_count}/{len(MODEL_MAPPING)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        traceback.print_exc()
        is_ready = False

# Load models on startup
load_models()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if is_ready else 'loading',
        'models_loaded': len(models),
        'total_models': len(MODEL_MAPPING),
        'is_ready': is_ready
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    if not is_ready:
        return jsonify({
            'success': False,
            'error': 'Models not ready. Please try again.'
        }), 503
    
    try:
        data = request.json
        
        emergency_type = data.get('emergency_type', 'Flood')
        total_people = int(data.get('total_people', 1))
        cooking_available = 1 if data.get('cooking_available', True) else 0
        children = int(data.get('children', 0))
        elderly = int(data.get('elderly', 0))
        pregnant = int(data.get('pregnant', 0))
        
        logger.info(f"📊 Prediction request: {emergency_type}, {total_people} people")
        
        # Encode emergency type
        try:
            encoded_type = label_encoder.transform([emergency_type])[0]
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
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)