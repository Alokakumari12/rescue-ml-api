# app.py - Complete Flask API for model hosting
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import json
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow all origins for mobile apps

# Global variables for models
models = {}
label_encoder = None
feature_names = []
is_ready = False
model_metadata = {}

# ============================================
# MODEL MAPPING - Map file names to food items
# ============================================
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
    'Milk powder_model.pkl': 'milk_powder',
    'Onions_model.pkl': 'onions',
    'Potatoes_model.pkl': 'potatoes',
    'Carrots_model.pkl': 'carrots',
    'Cabbage_model.pkl': 'cabbage',
    'Pumpkin_model.pkl': 'pumpkin',
    'Brinjal_model.pkl': 'brinjal',
    'Coconut_model.pkl': 'coconut',
    'Bread_model.pkl': 'bread',
    'Cooked rice_model.pkl': 'cooked_rice',
    'Noodles_model.pkl': 'noodles',
    'Sandwiches_model.pkl': 'sandwiches',
    'Boiled eggs_model.pkl': 'boiled_eggs',
    'Soup_model.pkl': 'soup',
    # Additional models
    'Vegetable fried rice_model.pkl': 'veg_fried_rice',
    'Chicken fried rice_model.pkl': 'chicken_fried_rice',
    'Rice and vegetable curry_model.pkl': 'rice_veg_curry',
    'Rice and chicken curry_model.pkl': 'rice_chicken_curry',
    'Rice and fish curry_model.pkl': 'rice_fish_curry',
    'Rice and dhal curry_model.pkl': 'rice_dhal_curry',
    'Rice and canned fish_model.pkl': 'rice_canned_fish',
}

# Food display names and units
FOOD_INFO = {
    'rice_kg': {'name': 'Rice', 'unit': 'kg', 'cooking': True},
    'dhal_kg': {'name': 'Dhal (lentils)', 'unit': 'kg', 'cooking': True},
    'oil_l': {'name': 'Cooking Oil', 'unit': 'L', 'cooking': True},
    'salt_kg': {'name': 'Salt', 'unit': 'kg', 'cooking': True},
    'sugar_kg': {'name': 'Sugar', 'unit': 'kg', 'cooking': True},
    'tea_kg': {'name': 'Tea', 'unit': 'kg', 'cooking': True},
    'biscuits_packs': {'name': 'Biscuits', 'unit': 'packs', 'cooking': False},
    'canned_tuna_cans': {'name': 'Canned Tuna', 'unit': 'cans', 'cooking': False},
    'water_l': {'name': 'Water', 'unit': 'L', 'cooking': False},
    'milk_powder': {'name': 'Milk Powder', 'unit': 'kg', 'cooking': False},
    'onions': {'name': 'Onions', 'unit': 'kg', 'cooking': True},
    'potatoes': {'name': 'Potatoes', 'unit': 'kg', 'cooking': True},
    'carrots': {'name': 'Carrots', 'unit': 'kg', 'cooking': True},
    'cabbage': {'name': 'Cabbage', 'unit': 'kg', 'cooking': True},
    'pumpkin': {'name': 'Pumpkin', 'unit': 'kg', 'cooking': True},
    'brinjal': {'name': 'Brinjal', 'unit': 'kg', 'cooking': True},
    'coconut': {'name': 'Coconut', 'unit': 'kg', 'cooking': True},
    'bread': {'name': 'Bread', 'unit': 'kg', 'cooking': False},
    'cooked_rice': {'name': 'Cooked Rice', 'unit': 'kg', 'cooking': False},
    'noodles': {'name': 'Noodles', 'unit': 'packs', 'cooking': False},
    'sandwiches': {'name': 'Sandwiches', 'unit': 'packs', 'cooking': False},
    'boiled_eggs': {'name': 'Boiled Eggs', 'unit': 'packs', 'cooking': False},
    'soup': {'name': 'Soup', 'unit': 'packs', 'cooking': False}
}

# ============================================
# LOAD MODELS
# ============================================
def load_models():
    """Load all models from the models directory"""
    global models, label_encoder, feature_names, is_ready, model_metadata
    
    logger.info("🔄 Loading models...")
    model_path = os.path.join(os.path.dirname(__file__), 'models')
    
    # Check if models directory exists
    if not os.path.exists(model_path):
        logger.error(f"❌ Models directory not found: {model_path}")
        is_ready = False
        return
    
    try:
        # Load label encoder
        encoder_path = os.path.join(model_path, 'label_encoder.pkl')
        if os.path.exists(encoder_path):
            try:
                label_encoder = joblib.load(encoder_path)
                logger.info("✅ Label encoder loaded")
            except Exception as e:
                logger.error(f"❌ Error loading label encoder: {e}")
                label_encoder = None
                is_ready = False
                return
        else:
            logger.error("❌ label_encoder.pkl not found")
            is_ready = False
            return
        
        # Load feature names from metadata
        metadata_path = os.path.join(model_path, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    feature_names = metadata.get('features', [
                        'emergency_type_encoded', 'total_people', 'cooking_available', 
                        'children', 'elderly', 'pregnant', 'days_required'
                    ])
                    model_metadata = metadata
                logger.info(f"✅ Features loaded: {feature_names}")
            except Exception as e:
                logger.warning(f"⚠️ Error loading metadata: {e}")
                feature_names = ['emergency_type_encoded', 'total_people', 'cooking_available', 
                               'children', 'elderly', 'pregnant']
        else:
            logger.warning("⚠️ metadata.json not found, using default features")
            feature_names = ['emergency_type_encoded', 'total_people', 'cooking_available', 
                           'children', 'elderly', 'pregnant']
        
        # Load each model
        loaded_count = 0
        for model_file, target in MODEL_MAPPING.items():
            file_path = os.path.join(model_path, model_file)
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
        
        if not is_ready:
            logger.warning("⚠️ No models loaded. Check the models directory.")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        traceback.print_exc()
        is_ready = False

# Load models on startup
load_models()

# ============================================
# HEALTH CHECK
# ============================================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if is_ready else 'loading',
        'models_loaded': len(models),
        'total_models': len(MODEL_MAPPING),
        'is_ready': is_ready,
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# LIST MODELS
# ============================================
@app.route('/models', methods=['GET'])
def list_models():
    """List all loaded models"""
    return jsonify({
        'models': list(models.keys()),
        'total': len(models),
        'is_ready': is_ready,
        'food_info': FOOD_INFO
    })

# ============================================
# PREDICTION ENDPOINT
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint using trained ML models"""
    # Check if models are ready
    if not is_ready:
        # Try to reload models first
        load_models()
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
        
        # Extract input with defaults
        emergency_type = data.get('emergency_type', 'Flood')
        total_people = int(data.get('total_people', 1))
        cooking_available = 1 if data.get('cooking_available', True) else 0
        children = int(data.get('children', 0))
        elderly = int(data.get('elderly', 0))
        pregnant = int(data.get('pregnant', 0))
        days_required = int(data.get('days_required', 3))
        
        logger.info(f"📊 Prediction request: {emergency_type}, {total_people} people, {days_required} days")
        
        # Encode emergency type
        try:
            if label_encoder is not None:
                encoded_type = label_encoder.transform([emergency_type])[0]
            else:
                # Fallback: use a default encoding
                encoded_type = 0
                logger.warning("⚠️ Label encoder not available, using default encoding")
        except Exception as e:
            logger.warning(f"⚠️ Emergency type '{emergency_type}' not in training, using 'Flood'")
            try:
                encoded_type = label_encoder.transform(['Flood'])[0] if label_encoder else 0
            except:
                encoded_type = 0
        
        # Create feature vector
        try:
            test_df = pd.DataFrame([{
                'emergency_type_encoded': encoded_type,
                'total_people': total_people,
                'cooking_available': cooking_available,
                'children': children,
                'elderly': elderly,
                'pregnant': pregnant,
                'days_required': days_required
            }])
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            return jsonify({
                'success': False,
                'error': f'Error creating feature vector: {str(e)}'
            }), 400
        
        # Make predictions using all loaded models
        predictions = {}
        for target, model in models.items():
            try:
                pred = model.predict(test_df)[0]
                # Ensure non-negative predictions
                pred = max(0, pred)
                predictions[target] = round(float(pred), 2)
            except Exception as e:
                logger.error(f"Error predicting {target}: {e}")
                predictions[target] = 0.0
        
        # Format predictions for mobile app
        formatted_predictions = {}
        for key, value in predictions.items():
            if key in FOOD_INFO:
                info = FOOD_INFO[key]
                formatted_predictions[info['name']] = {
                    'amount': value,
                    'unit': info['unit'],
                    'cooking': info['cooking'],
                    'display': f"{value:.2f} {info['unit']}"
                }
        
        # Calculate water separately
        water_amount = total_people * 3.0 * days_required
        formatted_predictions['Water'] = {
            'amount': water_amount,
            'unit': 'L',
            'cooking': False,
            'display': f"{water_amount:.2f} L"
        }
        
        logger.info(f"✅ Prediction successful for emergency: {emergency_type}")
        
        return jsonify({
            'success': True,
            'predictions': formatted_predictions,
            'metadata': {
                'emergency_type': emergency_type,
                'total_people': total_people,
                'cooking_available': cooking_available == 1,
                'children': children,
                'elderly': elderly,
                'pregnant': pregnant,
                'days_required': days_required,
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

# ============================================
# RELOAD MODELS (for debugging)
# ============================================
@app.route('/reload', methods=['POST'])
def reload_models():
    """Reload models from disk"""
    global is_ready
    load_models()
    return jsonify({
        'success': is_ready,
        'models_loaded': len(models),
        'total_models': len(MODEL_MAPPING)
    })

# ============================================
# RULE-BASED FALLBACK (when models fail)
# ============================================
@app.route('/fallback', methods=['POST'])
def fallback_predict():
    """Rule-based fallback prediction when models are unavailable"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        emergency_type = data.get('emergency_type', 'Flood')
        total_people = int(data.get('total_people', 1))
        cooking_available = data.get('cooking_available', True)
        days_required = int(data.get('days_required', 3))
        
        # Emergency type multipliers
        multipliers = {
            'Flood': 1.2, 'Earthquake': 1.3, 'Tsunami': 1.4,
            'Cyclone': 1.25, 'Landslide': 1.3, 'Fire': 1.1,
            'Drought': 1.15, 'Epidemic': 1.3
        }
        multiplier = multipliers.get(emergency_type, 1.2)
        
        # Base food quantities (per person per day)
        base_foods = {
            'Rice': 0.4, 'Dhal (lentils)': 0.15, 'Cooking Oil': 0.05,
            'Salt': 0.02, 'Sugar': 0.03, 'Tea': 0.02,
            'Biscuits': 0.5, 'Canned Tuna': 0.5, 'Water': 3.0,
            'Milk powder': 0.05, 'Onions': 0.1, 'Potatoes': 0.2
        }
        
        # Adjust based on cooking availability
        if not cooking_available:
            no_cook_multiplier = {
                'Rice': 0.1, 'Dhal (lentils)': 0.1, 'Cooking Oil': 0.1,
                'Salt': 0.5, 'Sugar': 0.5, 'Tea': 0.1,
                'Biscuits': 2.0, 'Canned Tuna': 2.0, 'Water': 1.0,
                'Milk powder': 1.0, 'Onions': 0.1, 'Potatoes': 0.1
            }
        else:
            no_cook_multiplier = {food: 1.0 for food in base_foods}
        
        # Calculate amounts
        predictions = {}
        for food, base in base_foods.items():
            cook_factor = no_cook_multiplier.get(food, 1.0)
            amount = base * total_people * days_required * multiplier * cook_factor
            predictions[food] = round(amount, 2)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'metadata': {
                'emergency_type': emergency_type,
                'total_people': total_people,
                'cooking_available': cooking_available,
                'days_required': days_required,
                'source': 'Rule-based Fallback',
                'multiplier': multiplier
            }
        })
        
    except Exception as e:
        logger.error(f"Fallback error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# ============================================
# ERROR HANDLERS
# ============================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)