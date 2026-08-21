"""
fix_models.py - Convert models to be compatible with current NumPy version
Run this once to fix all models for the current environment.
"""

import joblib
import numpy as np
import os
import pickle
import warnings
import sys
import traceback  # <-- ADD THIS LINE

warnings.filterwarnings("ignore")

print("=" * 60)
print("🔧 FIXING MODELS FOR NUMPY COMPATIBILITY")
print("=" * 60)

# Create a dummy numpy._core module for loading
def setup_numpy_compat():
    """Set up compatibility for numpy._core imports"""
    import types
    if not hasattr(sys.modules, 'numpy._core'):
        sys.modules['numpy._core'] = types.ModuleType('numpy._core')
    if not hasattr(sys.modules['numpy._core'], 'multiarray'):
        sys.modules['numpy._core'].multiarray = np.core.multiarray
    if not hasattr(sys.modules['numpy._core'], 'umath'):
        sys.modules['numpy._core'].umath = np.core.umath

setup_numpy_compat()

def load_with_compat(filepath):
    """Load a pickle file with compatibility for numpy._core"""
    try:
        # Try joblib first
        return joblib.load(filepath)
    except Exception as e:
        if 'numpy._core' in str(e):
            print(f"  ⚠️ NumPy compatibility issue, using pickle fallback...")
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise

def save_model(obj, filepath):
    """Save model using current NumPy version"""
    joblib.dump(obj, filepath)
    print(f"  ✅ Saved: {filepath}")

def fix_model(filepath):
    """Fix a single model file"""
    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        return False
    
    print(f"📥 Processing: {filepath}")
    
    try:
        # Load the model with compatibility
        obj = load_with_compat(filepath)
        
        # Save with current NumPy version
        base, ext = os.path.splitext(filepath)
        new_filepath = f"{base}_fixed{ext}"
        save_model(obj, new_filepath)
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def fix_all_models():
    """Fix all models in the models directory"""
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        print(f"❌ Directory '{models_dir}' not found!")
        return
    
    print(f"\n📂 Processing models in: {models_dir}")
    print("-" * 40)
    
    # Get all .pkl files
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    # Sort for consistent output
    pkl_files.sort()
    
    fixed_count = 0
    for filename in pkl_files:
        # Skip files that are already fixed
        if '_fixed.pkl' in filename:
            print(f"⏭️  Skipping already fixed: {filename}")
            continue
            
        filepath = os.path.join(models_dir, filename)
        if fix_model(filepath):
            fixed_count += 1
    
    print("-" * 40)
    print(f"✅ Fixed {fixed_count} models")
    print(f"📁 Fixed files are saved as *_fixed.pkl")
    
    # Create a list of fixed files
    fixed_files = [f for f in os.listdir(models_dir) if f.endswith('_fixed.pkl')]
    print(f"\n📋 Fixed files:")
    for f in fixed_files:
        print(f"   • {f}")

def update_app_py():
    """Print instructions for updating app.py"""
    print("\n" + "=" * 60)
    print("📝 UPDATE YOUR app.py")
    print("=" * 60)
    print("""
To use the fixed models, update your app.py to load '_fixed.pkl' files:

Replace the MODEL_MAPPING dictionary with:

MODEL_MAPPING = {
    'Rice_model_fixed.pkl': 'rice_kg',
    'Dhal (lentils)_fixed.pkl': 'dhal_kg',
    'Cooking oil_fixed.pkl': 'oil_l',
    'Salt_fixed.pkl': 'salt_kg',
    'Sugar_fixed.pkl': 'sugar_kg',
    'Tea_fixed.pkl': 'tea_kg',
    'Biscuits_fixed.pkl': 'biscuits_packs',
    'Canned Tuna_fixed.pkl': 'canned_tuna_cans',
    'Water_fixed.pkl': 'water_l',
}

Or keep the original mapping and modify the file path:

for model_file, target in MODEL_MAPPING.items():
    # Try fixed version first
    fixed_path = f'models/{model_file.replace(".pkl", "_fixed.pkl")}'
    file_path = fixed_path if os.path.exists(fixed_path) else f'models/{model_file}'
""")

if __name__ == '__main__':
    try:
        fix_all_models()
        update_app_py()
        print("\n" + "=" * 60)
        print("✅ COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Update your app.py to use the '_fixed.pkl' files")
        print("2. Run: python app.py")
        print("3. Test: curl http://localhost:10000/health")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(traceback.format_exc())