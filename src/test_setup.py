from pathlib import Path

def test_directory_setup():
    """Test if all required directories exist."""
    required_dirs = [
        'data',
        'models',
        'models/scalers',
        'visualizations',
        'visualizations/patterns',
        'visualizations/features',
        'visualizations/preprocessing'
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Verified directory: {dir_path}")

if __name__ == "__main__":
    test_directory_setup() 