from setuptools import setup, find_packages

setup(
    name="bank-churners",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scikit-learn==1.2.2',
        'xgboost==1.7.5',
        'seaborn==0.12.2',
        'matplotlib==3.7.1',
        'joblib==1.2.0',
        'imbalanced-learn==0.10.1',
    ],
)