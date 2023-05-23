from setuptools import setup, find_packages

setup(
    name = 'AutoMS',
    version = '1.0.0',
    packages = find_packages(),
    install_requires = [
        'fitter==1.2.3',
        'matplotlib==3.5.3',
        'pandas==1.4.3',
        'pymzml==2.5.1',
        'scikit-learn==1.2.2',
        'scipy==1.9.0',
        'seaborn==0.11.2',
        'tensorflow==2.9.1',
        'matchms==0.18.0',
        'umap-learn==0.5.3',
        'statsmodels==0.13.5',
        'adjustText==0.8',
        'lightgbm==3.3.5',
        'xgboost==1.2.1',
        'bokeh==2.4.3',
        'tqdm'
    ],
    license = 'AGPLv3',
    author = 'Ji Hongchao',
    author_email = 'ji.hongchao@foxmail.com',
	include_package_data = True
)
