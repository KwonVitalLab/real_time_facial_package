from setuptools import setup, find_packages

setup(
    name="realtime_facial_features",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.7.0",
        "opencv-python",
        "numpy",
        "py-feat",
    ],
    entry_points={
        'console_scripts': [
            'realtime-facial-features=realtime_facial_features.run:main',
        ],
    },
    package_data={
        'realtime_facial_features': ['models/*.pkl'],
    },
    author="Your Name",
    description="Real-time facial analysis and gaze estimation tool.",
    zip_safe=False,
)
