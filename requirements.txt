import os

# Define the required packages
required_packages = [
    "streamlit",
    "transformers",
    "torch",
    "torchaudio",
    "scipy"
]

# Install each package using pip
for package in required_packages:
    os.system(f"pip install {package}")

print("All required packages have been installed.")
