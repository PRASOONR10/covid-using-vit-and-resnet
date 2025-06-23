
COVID-19 Detection from Chest X-rays using ResNet50 and Vision Transformer (ViT)

This project presents a deep learning-based approach for detecting COVID-19 from lung X-ray images. We utilize a fusion of two powerful architectures: ResNet50 (for its residual learning capability) and Vision Transformer (ViT) (for its attention-based feature extraction). The model was implemented using TensorFlow and Keras.

------------------------------------------------------------

🚀 Features

- Fusion of ResNet50 and ViT for robust feature extraction
- Preprocessing and augmentation of X-ray images
- Binary classification: COVID-19 Positive vs Negative
- Model training with visual performance metrics (accuracy, loss)
- Evaluation on real-world datasets

------------------------------------------------------------

🛠️ Tools & Frameworks

- Python
- TensorFlow
- Keras
- NumPy, Matplotlib
- Scikit-learn
- OpenCV (optional for image preprocessing)

------------------------------------------------------------

📁 Directory Structure

covid19-detection/
│
├── data/                     # Dataset (train, val, test splits)
├── models/                   # Model saving and loading
├── notebooks/                # Jupyter notebooks for experiments
├── vit_resnet_fusion.py     # Main script for training/evaluation
├── utils/                    # Utility functions (preprocessing, plotting)
└── README.txt

------------------------------------------------------------

📊 Dataset

- Source: COVIDx / Kaggle datasets
- Preprocessed to ensure class balance and resized to suitable input dimensions

------------------------------------------------------------

🧠 Model Architecture

1. ResNet50: Pretrained on ImageNet, used to extract low-level and mid-level features.
2. ViT (Vision Transformer): Captures long-range dependencies using self-attention.
3. Fusion Layer: Concatenates outputs from both networks.
4. Dense Layers: For final classification (COVID-19 Positive/Negative).

------------------------------------------------------------

🧪 How to Run

# Install dependencies
pip install -r requirements.txt

# Train the model
python vit_resnet_fusion.py --mode train

# Evaluate the model
python vit_resnet_fusion.py --mode test

------------------------------------------------------------

📈 Results

- Accuracy: 98%
- Precision: 98%
- Recall: 98%
- F1-score: 98%


------------------------------------------------------------

✅ Future Work

- Extend to multi-class classification (COVID-19, pneumonia, normal)
- Use Grad-CAM for visual interpretability
- Deploy via Flask/Streamlit for browser-based diagnosis

------------------------------------------------------------

📌 License

This project is for academic/research use only.

------------------------------------------------------------

🙌 Acknowledgements

- Vision Transformer Paper (ViT)
- ResNet Paper
- Open-source datasets and research from the community
