
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
📊 Dataset
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

We used publicly available chest X-ray image datasets for training and evaluation:

COVIDx Dataset (GitHub - COVID-Net) link:https://github.com/lindawangg/COVID-Net

Kaggle COVID-19 Radiography Database
(Link:https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

🧾 Dataset Summary:
Class	Sample Count (Approx.)
COVID-19	3,616
Normal	10,192
Pneumonia	1,345

(Counts may vary depending on preprocessing/filtering.)

⚙️ Preprocessing Steps:
Converted all images to grayscale or RGB format

Resized images to a fixed shape (e.g., 224×224)

Normalized pixel values (0–1 scale)

Applied data augmentation: rotation, flip, zoom, contrast

📂 Splits:
Training Set: 70%

Validation Set: 15%
Test Set: 15%
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
