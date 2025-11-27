# 🔬 Colorectal Tissue Classification App

A comprehensive multi-functional web application for classifying colorectal tissue images using deep learning models. This app provides an intuitive interface for medical professionals and researchers to analyze colorectal tissue samples.

## 🎯 Features

### 1. **Single Image Prediction**
- Upload a single colorectal tissue image
- Get instant classification results with confidence scores
- View probability distribution across all 6 classes
- Interactive Grad-CAM visualization to see where the model focuses
- Color-coded results (Normal vs Abnormal tissue)

### 2. **Batch Prediction**
- Process multiple images at once
- Export results as CSV file
- Summary statistics (total images, normal vs abnormal counts)
- Progress tracking during batch processing

### 3. **Model Comparison**
- Compare predictions from both MobileNetV2 and ResNet50 models
- Side-by-side probability distributions
- Agreement/disagreement indicators
- Individual Grad-CAM visualizations for each model

## 📋 Classification Classes

The app can classify images into 6 categories:
1. **Adenocarcinoma** - Malignant tumor
2. **High-grade IN** - High-grade intraepithelial neoplasia
3. **Low-grade IN** - Low-grade intraepithelial neoplasia
4. **Normal** - Healthy tissue
5. **Polyp** - Benign growth
6. **Serrated adenoma** - Precancerous lesion

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd path/to/keras
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files are in place:**
   - `MobileNetV2/best_Colorectal_Classifier_Balanced.keras`
   - `ResNet50/best_Colorectal_Classifier_ResNet50.keras`

## 💻 Usage

### Running Locally

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

### Deploying to Streamlit Cloud

Want to deploy your app online? See **[DEPLOYMENT.md](DEPLOYMENT.md)** for step-by-step instructions!

**Quick steps:**
1. Upload required files to GitHub
2. Connect to Streamlit Cloud
3. Deploy with one click!

Your app will be live at: `https://your-app-name.streamlit.app`

### Using the App

1. **Select a Model:**
   - Choose between MobileNetV2 or ResNet50 from the sidebar

2. **Choose a Mode:**
   - **Single Image Prediction**: Upload one image for detailed analysis
   - **Batch Prediction**: Upload multiple images for bulk processing
   - **Model Comparison**: Compare predictions from both models

3. **Upload Images:**
   - Supported formats: PNG, JPG, JPEG
   - Recommended: Colorectal tissue histopathology images
   - Image size: Automatically resized to 128x128 pixels

4. **View Results:**
   - Predicted class with confidence score
   - Probability distribution chart
   - Grad-CAM heatmap (optional)
   - Batch results table (for batch mode)

## 📁 Project Structure

```
keras/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── colorectal (3).ipynb           # Training notebook
├── MobileNetV2/
│   └── best_Colorectal_Classifier_Balanced.keras
└── ResNet50/
    └── best_Colorectal_Classifier_ResNet50.keras
```

## 🔧 Technical Details

### Models
- **MobileNetV2 (Balanced)**: Lightweight model with balanced training data
- **ResNet50**: Deeper architecture with transfer learning

### Image Preprocessing
- Images are resized to 128x128 pixels
- Model-specific preprocessing applied:
  - MobileNetV2: MobileNetV2 preprocessing
  - ResNet50: ResNet50 preprocessing

### Grad-CAM
- Gradient-weighted Class Activation Mapping
- Visualizes important regions in the image
- Helps understand model decision-making

## 🎨 Features in Detail

### Probability Distribution
- Horizontal bar chart showing confidence for each class
- Color-coded from red (low) to green (high)
- Percentage labels on each bar

### Grad-CAM Visualization
- Heatmap overlay on original image
- Red/yellow regions indicate important areas
- Helps validate model predictions

### Batch Processing
- Process up to hundreds of images
- Progress bar for tracking
- Exportable CSV results
- Summary statistics

## ⚠️ Important Notes

1. **Medical Disclaimer**: This app is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis.

2. **Model Performance**: Model accuracy depends on image quality and similarity to training data.

3. **Image Quality**: For best results, use high-quality histopathology images similar to the training dataset.

4. **System Requirements**: 
   - Minimum 4GB RAM recommended
   - GPU optional but recommended for faster inference

## 🐛 Troubleshooting

### Model Loading Errors
- Ensure model files are in the correct directories
- Check file paths in `app.py` if models are in different locations

### Memory Issues
- Reduce batch size for batch prediction
- Process images in smaller groups

### Grad-CAM Not Working
- Some models may have different layer names
- Check `LAST_CONV_LAYERS` dictionary in `app.py`

## 📊 Model Performance

Refer to the classification reports in:
- `MobileNetV2/classification_report_Colorectal_Classifier_Balanced.csv`
- `ResNet50/classification_report_ResNet50.csv`

## 🔮 Future Enhancements

Potential features to add:
- [ ] Patient history tracking
- [ ] Export predictions with metadata
- [ ] Integration with DICOM format
- [ ] Real-time camera input
- [ ] Model retraining interface
- [ ] Advanced analytics dashboard
- [ ] Multi-model ensemble predictions

## 📝 License

This project is for research and educational purposes.

## 👥 Credits

Developed for colorectal tissue classification using deep learning models trained on the EBHI-SEG dataset.

---

**For questions or issues, please refer to the training notebook or model documentation.**

