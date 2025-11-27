import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model
import os
from io import BytesIO
import pickle
import base64

# Increase PIL image size limit to handle large matplotlib figures
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check

# Page configuration
st.set_page_config(
    page_title="Colorectal Tissue Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class labels (based on your training)
CLASS_LABELS = ['Adenocarcinoma', 'High-grade IN', 'Low-grade IN', 'Normal', 'Polyp', 'Serrated adenoma']
IMG_SIZE = (128, 128)

# Model paths
MODEL_PATHS = {
    'MobileNetV2': 'MobileNetV2/best_Colorectal_Classifier_Balanced.keras',
    'ResNet50': 'ResNet50/best_Colorectal_Classifier_ResNet50.keras'
}

# Last convolutional layer names for Grad-CAM
LAST_CONV_LAYERS = {
    'MobileNetV2': 'out_relu',
    'ResNet50': 'conv5_block3_out'
}

# Preprocessing functions
PREPROCESS_FUNCTIONS = {
    'MobileNetV2': mobilenet_preprocess,
    'ResNet50': resnet_preprocess
}

@st.cache_resource
def load_model(model_name):
    """Load and cache the model"""
    model_path = MODEL_PATHS[model_name]
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, model_name):
    """Preprocess image for the specific model"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert RGBA to RGB if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, IMG_SIZE)
    
    # Apply model-specific preprocessing
    preprocess_fn = PREPROCESS_FUNCTIONS[model_name]
    img_preprocessed = preprocess_fn(np.expand_dims(img_resized, axis=0))
    
    return img_preprocessed, img_resized

def predict_image(model, image_preprocessed, model_name):
    """Make prediction on preprocessed image"""
    predictions = model.predict(image_preprocessed, verbose=0)[0]
    pred_index = np.argmax(predictions)
    pred_label = CLASS_LABELS[pred_index]
    confidence = predictions[pred_index]
    
    # Get all class probabilities
    class_probs = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
    
    return pred_label, confidence, class_probs, pred_index

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    try:
        # Get the last convolutional layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # Handle model.inputs - it might be a list or a single tensor
        model_inputs = model.inputs
        if isinstance(model_inputs, list):
            if len(model_inputs) == 1:
                model_inputs = model_inputs[0]
        
        # Create grad model that outputs both conv layer and predictions
        grad_model = Model(inputs=model_inputs, outputs=[last_conv_layer.output, model.output])
        
        with tf.GradientTape() as tape:
            # Get outputs - might be list or tuple
            outputs = grad_model(img_array, training=False)
            
            # Handle both list and tuple outputs
            if isinstance(outputs, (list, tuple)):
                last_conv_layer_output = outputs[0]
                preds = outputs[1]
            else:
                # Single output (shouldn't happen but handle it)
                last_conv_layer_output = outputs
                preds = model(img_array, training=False)
            
            # Ensure preds is a tensor - handle list/tuple cases
            if isinstance(preds, (list, tuple)):
                # If it's a list/tuple, take the first element if it's a single-item list
                if len(preds) == 1:
                    preds = preds[0]
                else:
                    # Multiple outputs, take the predictions (usually the last one)
                    preds = preds[-1] if len(preds) > 1 else preds[0]
            
            # Convert to tensor if not already
            if not isinstance(preds, tf.Tensor):
                try:
                    preds = tf.convert_to_tensor(preds)
                except:
                    # If conversion fails, try to get numpy array first
                    if hasattr(preds, 'numpy'):
                        preds = tf.convert_to_tensor(preds.numpy())
                    else:
                        preds = tf.convert_to_tensor(np.array(preds))
            
            # Handle pred_index conversion
            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]).numpy())
            else:
                # Ensure pred_index is an integer
                if isinstance(pred_index, (tuple, list)):
                    pred_index = int(pred_index[0]) if len(pred_index) > 0 else 0
                elif isinstance(pred_index, (np.integer, np.int64, np.int32)):
                    pred_index = int(pred_index)
                elif hasattr(pred_index, 'numpy'):
                    pred_index = int(pred_index.numpy())
                else:
                    pred_index = int(pred_index)
            
            # Get the class channel - preds shape is (batch, classes)
            # Use tf.gather or direct indexing
            if len(preds.shape) == 2:
                class_channel = preds[0, pred_index]
            else:
                # Fallback to gather
                class_channel = tf.gather(preds[0], pred_index, axis=0)
        
        # Compute gradients
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get the first (and only) image's conv output
        last_conv_layer_output = last_conv_layer_output[0]
        
        # Generate heatmap
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def display_gradcam(img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on image"""
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

# Main App
def main():
    st.title("🔬 Colorectal Tissue Classification App")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=['MobileNetV2', 'ResNet50'],
        help="Choose which trained model to use for prediction"
    )
    
    # Load model
    with st.spinner(f"Loading {selected_model} model..."):
        model = load_model(selected_model)
    
    if model is None:
        st.error("Failed to load model. Please check the model file path.")
        return
    
    st.sidebar.success(f"✅ {selected_model} model loaded successfully!")
    
    # App mode selection
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["Single Image Prediction", "Batch Prediction", "Model Comparison"],
        help="Choose how you want to use the app"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.info(f"**Model:** {selected_model}\n\n**Classes:** {len(CLASS_LABELS)}\n\n**Input Size:** {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    
    # Single Image Prediction Mode
    if app_mode == "Single Image Prediction":
        st.header("📸 Single Image Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a colorectal tissue image for classification"
            )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Preprocess and predict
            with st.spinner("Processing image..."):
                img_preprocessed, img_resized = preprocess_image(image, selected_model)
                pred_label, confidence, class_probs, pred_index = predict_image(model, img_preprocessed, selected_model)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Class", pred_label)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            with col3:
                # Color code based on class
                if pred_label == "Normal":
                    st.success("✅ Normal Tissue")
                else:
                    st.warning("⚠️ Abnormal Tissue Detected")
            
            # Probability distribution
            st.subheader("📊 Probability Distribution")
            prob_df = pd.DataFrame(list(class_probs.items()), columns=['Class', 'Probability'])
            prob_df = prob_df.sort_values('Probability', ascending=False)
            
            # Display as table first
            with st.expander("View Probability Table", expanded=False):
                st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}), use_container_width=True)
            
            # Use Streamlit's native bar chart as primary method (more reliable)
            st.bar_chart(prob_df.set_index('Class')['Probability'])
            
            # Also try matplotlib as secondary visualization
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(prob_df['Class'], prob_df['Probability'], 
                              color=plt.cm.RdYlGn(prob_df['Probability']))
                ax.set_xlabel('Probability', fontsize=12)
                ax.set_title('Class Probabilities', fontsize=14, fontweight='bold')
                ax.set_xlim(0, 1)
                
                # Add value labels on bars
                for i, (idx, row) in enumerate(prob_df.iterrows()):
                    ax.text(row['Probability'] + 0.01, i, f'{row["Probability"]:.2%}', 
                           va='center', fontsize=10)
                
                plt.tight_layout()
                # Save figure to buffer to avoid PIL decompression bomb error
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                st.image(buf, use_container_width=True, caption="Detailed Probability Chart")
                plt.close(fig)
            except Exception as e:
                st.info(f"Matplotlib visualization unavailable. Using native chart above.")
            
            # Grad-CAM visualization
            st.subheader("🔍 Grad-CAM Visualization")
            show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
            
            if show_gradcam:
                with st.spinner("Generating Grad-CAM visualization..."):
                    last_conv_layer = LAST_CONV_LAYERS[selected_model]
                    heatmap = get_gradcam_heatmap(img_preprocessed, model, last_conv_layer, pred_index)
                    
                    if heatmap is not None:
                        gradcam_img = display_gradcam(img_resized, heatmap)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), 
                                   caption="Original Image", use_container_width=True)
                        with col2:
                            st.image(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB), 
                                   caption="Grad-CAM Overlay", use_container_width=True)
                        
                        st.caption("💡 The colored regions show where the model focuses when making its prediction.")
    
    # Batch Prediction Mode
    elif app_mode == "Batch Prediction":
        st.header("📦 Batch Prediction")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} image(s) uploaded")
            
            if st.button("🚀 Process All Images", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        image = Image.open(uploaded_file)
                        img_preprocessed, _ = preprocess_image(image, selected_model)
                        pred_label, confidence, _, _ = predict_image(model, img_preprocessed, selected_model)
                        
                        results.append({
                            'Filename': uploaded_file.name,
                            'Predicted Class': pred_label,
                            'Confidence': f"{confidence:.2%}",
                            'Confidence_Value': confidence
                        })
                    except Exception as e:
                        results.append({
                            'Filename': uploaded_file.name,
                            'Predicted Class': 'Error',
                            'Confidence': str(e),
                            'Confidence_Value': 0
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Display results
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Confidence_Value', ascending=False)
                results_df = results_df.drop('Confidence_Value', axis=1)
                
                st.subheader("📊 Batch Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"batch_predictions_{selected_model}.csv",
                    mime="text/csv"
                )
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    normal_count = len(results_df[results_df['Predicted Class'] == 'Normal'])
                    st.metric("Normal Tissue", normal_count)
                with col3:
                    abnormal_count = len(results_df) - normal_count
                    st.metric("Abnormal Tissue", abnormal_count)
    
    # Model Comparison Mode
    elif app_mode == "Model Comparison":
        st.header("⚖️ Model Comparison")
        st.info("Compare predictions from both MobileNetV2 and ResNet50 models")
        
        uploaded_file = st.file_uploader(
            "Upload an image for comparison",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to compare predictions from both models"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Initialize session state for comparison results
            if 'comparison_results' not in st.session_state:
                st.session_state.comparison_results = None
            if 'comparison_image_id' not in st.session_state:
                st.session_state.comparison_image_id = None
            
            # Check if we need to reprocess (new image or button clicked)
            current_image_id = id(uploaded_file)
            need_reprocess = (st.session_state.comparison_image_id != current_image_id or 
                            st.session_state.comparison_results is None)
            
            compare_button = st.button("🔄 Compare Models", type="primary")
            
            if compare_button or need_reprocess:
                col1, col2 = st.columns(2)
                
                results = {}
                
                for model_name in ['MobileNetV2', 'ResNet50']:
                    with st.spinner(f"Processing with {model_name}..."):
                        comp_model = load_model(model_name)
                        if comp_model is not None:
                            img_preprocessed, img_resized = preprocess_image(image, model_name)
                            pred_label, confidence, class_probs, pred_index = predict_image(
                                comp_model, img_preprocessed, model_name
                            )
                            # Store results (model and images will be reloaded/reprocessed when needed for Grad-CAM)
                            # Store image as base64 to preserve in session state
                            _, buffer = cv2.imencode('.png', img_resized)
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            results[model_name] = {
                                'pred_label': pred_label,
                                'confidence': float(confidence),
                                'class_probs': {k: float(v) for k, v in class_probs.items()},
                                'pred_index': int(pred_index),
                                'img_resized_base64': img_base64,  # Store as base64
                                'img_preprocessed_shape': img_preprocessed.shape,  # Store shape info
                                'model_name': model_name  # Store model name to reload later
                            }
                
                # Store results in session state
                st.session_state.comparison_results = results
                st.session_state.comparison_image_id = current_image_id
            
            # Display results from session state
            results = st.session_state.comparison_results
            
            if results and len(results) == 2:
                    # Display comparison
                    col1, col2 = st.columns(2)
                    
                    for idx, (model_name, result) in enumerate(results.items()):
                        with col1 if idx == 0 else col2:
                            st.subheader(f"{model_name} Results")
                            st.metric("Predicted Class", result['pred_label'])
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            # Probability bar chart
                            prob_df = pd.DataFrame(
                                list(result['class_probs'].items()),
                                columns=['Class', 'Probability']
                            ).sort_values('Probability', ascending=False)
                            
                            try:
                                fig, ax = plt.subplots(figsize=(8, 5))
                                ax.barh(prob_df['Class'], prob_df['Probability'], 
                                      color=plt.cm.RdYlGn(prob_df['Probability']))
                                ax.set_xlabel('Probability')
                                ax.set_title(f'{model_name} Probabilities')
                                ax.set_xlim(0, 1)
                                # Save figure to buffer to avoid PIL decompression bomb error
                                buf = BytesIO()
                                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
                                buf.seek(0)
                                st.image(buf, use_container_width=True)
                                plt.close(fig)
                            except Exception as e:
                                # Fallback to Streamlit native chart
                                st.bar_chart(prob_df.set_index('Class')['Probability'])
                                st.warning(f"Matplotlib display failed for {model_name}, using native chart.")
                            
                            # Grad-CAM - Use stored model from results
                            show_gradcam_key = f"gradcam_{model_name}"
                            
                            if st.checkbox(f"Show Grad-CAM ({model_name})", key=show_gradcam_key):
                                with st.spinner(f"Generating Grad-CAM for {model_name}..."):
                                    try:
                                        # Reload model and reprocess image
                                        comp_model = load_model(result['model_name'])
                                        
                                        # Decode base64 image
                                        img_bytes = base64.b64decode(result['img_resized_base64'])
                                        nparr = np.frombuffer(img_bytes, np.uint8)
                                        img_resized = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                        
                                        # Reprocess image for model
                                        img_preprocessed, _ = preprocess_image(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)), result['model_name'])
                                        
                                        last_conv_layer = LAST_CONV_LAYERS[result['model_name']]
                                        heatmap = get_gradcam_heatmap(
                                            img_preprocessed, 
                                            comp_model, 
                                            last_conv_layer, 
                                            result['pred_index']
                                        )
                                        if heatmap is not None:
                                            gradcam_img = display_gradcam(img_resized, heatmap)
                                            
                                            col_grad1, col_grad2 = st.columns(2)
                                            with col_grad1:
                                                st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), 
                                                       caption=f"{model_name} Original", use_container_width=True)
                                            with col_grad2:
                                                st.image(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB), 
                                                       caption=f"{model_name} Grad-CAM", use_container_width=True)
                                        else:
                                            st.warning(f"Could not generate Grad-CAM for {model_name}")
                                    except Exception as e:
                                        st.error(f"Error generating Grad-CAM: {str(e)}")
                                        import traceback
                                        st.code(traceback.format_exc())
                    
                    # Agreement check
                    st.markdown("---")
                    if results['MobileNetV2']['pred_label'] == results['ResNet50']['pred_label']:
                        st.success(f"✅ Both models agree: **{results['MobileNetV2']['pred_label']}**")
                    else:
                        st.warning(f"⚠️ Models disagree:\n- MobileNetV2: **{results['MobileNetV2']['pred_label']}** ({results['MobileNetV2']['confidence']:.2%})\n- ResNet50: **{results['ResNet50']['pred_label']}** ({results['ResNet50']['confidence']:.2%})")

if __name__ == "__main__":
    main()

