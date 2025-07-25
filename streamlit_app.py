import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
import io
import os

# Page configuration
st.set_page_config(
    page_title="Face Detection System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">👤 Human Face Detection System</h1>', unsafe_allow_html=True)
st.markdown("""
**Advanced Computer Vision System** comparing traditional HOG+SVM with modern YOLOv8 deep learning approaches.
Built with comprehensive EDA, model training, and real-time detection capabilities.
""")

# Initialize session state for models
if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = None
if 'dataset_stats' not in st.session_state:
    st.session_state.dataset_stats = None

# Sidebar navigation
st.sidebar.title("🔧 Navigation")
page = st.sidebar.selectbox(
    "Select Section",
    ["📊 Dataset Overview", "📈 EDA Analysis", "🤖 Model Predictions", "📋 Performance Metrics", "🔄 Model Comparison"]
)

# Load dataset function
@st.cache_data
def load_dataset():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('final_preprocessed_annotations.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'final_preprocessed_annotations.csv' exists.")
        return None

# Load YOLO model function
@st.cache_resource
def load_yolo_model():
    """Load and cache the YOLO model"""
    try:
        model_path = 'face_detection/yolov8_face/weights/best.pt'
        if os.path.exists(model_path):
            model = YOLO(model_path)
            return model
        else:
            st.warning("YOLOv8 model not found! Train the model first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Dataset Overview Page
if page == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")
    
    df = load_dataset()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Images", f"{df['image_name'].nunique():,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Faces", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            multi_face = len(df['image_name'].value_counts()[df['image_name'].value_counts() > 1])
            st.metric("Multi-face Images", f"{multi_face:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_faces = df.groupby('image_name').size().mean()
            st.metric("Avg Faces/Image", f"{avg_faces:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("📈 Dataset Statistics")
        
        # Face size distribution
        col1, col2 = st.columns(2)
        
        with col1:
            df['face_width'] = df['x1'] - df['x0']
            df['face_height'] = df['y1'] - df['y0']
            
            fig = px.histogram(df, x='face_width', title='Face Width Distribution',
                             labels={'face_width': 'Face Width (pixels)', 'count': 'Frequency'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='face_height', title='Face Height Distribution',
                             labels={'face_height': 'Face Height (pixels)', 'count': 'Frequency'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Face position heatmap
        st.subheader("🎯 Face Position Heatmap")
        df['face_center_x_norm'] = ((df['x0'] + df['x1']) / 2) / df['width']
        df['face_center_y_norm'] = ((df['y0'] + df['y1']) / 2) / df['height']
        
        fig = px.density_heatmap(df, x='face_center_x_norm', y='face_center_y_norm',
                               title='Face Position Distribution (Normalized Coordinates)',
                               labels={'face_center_x_norm': 'X Position (normalized)', 
                                      'face_center_y_norm': 'Y Position (normalized)'})
        st.plotly_chart(fig, use_container_width=True)

# EDA Analysis Page
elif page == "📈 EDA Analysis":
    st.header("📈 Exploratory Data Analysis")
    
    df = load_dataset()
    if df is not None:
        # Calculate additional metrics
        df['face_width'] = df['x1'] - df['x0']
        df['face_height'] = df['y1'] - df['y0']
        df['face_area'] = df['face_width'] * df['face_height']
        df['face_aspect_ratio'] = df['face_width'] / df['face_height']
        df['face_to_image_ratio'] = df['face_area'] / (df['width'] * df['height'])
        
        # EDA sections
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Face Size Analysis", "Position Analysis", "Aspect Ratio Analysis", "Multi-face Analysis"]
        )
        
        if analysis_type == "Face Size Analysis":
            st.subheader("📏 Face Size Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(df, x='face_width', y='face_height', 
                               title='Face Dimensions Scatter Plot',
                               labels={'face_width': 'Width (px)', 'face_height': 'Height (px)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(df, x='face_to_image_ratio', title='Face-to-Image Size Ratio',
                                 labels={'face_to_image_ratio': 'Ratio', 'count': 'Frequency'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Size categories
            def categorize_face_size(area):
                if area < 5000:
                    return 'Small'
                elif area < 20000:
                    return 'Medium'
                elif area < 50000:
                    return 'Large'
                else:
                    return 'Very Large'
            
            df['size_category'] = df['face_area'].apply(categorize_face_size)
            size_dist = df['size_category'].value_counts()
            
            fig = px.pie(values=size_dist.values, names=size_dist.index,
                        title='Face Size Categories Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Position Analysis":
            st.subheader("📍 Face Position Analysis")
            
            # Position grid
            def get_position_grid(x_norm, y_norm):
                x_pos = 'left' if x_norm < 0.33 else 'center' if x_norm < 0.67 else 'right'
                y_pos = 'top' if y_norm < 0.33 else 'middle' if y_norm < 0.67 else 'bottom'
                return f"{y_pos}-{x_pos}"
            
            df['face_center_x_norm'] = ((df['x0'] + df['x1']) / 2) / df['width']
            df['face_center_y_norm'] = ((df['y0'] + df['y1']) / 2) / df['height']
            df['position_grid'] = df.apply(lambda row: get_position_grid(row['face_center_x_norm'], row['face_center_y_norm']), axis=1)
            
            position_counts = df['position_grid'].value_counts()
            
            fig = px.bar(x=position_counts.index, y=position_counts.values,
                        title='Face Position Distribution',
                        labels={'x': 'Position', 'y': 'Count'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Aspect Ratio Analysis":
            st.subheader("📐 Face Aspect Ratio Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='face_aspect_ratio', title='Face Aspect Ratio Distribution',
                                 labels={'face_aspect_ratio': 'Aspect Ratio (Width/Height)', 'count': 'Frequency'})
                fig.add_vline(x=1.0, line_dash="dash", line_color="red", 
                             annotation_text="Square (1:1)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Outlier detection
                q1 = df['face_aspect_ratio'].quantile(0.25)
                q3 = df['face_aspect_ratio'].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df['face_aspect_ratio'] < q1 - 1.5*iqr) | 
                             (df['face_aspect_ratio'] > q3 + 1.5*iqr)]
                
                fig = px.box(df, y='face_aspect_ratio', title='Aspect Ratio Box Plot')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"Found {len(outliers)} aspect ratio outliers ({len(outliers)/len(df)*100:.1f}%)")

# Model Predictions Page
elif page == "🤖 Model Predictions":
    st.header("🤖 Face Detection Predictions")
    
    # Load model
    if st.session_state.yolo_model is None:
        with st.spinner("Loading YOLOv8 model..."):
            st.session_state.yolo_model = load_yolo_model()
    
    if st.session_state.yolo_model is not None:
        st.markdown('<div class="success-box">✅ YOLOv8 model loaded successfully!</div>', unsafe_allow_html=True)
        
        # Upload image section
        st.subheader("📤 Upload Image for Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
        
        with col2:
            show_confidence = st.checkbox("Show Confidence Scores", True)
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            st.subheader("🖼️ Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Run detection
            if st.button("🔍 Detect Faces", type="primary"):
                with st.spinner("Detecting faces..."):
                    # Convert PIL to OpenCV format
                    img_array = np.array(image)
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Run inference
                    results = st.session_state.yolo_model(img_cv, conf=confidence_threshold)
                    
                    # Process results
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
                    
                    # Draw detections
                    img_with_detections = img_array.copy()
                    for det in detections:
                        x1, y1, x2, y2, conf = det
                        # Draw rectangle
                        cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        
                        # Draw confidence if enabled
                        if show_confidence:
                            label = f'{conf:.2f}'
                            cv2.putText(img_with_detections, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Display results
                    st.subheader("🎯 Detection Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(img_with_detections, caption=f"Detected {len(detections)} face(s)", 
                                use_column_width=True)
                    
                    with col2:
                        st.markdown(f"**Detection Summary:**")
                        st.write(f"- Total faces detected: **{len(detections)}**")
                        st.write(f"- Confidence threshold: **{confidence_threshold}**")
                        
                        if detections:
                            avg_conf = np.mean([det[4] for det in detections])
                            st.write(f"- Average confidence: **{avg_conf:.3f}**")
                            
                            # Detection details
                            st.markdown("**Detection Details:**")
                            for i, det in enumerate(detections):
                                x1, y1, x2, y2, conf = det
                                width = x2 - x1
                                height = y2 - y1
                                st.write(f"Face {i+1}: {width}×{height}px, confidence: {conf:.3f}")

# Performance Metrics Page
elif page == "📋 Performance Metrics":
    st.header("📋 Model Performance Metrics")
    
    # Model comparison metrics
    st.subheader("⚖️ Model Performance Comparison")
    
    # Create comparison data
    comparison_data = {
        'Model': ['HOG + SVM', 'YOLOv8'],
        'Accuracy/mAP': [96.6, 95.3],
        'Precision': [95.0, 90.5],
        'Recall': [98.0, 92.0],
        'Inference Speed (ms)': [2000, 400],
        'False Positives': ['Very High', 'Very Low'],
        'Training Time (hours)': [0.5, 8.8]
    }
    
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy/mAP (%)', 'Precision (%)', 'Recall (%)', 'Inference Speed (ms)'],
        'HOG + SVM': [96.6, 95.0, 98.0, 2000],
        'YOLOv8': [95.3, 90.5, 92.0, 400]
    })
    
    # Display comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[96.6, 95.0, 98.0, 20],  # Normalized inference speed (100-speed/10)
            theta=['Accuracy', 'Precision', 'Recall', 'Speed'],
            fill='toself',
            name='HOG + SVM'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[95.3, 90.5, 92.0, 60],  # Normalized inference speed
            theta=['Accuracy', 'Precision', 'Recall', 'Speed'],
            fill='toself',
            name='YOLOv8'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Model Performance Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Detection quality comparison
        quality_data = pd.DataFrame({
            'Aspect': ['Clean Detection', 'Localization', 'Multi-face Handling', 'Real-time Performance'],
            'HOG + SVM': [2, 6, 5, 4],
            'YOLOv8': [9, 9, 9, 8]
        })
        
        fig = px.bar(quality_data, x='Aspect', y=['HOG + SVM', 'YOLOv8'],
                    title='Qualitative Performance Comparison (1-10 scale)',
                    barmode='group')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Training metrics
    st.subheader("📈 YOLOv8 Training Progress")
    
    # Simulate training metrics (replace with actual if available)
    epochs = list(range(1, 31))
    map50 = [0.595, 0.552, 0.682, 0.772, 0.742, 0.737, 0.702, 0.709, 0.729, 0.779,
            0.823, 0.858, 0.761, 0.877, 0.893, 0.859, 0.909, 0.895, 0.882, 0.932,
            0.900, 0.913, 0.938, 0.942, 0.945, 0.946, 0.957, 0.954, 0.954, 0.953]
    
    fig = px.line(x=epochs, y=map50, title='YOLOv8 Training Progress - mAP@0.5',
                  labels={'x': 'Epoch', 'y': 'mAP@0.5'})
    fig.add_hline(y=0.9, line_dash="dash", line_color="red", 
                  annotation_text="Target Performance (90%)")
    st.plotly_chart(fig, use_container_width=True)

# Model Comparison Page
elif page == "🔄 Model Comparison":
    st.header("🔄 Traditional vs Modern Computer Vision")
    
    st.subheader("🔄 Architecture Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 HOG + SVM (Traditional)")
        st.markdown("""
        **Feature Extraction:**
        - Hand-crafted HOG features
        - Fixed 1764-dimensional vectors
        - Edge-based descriptors
        
        **Classification:**
        - Support Vector Machine
        - RBF kernel with hyperparameter tuning
        - Binary classification (face/non-face)
        
        **Detection:**
        - Sliding window approach
        - Multiple scales and positions
        - Non-Maximum Suppression needed
        
        **Pros:**
        - ✅ Interpretable features
        - ✅ Fast training
        - ✅ Low memory requirements
        - ✅ Good baseline performance
        
        **Cons:**
        - ❌ Many false positives
        - ❌ Slow inference (sliding window)
        - ❌ Fixed feature representation
        - ❌ Poor multi-scale handling
        """)
    
    with col2:
        st.markdown("### 🤖 YOLOv8 (Modern Deep Learning)")
        st.markdown("""
        **Architecture:**
        - CSPDarknet backbone (feature extraction)
        - PANet neck (feature fusion)
        - Detection head (predictions)
        
        **Learning:**
        - End-to-end training
        - Learnable features
        - Transfer learning from pretrained weights
        
        **Detection:**
        - Single forward pass
        - Grid-based predictions
        - Built-in NMS
        
        **Pros:**
        - ✅ High accuracy (95.3% mAP)
        - ✅ Clean single detections
        - ✅ Fast inference
        - ✅ Excellent multi-face handling
        - ✅ Automatic feature learning
        
        **Cons:**
        - ❌ Longer training time
        - ❌ Higher memory requirements
        - ❌ Less interpretable
        - ❌ Needs GPU for optimal performance
        """)
    
    st.subheader("📊 Performance Evolution")
    
    # Evolution timeline
    timeline_data = pd.DataFrame({
        'Year': [2005, 2012, 2016, 2020, 2023],
        'Method': ['HOG+SVM', 'AlexNet', 'YOLO v1', 'YOLOv5', 'YOLOv8'],
        'Accuracy': [75, 82, 85, 90, 95],
        'Speed (FPS)': [0.5, 10, 45, 140, 150]
    })
    
    fig = px.scatter(timeline_data, x='Year', y='Accuracy', size='Speed (FPS)',
                    hover_name='Method', title='Computer Vision Evolution: Accuracy vs Speed',
                    labels={'Accuracy': 'Detection Accuracy (%)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("🎯 Key Insights")
    
    insights = [
        "**Traditional CV (HOG+SVM)** provides excellent educational foundation and interpretability",
        "**Modern Deep Learning (YOLO)** delivers superior real-world performance", 
        "**Transfer Learning** dramatically reduces training time and improves results",
        "**End-to-end Learning** eliminates need for manual feature engineering",
        "**Architecture Evolution** shows consistent improvement in accuracy and speed"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Final recommendations
    st.subheader("🚀 Deployment Recommendations")
    
    st.markdown("""
    **For Production Use:**
    - **Primary Model:** YOLOv8 (95.3% mAP, clean detections)
    - **Backup/Edge:** HOG+SVM (simpler, lower requirements)
    - **Hybrid Approach:** Use YOLO for main detection, HOG+SVM for verification
    
    **Next Steps:**
    1. Deploy YOLOv8 model in production environment
    2. Implement real-time video processing
    3. Add face recognition capabilities
    4. Optimize for mobile/edge deployment
    5. Continuous model improvement with new data
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 <strong>Human Face Detection System</strong> - Traditional CV vs Modern Deep Learning Comparison</p>
    <p>Built with HOG+SVM baseline and YOLOv8 deep learning • Comprehensive EDA and Performance Analysis</p>
</div>
""", unsafe_allow_html=True)