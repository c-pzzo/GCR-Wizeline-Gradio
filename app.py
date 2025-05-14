import gradio as gr
import requests
import pandas as pd
import numpy as np
import json
import io
from typing import Dict, List, Tuple
import time

# Configuration
API_URL = "https://ml-prediction-service-239475924060.us-central1.run.app"
FEATURE_NAMES = [f'feature_{i}' for i in range(20)]

# Feature descriptions for better UX
FEATURE_DESCRIPTIONS = {
    'feature_0': 'Feature 0 (Numeric)',
    'feature_1': 'Feature 1 (Numeric)', 
    # Add more descriptive names if you know what each feature represents
}

def call_api(endpoint: str, data: dict) -> dict:
    """Call the ML API endpoint"""
    try:
        response = requests.post(f"{API_URL}/{endpoint}", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API call failed: {str(e)}"}

def get_model_info() -> dict:
    """Get current model information"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Could not fetch model info: {str(e)}"}

def predict_single(*features) -> Tuple[str, str]:
    """Make a single prediction"""
    # Create request payload
    feature_dict = {f'feature_{i}': features[i] for i in range(20)}
    
    # Call API
    result = call_api("predict", feature_dict)
    
    if "error" in result:
        return f"❌ Error: {result['error']}", ""
    
    # Format response
    prediction = result['predictions'][0]
    model_info = result['model_info']
    
    result_text = f"""
    🎯 **Prediction Result**
    
    **Predicted Target:** {prediction:.4f}
    
    **Model Information:**
    - Model Type: {model_info.get('model_type', 'Unknown')}
    - Version: {model_info.get('version', 'Unknown')}
    - Prediction Count: {model_info.get('prediction_count', 'Unknown')}
    """
    
    # Create feature values summary
    feature_summary = "**Input Features:**\n"
    for i, value in enumerate(features):
        feature_summary += f"- Feature {i}: {value}\n"
    
    return result_text, feature_summary

def predict_batch(file) -> Tuple[str, str]:
    """Make batch predictions from CSV file"""
    if file is None:
        return "❌ Please upload a CSV file", ""
    
    try:
        # Read the uploaded file
        df = pd.read_csv(file.name)
        
        # Verify required columns
        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_features:
            return f"❌ Missing features in CSV: {missing_features}", ""
        
        # Prepare data
        instances = df[FEATURE_NAMES].to_dict('records')
        
        # Call API
        result = call_api("predict/batch", {"instances": instances})
        
        if "error" in result:
            return f"❌ Error: {result['error']}", ""
        
        # Process results
        predictions = result['predictions']
        model_info = result['model_info']
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['predicted_target'] = predictions
        
        # Summary statistics
        pred_stats = {
            'count': len(predictions),
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions)
        }
        
        result_text = f"""
        🎯 **Batch Prediction Results**
        
        **Summary Statistics:**
        - Total Predictions: {pred_stats['count']}
        - Mean: {pred_stats['mean']:.4f}
        - Std Dev: {pred_stats['std']:.4f}
        - Min: {pred_stats['min']:.4f}
        - Max: {pred_stats['max']:.4f}
        
        **Model Information:**
        - Model Type: {model_info.get('model_type', 'Unknown')}
        - Version: {model_info.get('version', 'Unknown')}
        """
        
        # Create downloadable CSV
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        return result_text, csv_data
        
    except Exception as e:
        return f"❌ Error processing file: {str(e)}", ""

def show_model_info() -> str:
    """Display current model information"""
    info = get_model_info()
    
    if "error" in info:
        return f"❌ {info['error']}"
    
    info_text = f"""
    # 🤖 Current Model Information
    
    **Model Status:** {'✅ Loaded' if info.get('model_loaded', False) else '❌ Not Loaded'}
    
    **Model Details:**
    - Type: {info.get('model_type', 'Unknown')}
    - Version: {info.get('timestamp', 'Unknown')}
    - Feature Count: {info.get('feature_count', 'Unknown')}
    
    **Performance Metrics:**
    """
    
    if 'performance' in info:
        perf = info['performance']
        info_text += f"""
        - RMSE: {perf.get('rmse', 'N/A')}
        - MAE: {perf.get('mae', 'N/A')}
        - R²: {perf.get('r2', 'N/A')}
        - MAPE: {perf.get('mape', 'N/A')}%
        """
    else:
        info_text += "\n    *Performance metrics not available*"
    
    if 'feature_importance' in info and info['feature_importance']:
        info_text += "\n\n**Top 5 Important Features:**\n"
        # Sort features by importance
        features = sorted(info['feature_importance'], key=lambda x: x['importance'], reverse=True)[:5]
        for feat in features:
            info_text += f"    - {feat['feature']}: {feat['importance']:.4f}\n"
    
    return info_text

# Custom CSS for better styling
custom_css = """
#component-0 {
    max-width: 1200px;
    margin: 0 auto;
}

.prediction-result {
    font-size: 18px;
    font-weight: bold;
    color: #2563eb;
}

.error-message {
    color: #dc2626;
    font-weight: bold;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="ML Model Prediction Interface") as interface:
    
    gr.Markdown("""
    # 🤖 ML Model Prediction Interface
    
    This interface allows you to make predictions using our deployed LightGBM model.
    Choose between single predictions or batch processing from CSV files.
    """)
    
    with gr.Tabs():
        # Single Prediction Tab
        with gr.Tab("🎯 Single Prediction"):
            gr.Markdown("### Enter values for all 20 features:")
            
            # Create input fields for all features
            feature_inputs = []
            with gr.Row():
                for i in range(10):
                    with gr.Column():
                        inp = gr.Number(label=f"Feature {i}", value=0.0, precision=4)
                        feature_inputs.append(inp)
            
            with gr.Row():
                for i in range(10, 20):
                    with gr.Column():
                        inp = gr.Number(label=f"Feature {i}", value=0.0, precision=4)
                        feature_inputs.append(inp)
            
            predict_button = gr.Button("🔮 Make Prediction", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    prediction_result = gr.Markdown(label="Prediction Result")
                with gr.Column():
                    feature_summary = gr.Markdown(label="Feature Summary")
            
            predict_button.click(
                fn=predict_single,
                inputs=feature_inputs,
                outputs=[prediction_result, feature_summary]
            )
        
        # Batch Prediction Tab
        with gr.Tab("📊 Batch Prediction"):
            gr.Markdown("""
            ### Upload a CSV file with feature columns
            
            Your CSV should contain columns: `feature_0`, `feature_1`, ..., `feature_19`
            """)
            
            file_upload = gr.File(
                label="Upload CSV File",
                file_types=[".csv"],
                type="filepath"
            )
            
            batch_predict_button = gr.Button("📈 Predict Batch", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    batch_result = gr.Markdown(label="Batch Results")
                with gr.Column():
                    csv_download = gr.Textbox(
                        label="Results CSV (copy to save)",
                        lines=10,
                        max_lines=20
                    )
            
            batch_predict_button.click(
                fn=predict_batch,
                inputs=[file_upload],
                outputs=[batch_result, csv_download]
            )
        
        # Model Information Tab
        with gr.Tab("ℹ️ Model Information"):
            gr.Markdown("### Current Model Status and Performance")
            
            info_button = gr.Button("🔄 Refresh Model Info", variant="secondary")
            model_info_display = gr.Markdown()
            
            info_button.click(
                fn=show_model_info,
                outputs=[model_info_display]
            )
            
            # Load model info on startup
            interface.load(fn=show_model_info, outputs=[model_info_display])
    
    gr.Markdown("""
    ---
    
    **API Endpoint:** `{}`
    
    **Features:** Single predictions, batch processing, model information
    """.format(API_URL))

# Launch the interface
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        show_error=True
    )