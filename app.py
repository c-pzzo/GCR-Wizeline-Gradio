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

def test_api_connection() -> str:
    """Test if API is reachable"""
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        return f"✅ API Status: {response.status_code} - {response.json()}"
    except Exception as e:
        return f"❌ API Connection Error: {str(e)}"

def call_api_get(endpoint: str) -> dict:
    """Call GET API endpoint"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API call failed: {str(e)}"}

def call_api_post(endpoint: str, data: dict) -> dict:
    """Call POST API endpoint with proper headers"""
    try:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.post(
            f"{API_URL}/{endpoint}", 
            json=data, 
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_detail = ""
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        return {"error": f"API call failed: {str(e)}", "detail": error_detail}

def get_model_info() -> dict:
    """Get current model information"""
    return call_api_get("model/info")

def predict_single(*features) -> Tuple[str, str]:
    """Make a single prediction"""
    # Create request payload exactly as FastAPI expects
    feature_dict = {f'feature_{i}': float(features[i]) for i in range(20)}
    
    print(f"Sending request: {feature_dict}")  # Debug
    
    # Call API
    result = call_api_post("predict", feature_dict)
    
    print(f"Received response: {result}")  # Debug
    
    if "error" in result:
        error_msg = f"❌ Error: {result['error']}"
        if "detail" in result:
            error_msg += f"\nDetail: {result['detail']}"
        return error_msg, ""
    
    # Format response
    try:
        predictions = result.get('predictions', [])
        if isinstance(predictions, list) and len(predictions) > 0:
            prediction = predictions[0]
        else:
            prediction = predictions
            
        model_info = result.get('model_info', {})
        
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
        
    except Exception as e:
        return f"❌ Error parsing response: {str(e)}\nRaw response: {result}", ""

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
        
        # Prepare data - convert to list of dictionaries
        instances = []
        for _, row in df.iterrows():
            instance = {f'feature_{i}': float(row[f'feature_{i}']) for i in range(20)}
            instances.append(instance)
        
        # Call API with the correct structure
        payload = {"instances": instances}
        result = call_api_post("predict/batch", payload)
        
        if "error" in result:
            error_msg = f"❌ Error: {result['error']}"
            if "detail" in result:
                error_msg += f"\nDetail: {result['detail']}"
            return error_msg, ""
        
        # Process results
        predictions = result.get('predictions', [])
        model_info = result.get('model_info', {})
        
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

# Create Gradio interface
with gr.Blocks(title="ML Model Prediction Interface") as interface:
    
    gr.Markdown("""
    # 🤖 ML Model Prediction Interface
    
    This interface allows you to make predictions using our deployed LightGBM model.
    """)
    
    # API Connection Test
    with gr.Row():
        test_button = gr.Button("🔧 Test API Connection", variant="secondary")
        api_status = gr.Textbox(label="API Status", interactive=False)
    
    test_button.click(fn=test_api_connection, outputs=[api_status])
    
    with gr.Tabs():
        # Single Prediction Tab
        with gr.Tab("🎯 Single Prediction"):
            gr.Markdown("### Enter values for all 20 features:")
            
            # Create input fields for all features
            feature_inputs = []
            with gr.Row():
                for i in range(10):
                    with gr.Column(scale=1):
                        inp = gr.Number(label=f"Feature {i}", value=0.0, precision=4)
                        feature_inputs.append(inp)
            
            with gr.Row():
                for i in range(10, 20):
                    with gr.Column(scale=1):
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
    
    gr.Markdown(f"""
    ---
    
    **API Endpoint:** `{API_URL}`
    """)

# Launch the interface
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        show_error=True
    )