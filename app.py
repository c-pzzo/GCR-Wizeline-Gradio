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
        result = response.json()
        status = "✅ Connected" if response.status_code == 200 else f"⚠️ Status {response.status_code}"
        return f"{status}\n{json.dumps(result, indent=2)}"
    except Exception as e:
        return f"❌ Connection Failed: {str(e)}"

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

def call_api_post_file(endpoint: str, files: dict) -> dict:
    """Call POST API endpoint with file upload"""
    try:
        response = requests.post(f"{API_URL}/{endpoint}", files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"File upload failed: {str(e)}"}

def get_model_info() -> dict:
    """Get current model information"""
    return call_api_get("model/info")

def predict_single(*features) -> Tuple[str, str]:
    """Make a single prediction with detailed output"""
    # Create request payload exactly as FastAPI expects
    feature_dict = {f'feature_{i}': float(features[i]) for i in range(20)}
    
    # Call API
    result = call_api_post("predict", feature_dict)
    
    if "error" in result:
        error_msg = f"❌ **Prediction Error**\n\n{result['error']}"
        if "detail" in result:
            error_msg += f"\n\n**Details:** {result['detail']}"
        return error_msg, ""
    
    # Format detailed response
    try:
        predictions = result.get('predictions', [])
        prediction = predictions[0] if isinstance(predictions, list) else predictions
        model_info = result.get('model_info', {})
        
        result_text = f"""# 🎯 **Prediction Result**

## **Predicted Target:** `{prediction:.6f}`

## **Model Information:**
- **Type:** {model_info.get('model_type', 'Unknown')}
- **Version:** {model_info.get('version', 'Unknown')}
- **Predictions Made:** {model_info.get('prediction_count', 1)}

## **Performance Metrics:**"""
        
        if 'performance' in model_info and model_info['performance']:
            perf = model_info['performance']
            result_text += f"""
- **RMSE:** {perf.get('rmse', 'N/A')}
- **MAE:** {perf.get('mae', 'N/A')}
- **R²:** {perf.get('r2', 'N/A')}
- **MAPE:** {perf.get('mape', 'N/A')}%"""
        else:
            result_text += "\n*Not available in response*"
        
        # Create detailed feature summary
        feature_summary = "# 📊 **Input Features Summary**\n\n"
        feature_summary += "| Feature | Value |\n|---------|-------|\n"
        for i, value in enumerate(features):
            feature_summary += f"| feature_{i} | {value} |\n"
        
        return result_text, feature_summary
        
    except Exception as e:
        return f"❌ Error parsing response: {str(e)}\n\n**Raw Response:**\n```json\n{json.dumps(result, indent=2)}\n```", ""

def predict_batch(file) -> Tuple[str, str, str]:
    """Make batch predictions from CSV file with detailed output"""
    if file is None:
        return "❌ Please upload a CSV file", "", ""
    
    try:
        # Read the uploaded file
        df = pd.read_csv(file.name)
        
        # Verify required columns
        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_features:
            return f"❌ Missing features in CSV: {missing_features}", "", ""
        
        # Show file preview
        file_preview = f"""# 📄 **Uploaded File Preview**

**Shape:** {df.shape[0]} rows × {df.shape[1]} columns

**First 5 rows:**
{df.head().to_string()}

**Columns:** {', '.join(df.columns.tolist())}
"""
        
        # Prepare data for API call
        instances = []
        for _, row in df.iterrows():
            instance = {f'feature_{i}': float(row[f'feature_{i}']) for i in range(20)}
            instances.append(instance)
        
        # Call batch prediction API
        payload = {"instances": instances}
        result = call_api_post("predict/batch", payload)
        
        if "error" in result:
            error_msg = f"❌ **Batch Prediction Error**\n\n{result['error']}"
            if "detail" in result:
                error_msg += f"\n\n**Details:** {result['detail']}"
            return file_preview, error_msg, ""
        
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
            'max': np.max(predictions),
            'median': np.median(predictions),
            'q25': np.percentile(predictions, 25),
            'q75': np.percentile(predictions, 75)
        }
        
        # Detailed results
        result_text = f"""# 🎯 **Batch Prediction Results**

## **Summary Statistics:**
- **Total Predictions:** {pred_stats['count']}
- **Mean:** {pred_stats['mean']:.6f}
- **Median:** {pred_stats['median']:.6f}
- **Std Dev:** {pred_stats['std']:.6f}
- **Min:** {pred_stats['min']:.6f}
- **Max:** {pred_stats['max']:.6f}
- **Q25:** {pred_stats['q25']:.6f}
- **Q75:** {pred_stats['q75']:.6f}

## **Model Information:**
- **Type:** {model_info.get('model_type', 'Unknown')}
- **Version:** {model_info.get('version', 'Unknown')}

## **Performance Metrics:**"""
        
        if 'performance' in model_info and model_info['performance']:
            perf = model_info['performance']
            result_text += f"""
- **RMSE:** {perf.get('rmse', 'N/A')}
- **MAE:** {perf.get('mae', 'N/A')}
- **R²:** {perf.get('r2', 'N/A')}
- **MAPE:** {perf.get('mape', 'N/A')}%"""
        else:
            result_text += "\n*Not available*"
        
        # Create downloadable CSV
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        return file_preview, result_text, csv_data
        
    except Exception as e:
        return file_preview if 'file_preview' in locals() else "", f"❌ Error processing file: {str(e)}", ""

def predict_csv_direct(file) -> Tuple[str, str]:
    """Use the direct CSV endpoint"""
    if file is None:
        return "❌ Please upload a CSV file", ""
    
    try:
        # Use the CSV endpoint directly
        with open(file.name, 'rb') as f:
            files = {'file': f}
            result = call_api_post_file("predict/csv", files)
        
        if "error" in result:
            return f"❌ **CSV Prediction Error**\n\n{result['error']}", ""
        
        # Format results
        predictions = result.get('predictions', [])
        model_info = result.get('model_info', {})
        row_count = result.get('row_count', len(predictions))
        
        # Create summary
        summary_text = f"""# 🎯 **CSV Prediction Results**

## **Summary:**
- **Rows Processed:** {row_count}
- **Predictions Made:** {len(predictions)}

## **Model:** {model_info.get('model_type', 'Unknown')} v{model_info.get('version', 'Unknown')}

## **Sample Predictions:**
"""
        
        # Show first 10 predictions
        for i, pred in enumerate(predictions[:10]):
            summary_text += f"- Row {i+1}: {pred:.6f}\n"
        
        if len(predictions) > 10:
            summary_text += f"- ... and {len(predictions) - 10} more\n"
        
        # Create downloadable results
        results_csv = "row_id,predicted_target\n"
        for i, pred in enumerate(predictions):
            results_csv += f"{i},{pred}\n"
        
        return summary_text, results_csv
        
    except Exception as e:
        return f"❌ Error with CSV endpoint: {str(e)}", ""

def show_model_info() -> str:
    """Display detailed current model information"""
    info = get_model_info()
    
    if "error" in info:
        return f"❌ **Error Getting Model Info**\n\n{info['error']}"
    
    info_text = f"""# 🤖 **Current Model Information**

## **Status:** {'✅ Loaded' if info.get('model_loaded', False) else '❌ Not Loaded'}

## **Model Details:**
- **Type:** {info.get('model_type', 'Unknown')}
- **Version:** {info.get('version', 'Unknown')}
- **Feature Count:** {info.get('feature_count', 'Unknown')}

## **Performance Metrics:**"""
    
    if 'performance' in info and info['performance']:
        perf = info['performance']
        info_text += f"""
- **RMSE:** {perf.get('rmse', 'N/A')}
- **MAE:** {perf.get('mae', 'N/A')}
- **R²:** {perf.get('r2', 'N/A')}
- **MAPE:** {perf.get('mape', 'N/A')}%"""
    else:
        info_text += "\n*Performance metrics not available*"
    
    if 'feature_importance' in info and info['feature_importance']:
        info_text += "\n\n## **Top 10 Important Features:**\n"
        # Handle both list and dict formats
        if isinstance(info['feature_importance'], list):
            features = sorted(info['feature_importance'], key=lambda x: x['importance'], reverse=True)[:10]
        else:
            features = sorted(info['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10]
            features = [{'feature': f, 'importance': i} for f, i in features]
        
        for feat in features:
            feature_name = feat.get('feature', feat.get('feature_name', 'Unknown'))
            importance = feat.get('importance', 0)
            info_text += f"- **{feature_name}:** {importance:.6f}\n"
    
    if 'all_model_comparison' in info:
        info_text += "\n\n## **All Models Comparison:**\n"
        comparison = info['all_model_comparison']
        info_text += "| Model | RMSE | R² | MAE | MAPE |\n"
        info_text += "|-------|------|----|----- |------|\n"
        for model_name, metrics in comparison.items():
            info_text += f"| {model_name} | {metrics.get('rmse', 'N/A')} | {metrics.get('r2', 'N/A')} | {metrics.get('mae', 'N/A')} | {metrics.get('mape', 'N/A')} |\n"
    
    return info_text

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 1200px;
    margin: 0 auto;
}

.tab-content {
    padding: 20px;
}

.success {
    color: #22c55e;
    font-weight: bold;
}

.error {
    color: #ef4444;
    font-weight: bold;
}

.highlight {
    background: #f0f9ff;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #0369a1;
}
"""

# Create the complete Gradio interface
with gr.Blocks(css=custom_css, title="ML Model Prediction Interface - Complete") as interface:
    
    gr.Markdown("""
    # 🤖 ML Model Prediction Interface - Complete Edition
    
    Comprehensive interface for ML model predictions with detailed outputs and multiple prediction methods.
    """)
    
    # API Connection Test at the top
    with gr.Row():
        with gr.Column(scale=1):
            test_api_btn = gr.Button("🔧 Test API Connection", variant="secondary")
        with gr.Column(scale=3):
            api_status = gr.Code(label="API Status", language="json")
    
    test_api_btn.click(fn=test_api_connection, outputs=[api_status])
    
    with gr.Tabs():
        # Single Prediction Tab
        with gr.Tab("🎯 Single Prediction"):
            gr.Markdown("### Enter values for all 20 features:")
            
            # Create input fields for all features in a more compact layout
            feature_inputs = []
            with gr.Row():
                for i in range(5):
                    with gr.Column(scale=1):
                        for j in range(4):
                            idx = i * 4 + j
                            if idx < 20:
                                inp = gr.Number(
                                    label=f"Feature {idx}", 
                                    value=0.0, 
                                    precision=6,
                                    step=0.000001
                                )
                                feature_inputs.append(inp)
            
            predict_btn = gr.Button("🔮 Make Prediction", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column(scale=1):
                    prediction_result = gr.Markdown(label="Prediction Result")
                with gr.Column(scale=1):
                    feature_summary = gr.Markdown(label="Input Features")
            
            predict_btn.click(
                fn=predict_single,
                inputs=feature_inputs,
                outputs=[prediction_result, feature_summary]
            )
        
        # Batch Prediction Tab (CSV Upload Only)
        with gr.Tab("📊 Batch Prediction"):
            gr.Markdown("### Upload a CSV file for batch predictions")
            gr.Markdown("Your CSV should contain columns: `feature_0`, `feature_1`, ..., `feature_19`")
            
            file_upload_csv = gr.File(
                label="Upload CSV File",
                file_types=[".csv"],
                type="filepath"
            )
            csv_predict_btn = gr.Button("📄 Process CSV", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    csv_result = gr.Markdown(label="CSV Results")
                with gr.Column():
                    csv_download_direct = gr.Code(label="Results CSV", lines=15)
            
            csv_predict_btn.click(
                fn=predict_csv_direct,
                inputs=[file_upload_csv],
                outputs=[csv_result, csv_download_direct]
            )
        # Model Information Tab
        with gr.Tab("ℹ️ Model Information"):
            gr.Markdown("### Detailed Model Status and Performance")
            
            info_btn = gr.Button("🔄 Refresh Model Info", variant="secondary")
            model_info_display = gr.Markdown()
            
            info_btn.click(
                fn=show_model_info,
                outputs=[model_info_display]
            )
            
            # Load model info on startup
            interface.load(fn=show_model_info, outputs=[model_info_display])
    
    # Footer with API information
    gr.Markdown(f"""
    ---
    
    ### 📡 **API Information**
    
    **Endpoint:** `{API_URL}`
    
    **Available Methods:**
    - **Single Prediction:** `/predict` - Real-time single sample prediction
    - **Batch Prediction:** `/predict/batch` - Multiple samples in JSON format  
    - **CSV Upload:** `/predict/csv` - Direct CSV file processing
    - **Model Info:** `/model/info` - Detailed model information
    - **Health Check:** `/health` - Service status
    
    **Features:**
    - ✅ Detailed prediction results with model metadata
    - ✅ Multiple batch processing options
    - ✅ Comprehensive model information
    - ✅ Real-time API connection testing
    - ✅ Downloadable prediction results
    """)

# Launch the interface
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        show_error=True
    )
