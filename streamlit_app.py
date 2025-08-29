import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import numpy as np
import shap
from pipeline_orchestrator import MLPipelineOrchestrator
from chatbot import ChatbotHelper

# Streamlit UI
st.title("ML Pipeline App")

# Tabs for Training and Prediction
tab1, tab2 = st.tabs(["Train Pipeline", "Make Predictions"])

with tab1:
    # File upload
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'], key="train_uploader")

    if uploaded_file:
        # Save temp file
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load data for preview
        data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        st.dataframe(data.head())

        # Task type selection
        task_type = st.selectbox("Select task type", ['regression', 'classification'])

        # Detect task_type change and reset session state for a fresh start
        if 'previous_task_type' not in st.session_state:
            st.session_state['previous_task_type'] = task_type
        elif st.session_state['previous_task_type'] != task_type:
            # Reset relevant session states to start a new "session"
            keys_to_reset = ['suggestions', 'target_column', 'drop_columns', 'encoding_methods', 'selected_models']
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['previous_task_type'] = task_type
            st.rerun()  # Force rerun to refresh the UI after reset

        # Step 1: Target and drops selection (before suggestions/encoding)
        default_target = st.session_state.get('target_column', data.columns[0])
        target_column = st.selectbox("Select target column", data.columns.tolist(), index=data.columns.tolist().index(default_target))
        st.session_state['target_column'] = target_column

        available_cols_for_drop = [col for col in data.columns if col != target_column]
        default_drops = st.session_state.get('drop_columns', [])
        drop_columns = st.multiselect("Select columns to drop", available_cols_for_drop, default=default_drops)
        st.session_state['drop_columns'] = drop_columns

        # Remaining columns after hypothetical drops (for UI only; actual drops happen in cleaning)
        remaining_cols = [col for col in data.columns if col not in drop_columns and col != target_column]

        # Button to get suggestions (invokes LLM after task_type/target/drops)
        if st.button("Get Precomputed Suggestions"):
            # Init chatbot with full data and task_type
            chatbot = ChatbotHelper(data, task_type)
            suggestions = chatbot.precompute_suggestions()
            st.session_state['suggestions'] = suggestions  # Store in session for display

        # Display suggestions if available
        if 'suggestions' in st.session_state:
            suggestions = st.session_state['suggestions']
            st.write(suggestions['target'])
            st.write(suggestions['drops'])
            st.write(suggestions['encoding'])
            st.write(suggestions['other'])  # Display any other tips

        # Step 2: Encoding for remaining categorical columns
        remaining_data = data[remaining_cols + [target_column]]  # Simulate post-drop data for cat detection
        categorical_cols = remaining_data.select_dtypes(include=['object', 'category']).columns.tolist()
        encoding_methods = {}
        for col in categorical_cols:
            if col == target_column:
                continue  # Skip encoding for target
            method = st.selectbox(f"Encoding for {col}", ['onehot', 'label', 'none'])
            if method != 'none':
                encoding_methods[col] = method
        st.session_state['encoding_methods'] = encoding_methods

        # Model selection
        if task_type == 'regression':
            all_models = ['LinearRegression', 'Ridge', 'RandomForest', 'XGBoost']
        else:
            all_models = ['LogisticRegression', 'RandomForest', 'XGBoost']
        selected_models = st.multiselect("Select models to train", all_models, default=all_models)
        st.session_state['selected_models'] = selected_models

        # Param grids (simple example, filtered by selected)
        param_grids = {
            'RandomForest': {'n_estimators': [50, 100]},
            'XGBoost': {'n_estimators': [50, 100]}
        }
        param_grids = {k: v for k, v in param_grids.items() if k in selected_models}

        if st.button("Run Pipeline"):
            target_column = st.session_state.get('target_column', data.columns[0])
            drop_columns = st.session_state.get('drop_columns', [])
            encoding_methods = st.session_state.get('encoding_methods', {})
            selected_models = st.session_state.get('selected_models', [])
            orchestrator = MLPipelineOrchestrator(
                file_path, target_column, task_type,
                drop_columns=drop_columns, encoding_methods=encoding_methods,
                param_grids=param_grids, selected_models=selected_models
            )
            results = orchestrator.run_pipeline()
            
            # Save the list of selected models in metadata
            metadata_path = 'models/metadata.json'
            try:
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Add selected models to metadata
                    metadata['selected_models'] = selected_models
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f)
            except Exception as e:
                st.warning(f"Could not update metadata with selected models: {str(e)}")
            
            # Save a copy of the processed data for SHAP analysis later
            try:
                # Save a sample of the processed data (to avoid large files)
                sample_data = data.sample(min(500, len(data))) if len(data) > 500 else data
                
                # Save with task type and include metadata in filename to avoid conflicts
                temp_file_path = f"temp_{task_type}.csv"
                sample_data.to_csv(temp_file_path, index=False)
                st.info(f"Saved training data sample to {temp_file_path} for future SHAP analysis")
            except Exception as e:
                st.warning(f"Could not save training data for SHAP: {str(e)}")

            st.subheader("EDA Stats")
            st.json(results['eda_stats'])

            st.subheader("Model Evaluations")
            st.json(results['evaluations'])

            # Combined interactive graph for feature importances/coefficients
            if results['importances']:
                st.subheader("Feature Importances/Coefficients Graph")
                all_imp = []
                for name, df in results['importances'].items():
                    df = df.copy()
                    df['Model'] = name
                    if 'Importance' in df.columns:
                        df = df.rename(columns={'Importance': 'Value'})
                    elif 'Coefficient' in df.columns:
                        df = df.rename(columns={'Coefficient': 'Value'})
                    all_imp.append(df)
                combined = pd.concat(all_imp)
                fig = px.bar(combined, x='Feature', y='Value', color='Model', barmode='group',
                             title='Feature Importances/Coefficients Across Models')
                st.plotly_chart(fig, use_container_width=True)

            # Plot predicted vs actual with Plotly
            if results['predictions'] and task_type == 'regression':
                st.subheader("Predicted vs Actual Plots")
                y_test = results['predictions']['y_test']
                for name in results['models'].keys():
                    y_pred = results['predictions'][name]
                    df_plot = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                    fig = px.scatter(df_plot, x='Actual', y='Predicted', title=f'{name}: Predicted vs Actual',
                                     trendline='ols')  # Add regression line
                    # Add diagonal line
                    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                    fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                  line=dict(color='red', dash='dash'))
                    st.plotly_chart(fig, use_container_width=True)
            elif results['predictions'] and task_type == 'classification':
                st.write("Classification plots (e.g., confusion matrix) can be added if needed.")

    # Chatbot interface in sidebar (shared across tabs)
    with st.sidebar:
        st.subheader("Chatbot Helper")
        user_query = st.text_input("Ask for data analysis or preprocessing guidance (e.g., 'suggest handling outliers', 'EDA insights')")
        if user_query and task_type:  # Ensure task_type is selected
            chatbot = ChatbotHelper(data, task_type)  # Re-init if needed
            response = chatbot.get_guidance(user_query)
            st.write(response)

with tab2:
    st.subheader("Make Predictions")

    # Load metadata
    metadata_path = 'models/metadata.json'
    if os.path.exists(metadata_path) and os.path.getsize(metadata_path) > 0:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            st.warning(f"Metadata file corrupted or incomplete: {str(e)}. Re-run the pipeline to regenerate.")
            st.stop()
    else:
        st.warning("Metadata file not found or empty. Run the pipeline first in the Train tab to generate files.")
        st.stop()

    # List available models (only those that were selected in training)
    model_dir = 'models/'
    
    # Get list of all model files in the directory
    all_model_files = [f.replace('.joblib', '') for f in os.listdir(model_dir) 
                     if f.endswith('.joblib') and f != 'preprocessor.joblib']
    
    # Filter by selected models from metadata if available
    if 'selected_models' in metadata:
        # Only show models that were explicitly selected during training
        available_models = [model for model in all_model_files 
                          if model in metadata['selected_models']]
    else:
        # Fallback to showing all models if metadata doesn't have selection info
        available_models = all_model_files
    
    if not available_models:
        st.warning("No models found. Train and save models first in the Train tab.")
        st.stop()

    # Select model
    selected_model = st.selectbox("Select model for prediction", available_models)

    # Load preprocessor and model
    preprocessor_path = 'models/preprocessor.joblib'
    model_path = f'models/{selected_model}.joblib'
    if os.path.exists(preprocessor_path) and os.path.getsize(preprocessor_path) > 0:
        try:
            preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            st.warning(f"Error loading preprocessor (may be incomplete save): {str(e)}. Re-run the pipeline.")
            st.stop()
    else:
        st.warning("Preprocessor file not found or empty. Run the pipeline first.")
        st.stop()

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.warning(f"Error loading model (may be incomplete save): {str(e)}. Re-run the pipeline.")
            st.stop()
    else:
        st.warning(f"Selected model file not found or empty.")
        st.stop()

    # Input form for features
    st.subheader("Enter input values")
    input_data = {}
    for feature in metadata['features']:
        if feature in metadata['categorical_cols']:
            options = metadata['categories'].get(feature, [])
            input_data[feature] = st.selectbox(feature, options=options, index=0 if options else None)
        else:
            input_data[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict"):
        # Show status indicator
        with st.spinner('Processing input data and generating prediction...'):
            # Create DataFrame from inputs
            input_df = pd.DataFrame([input_data])
            
            # Apply preprocessor
            input_transformed = preprocessor.transform(input_df)
            
            # Get transformed feature names and convert to DataFrame (fixes the warning)
            transformed_columns = preprocessor.get_feature_names_out()
            input_transformed_df = pd.DataFrame(input_transformed, columns=transformed_columns)
            
            # Predict
            prediction = model.predict(input_transformed_df)
            
            # Display prediction
            st.subheader("Prediction")
            
            # Format prediction based on task type
            if metadata.get('task_type') == 'classification':
                st.success(f"Predicted class: {prediction[0]}")
            else:  # regression
                st.success(f"Predicted value: {prediction[0]:.4f}")
                
            # Display the original input features for reference
            st.subheader("Input Features")
            st.dataframe(input_df)
        
        # SHAP Feature Importance Section
        st.subheader("SHAP Feature Importance")
        
        with st.expander("ℹ️ What are SHAP values?", expanded=False):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** values explain how each feature contributes to the prediction:
            
            - **Positive values** (red) indicate features that push the prediction **higher**
            - **Negative values** (blue) indicate features that push the prediction **lower**
            - The **magnitude** of the value shows how strong the impact is
            
            SHAP values help you understand which features are most important for a specific prediction and how they influence the model's output.
            """)
        
        # Calculate SHAP values
        # Create placeholder for progress updates
        progress_text = st.empty()
        
        # Determine the task type from metadata or available models
        task_type = None
        if 'task_type' in metadata:
            task_type = metadata['task_type']
        else:
            # Infer from available models
            task_type = 'regression' if any(m in available_models for m in ['LinearRegression', 'Ridge']) else 'classification'
        
        # Check for temp files with training data
        train_file_path = None
        if os.path.exists(f"temp_{task_type}.csv"):
            train_file_path = f"temp_{task_type}.csv"
        
        # For most accurate SHAP values, we need some background data
        # First try to load the original dataset if available
        try:
            # Start with the input as the default background data
            bg_transformed_df = input_transformed_df
            
            # Try to load background data if available
            if train_file_path and os.path.exists(train_file_path):
                bg_data = pd.read_csv(train_file_path)
                
                # Identify target column from metadata or common names
                target_col = metadata.get('target_column')
                if not target_col:
                    # Try common target column names
                    target_candidates = ['target', 'y', 'label', 'price', 'class']
                    for col in bg_data.columns:
                        if col in target_candidates or col.lower() in target_candidates:
                            target_col = col
                            break
                
                # Remove target column if found
                if target_col and target_col in bg_data.columns:
                    bg_data = bg_data.drop(columns=[target_col])
                
                # Filter to only include feature columns that exist in metadata
                feature_cols = metadata.get('features', [])
                if feature_cols:
                    # Keep only columns that are in both bg_data and feature_cols
                    valid_cols = [col for col in bg_data.columns if col in feature_cols]
                    # Only filter if we have valid columns
                    if valid_cols:
                        bg_data = bg_data[valid_cols]
                
                # Make sure all expected features from metadata are in the data
                for feature in metadata.get('features', []):
                    if feature not in bg_data.columns:
                        # This means we're missing a feature - set it to a default value
                        if feature in metadata.get('categorical_cols', []):
                            # For categorical features, use the first category
                            categories = metadata.get('categories', {}).get(feature, ['unknown'])
                            bg_data[feature] = categories[0]
                        else:
                            # For numerical features, use 0
                            bg_data[feature] = 0
                
                # If we have data after all this processing, transform it
                if not bg_data.empty and all(col in bg_data.columns for col in metadata.get('features', [])):
                    # Sample to reduce computation time
                    bg_sample = bg_data.head(100)
                    bg_transformed = preprocessor.transform(bg_sample)
                    bg_transformed_df = pd.DataFrame(bg_transformed, columns=transformed_columns)
        except Exception as e:
            st.warning(f"Could not load background data for SHAP: {str(e)}. Using input data as background.")
            # Fallback to using input data as background
            bg_transformed_df = input_transformed_df
        except Exception as e:
            st.warning(f"Could not load background data for SHAP: {str(e)}. Using input data as background.")
            bg_transformed_df = input_transformed_df
        
        try:
            # Create a status container for SHAP calculation status
            status_container = st.empty()
            status_container.info("Calculating SHAP values... This might take a moment.")
            
            # We'll try different explainers in order of preference based on model type
            explainer = None
            
            try:
                if selected_model in ['LinearRegression', 'Ridge', 'LogisticRegression']:
                    # Linear models - try LinearExplainer first
                    explainer = shap.LinearExplainer(model, bg_transformed_df)
                
                elif selected_model in ['RandomForest', 'XGBoost']:
                    # Tree-based models - try TreeExplainer
                    explainer = shap.TreeExplainer(model)
                
                else:
                    # For other models, use the general Explainer
                    explainer = shap.Explainer(model, bg_transformed_df)
                    
            except Exception as e1:
                st.warning(f"First attempt at SHAP explainer failed: {str(e1)}. Trying alternative approach.")
                
                try:
                    # Fallback to KernelExplainer which works with any model
                    explainer = shap.KernelExplainer(model.predict, bg_transformed_df)
                except Exception as e2:
                    st.warning(f"Second attempt failed: {str(e2)}. Using general Explainer.")
                    
                    try:
                        # Last resort - generic Explainer
                        explainer = shap.Explainer(model)
                    except Exception as e3:
                        st.error(f"Could not create SHAP explainer: {str(e3)}")
                        raise Exception("Failed to create SHAP explainer after multiple attempts")
            
                # Calculate SHAP values for the input
            try:
                progress_text.text("Computing SHAP values for your input...")
                shap_values = explainer.shap_values(input_transformed_df)
                
                # Handle different return types from different explainers
                if isinstance(shap_values, list):
                    # For multi-class models, take the predicted class or first class
                    if task_type == 'classification':
                        pred_class = int(prediction[0])
                        if pred_class < len(shap_values) and pred_class >= 0:
                            shap_values = shap_values[pred_class]
                        else:
                            shap_values = shap_values[0]  # Default to first class
                    else:
                        shap_values = shap_values[0]  # For regression
                        
                # For newer SHAP versions that return an Explanation object
                elif hasattr(shap_values, "values"):
                    shap_values = shap_values.values
            except Exception as e:
                st.warning(f"Error in calculating SHAP values: {str(e)}")
                # Try alternative approach
                try:
                    # For models where shap_values() doesn't work, try __call__
                    shap_values = explainer(input_transformed_df).values
                except Exception as e2:
                    st.error(f"Failed to calculate SHAP values: {str(e2)}")
                    raise Exception("Could not calculate SHAP values")
            
            # Clear status messages
            status_container.empty()
            progress_text.empty()
            
            # Add success message
            st.success("SHAP values calculated successfully!")
            
            # Create a DataFrame to display the feature contributions
            feature_names = transformed_columns
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_values[0]
            })
            
            # Sort by absolute SHAP value for better visualization
            shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values('Abs SHAP', ascending=False).drop('Abs SHAP', axis=1)
            
            # Clean up feature names for display
            def clean_feature_name(name):
                # Handle one-hot encoded features with different prefixes
                onehot_prefixes = ['onehot_auto__', 'onehot_user__', 'onehot__']
                
                for prefix in onehot_prefixes:
                    if name.startswith(prefix):
                        parts = name.split('__')
                        if len(parts) > 1:
                            # Extract feature and value, handling cases where value might contain underscores
                            feature_value_part = parts[1]
                            
                            # Handle special case for 'formerly smoked', 'never smoked', etc.
                            if 'smoking_status' in feature_value_part:
                                if 'formerly smoked' in feature_value_part:
                                    feature = 'smoking_status'
                                    value = 'formerly smoked'
                                elif 'never smoked' in feature_value_part:
                                    feature = 'smoking_status'
                                    value = 'never smoked'
                                else:
                                    # Normal processing for other smoking statuses
                                    feature_and_value = feature_value_part.split('_')
                                    feature = '_'.join(feature_and_value[:-1])
                                    value = feature_and_value[-1]
                            else:
                                # Normal case - split at the last underscore
                                last_underscore = feature_value_part.rfind('_')
                                if last_underscore > 0:
                                    feature = feature_value_part[:last_underscore]
                                    value = feature_value_part[last_underscore+1:]
                                else:
                                    # Fallback if no underscore found
                                    feature = feature_value_part
                                    value = "Unknown"
                            
                            # Check if this is the selected value for this feature
                            if feature in input_data:
                                # Convert both to string for comparison and handle special cases
                                input_value = str(input_data[feature])
                                if input_value.lower() == value.lower():
                                    return feature  # Return just the feature name without the value
                                else:
                                    return None  # Not the selected value, skip it
                            
                            # If we can't determine if it's selected, just show the feature name
                            return feature
                
                # Handle numeric features with different prefixes
                num_prefixes = ['num__', 'num_']
                for prefix in num_prefixes:
                    if name.startswith(prefix):
                        feature = name.replace(prefix, '')
                        return feature
                
                # For boolean features
                if name.startswith('bool__'):
                    feature = name.replace('bool__', '')
                    return feature
                
                # For any other feature
                return name

            # Create a dictionary to track and combine contributions for each feature
            feature_contributions = {}
            feature_mapping = {}  # To store original to cleaned name mapping
            
            # First pass: clean names and track which features to include
            for _, row in shap_df.iterrows():
                original_name = row['Feature']
                clean_name = clean_feature_name(original_name)
                
                if clean_name is not None:  # Skip None values (unselected categorical values)
                    # Track mapping from original to cleaned names
                    feature_mapping[original_name] = clean_name
                    
                    # Initialize or update the contribution for this feature
                    if clean_name not in feature_contributions:
                        feature_contributions[clean_name] = row['SHAP Value']
                    else:
                        # For one-hot encoded features, we might need to combine contributions
                        # This happens when we have multiple columns for one feature
                        feature_contributions[clean_name] += row['SHAP Value']
            
            # Second pass: create filtered dataframe with combined values
            filtered_rows = []
            for feature, value in feature_contributions.items():
                filtered_rows.append({
                    'Feature': feature,
                    'SHAP Value': value
                })

            # Create new filtered dataframe
            filtered_shap_df = pd.DataFrame(filtered_rows)
            
            # Sort by absolute SHAP value to show most important features at top
            filtered_shap_df['Abs_SHAP'] = filtered_shap_df['SHAP Value'].abs()
            filtered_shap_df = filtered_shap_df.sort_values('Abs_SHAP', ascending=False)
            filtered_shap_df = filtered_shap_df.drop('Abs_SHAP', axis=1)
            
            # Display SHAP values table with clean feature names
            st.write("Feature Contributions to Prediction:")
            st.dataframe(filtered_shap_df)
            
            # We'll skip this section as we'll create a better Plotly visualization below
            
            # Add visualization of SHAP values
            st.write("SHAP Feature Contributions:")
            
            # Create a Plotly horizontal bar chart showing feature contributions
            filtered_shap_df = filtered_shap_df.sort_values('SHAP Value')  # Sort for better visualization
            
            # Create color scale based on SHAP values
            colors = ['red' if x > 0 else 'blue' for x in filtered_shap_df['SHAP Value']]
            
            fig = go.Figure(go.Bar(
                y=filtered_shap_df['Feature'],
                x=filtered_shap_df['SHAP Value'],
                orientation='h',
                marker_color=colors
            ))
            
            fig.update_layout(
                title='Feature Contribution to Prediction',
                xaxis_title='SHAP Value (Impact on Prediction)',
                yaxis_title='Feature',
                height=500,
                width=700
            )
            
            # Add a vertical line at x=0 to help visualize positive vs negative impact
            fig.add_shape(
                type="line",
                x0=0, x1=0, y0=-0.5, y1=len(filtered_shap_df)-0.5,
                line=dict(color="black", width=1, dash="dash")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            # Clear any status indicators
            if 'status_container' in locals():
                status_container.empty()
            if 'progress_text' in locals():
                progress_text.empty()
                
            st.error(f"Could not generate SHAP values: {str(e)}")
            st.info("SHAP values calculation might not be supported for this model type or configuration.")
            
            # Provide simple feature importance as fallback
            try:
                st.subheader("Feature Importance (Built-in)")
                if hasattr(model, "feature_importances_"):
                    # For tree-based models
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    feature_names = transformed_columns
                    
                    # Create Plotly bar chart for feature importance
                    fig = go.Figure(go.Bar(
                        x=[i for i in range(len(indices))],
                        y=importances[indices],
                        marker_color='forestgreen'
                    ))
                    
                    fig.update_layout(
                        title="Feature Importance",
                        xaxis=dict(
                            tickmode='array',
                            tickvals=list(range(len(indices))),
                            ticktext=[feature_names[i] for i in indices],
                            tickangle=90
                        ),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif hasattr(model, "coef_"):
                    # For linear models
                    coeffs = model.coef_
                    if len(coeffs.shape) > 1:  # For multi-class
                        coeffs = np.abs(coeffs).mean(axis=0)
                    feature_names = transformed_columns
                    
                    # Create Plotly bar chart for coefficients
                    fig = go.Figure(go.Bar(
                        x=feature_names,
                        y=np.abs(coeffs),
                        marker_color='darkorange'
                    ))
                    
                    fig.update_layout(
                        title="Feature Coefficients (Absolute Value)",
                        xaxis=dict(
                            tickangle=90
                        ),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as fallback_error:
                st.warning("Could not generate any feature importance visualization.")