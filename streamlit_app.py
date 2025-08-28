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

    # List available models
    model_dir = 'models/'
    available_models = [f.replace('.joblib', '') for f in os.listdir(model_dir) if f.endswith('.joblib') and f != 'preprocessor.joblib']
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
        progress_text = st.empty()
        progress_text.text("Calculating SHAP values... Please wait.")
        
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
            # Create SHAP explainer based on model type
            st.info("Calculating SHAP values... This might take a moment.")
            
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
            
            # Clear progress indicator and update status
            progress_text.text("SHAP values calculated successfully!")
            
            # Create a DataFrame to display the feature contributions
            feature_names = transformed_columns
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_values[0]
            })
            
            # Sort by absolute SHAP value for better visualization
            shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values('Abs SHAP', ascending=False).drop('Abs SHAP', axis=1)
            
            # Display SHAP values table
            st.write("Feature Contributions to Prediction:")
            st.dataframe(shap_df)
            
            # We'll skip this section as we'll create a better Plotly visualization below
            
            # Add visualization of SHAP values
            progress_text.text("Generating SHAP visualizations...")
            st.write("SHAP Feature Contributions:")
            
            # Create a Plotly horizontal bar chart showing feature contributions
            shap_df = shap_df.sort_values('SHAP Value')  # Sort for better visualization
            
            # Create color scale based on SHAP values
            colors = ['red' if x > 0 else 'blue' for x in shap_df['SHAP Value']]
            
            fig = go.Figure(go.Bar(
                y=shap_df['Feature'],
                x=shap_df['SHAP Value'],
                orientation='h',
                marker_color=colors
            ))
            
            fig.update_layout(
                title='Feature Contribution to Prediction',
                xaxis_title='SHAP Value (Impact on Prediction)',
                height=500,
                width=700
            )
            
            # Add a vertical line at x=0 to help visualize positive vs negative impact
            fig.add_shape(
                type="line",
                x0=0, x1=0, y0=-0.5, y1=len(shap_df)-0.5,
                line=dict(color="black", width=1, dash="dash")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Try to create a detailed waterfall plot using Plotly
            try:
                st.write("SHAP Waterfall Plot (Detailed Explanation):")
                
                # Get the base value (expected value)
                base_value = 0  # Default
                if hasattr(explainer, 'expected_value'):
                    expected_val = explainer.expected_value
                    # Handle different expected_value types
                    if isinstance(expected_val, np.ndarray):
                        if task_type == 'classification':
                            pred_class = int(prediction[0])
                            if pred_class < len(expected_val) and pred_class >= 0:
                                base_value = expected_val[pred_class]
                            else:
                                base_value = expected_val[0]
                        else:
                            base_value = expected_val[0]
                    else:
                        base_value = expected_val
                        
                # Sort features by SHAP value magnitude for waterfall
                shap_df_sorted = shap_df.copy()
                shap_df_sorted['Abs_Value'] = abs(shap_df_sorted['SHAP Value'])
                shap_df_sorted = shap_df_sorted.sort_values('Abs_Value', ascending=False).head(10)
                
                # Create a Plotly waterfall chart
                cumulative_values = [base_value]
                y_values = ['Base Value']
                text_values = [f"{base_value:.4f}"]
                
                # Add each feature contribution
                for _, row in shap_df_sorted.iterrows():
                    cumulative_values.append(row['SHAP Value'])
                    y_values.append(row['Feature'])
                    text_values.append(f"{row['SHAP Value']:.4f}")
                
                # Add final prediction
                final_prediction = base_value + sum(shap_df_sorted['SHAP Value'])
                cumulative_values.append(0)  # Placeholder, not used in waterfall
                y_values.append('Final Prediction')
                text_values.append(f"{final_prediction:.4f}")
                
                # Create measure and text arrays for the waterfall chart
                measure = ['absolute'] + ['relative'] * len(shap_df_sorted) + ['total']
                
                # Create the waterfall chart
                fig = go.Figure(go.Waterfall(
                    name="SHAP Waterfall",
                    orientation="v",
                    measure=measure,
                    y=y_values,
                    x=cumulative_values,
                    text=text_values,
                    textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    decreasing={"marker": {"color": "blue"}},
                    increasing={"marker": {"color": "red"}},
                    totals={"marker": {"color": "green"}}
                ))
                
                fig.update_layout(
                    title=f"SHAP Value Contributions (Base: {base_value:.4f})",
                    showlegend=False,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                progress_text.text("SHAP analysis complete!")
                
            except Exception as e:
                st.warning(f"Could not generate waterfall plot: {str(e)}")
                
                # Try a different visualization - summary plot with Plotly
                try:
                    st.write("SHAP Summary Plot:")
                    
                    # Calculate mean absolute SHAP value for each feature
                    feature_importance = {}
                    if isinstance(shap_values, np.ndarray):
                        if len(shap_values.shape) == 2:
                            # If we have a 2D array, take mean across rows
                            mean_abs_shap = np.abs(shap_values).mean(axis=0)
                            for i, name in enumerate(feature_names):
                                feature_importance[name] = mean_abs_shap[i]
                        else:
                            # Otherwise just take absolute values of the array
                            mean_abs_shap = np.abs(shap_values[0])
                            for i, name in enumerate(feature_names):
                                feature_importance[name] = mean_abs_shap[i]
                    else:
                        # For other formats, calculate from the DataFrame
                        for feature in shap_df['Feature'].unique():
                            feature_values = shap_df[shap_df['Feature'] == feature]['SHAP Value'].values
                            feature_importance[feature] = np.abs(feature_values).mean()
                    
                    # Create a sorted list for the bar chart
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    features = [x[0] for x in sorted_features]
                    importance_values = [x[1] for x in sorted_features]
                    
                    # Create Plotly bar chart for feature importance
                    fig = go.Figure(go.Bar(
                        y=features,
                        x=importance_values,
                        orientation='h',
                        marker_color='#77B5FE'  # A blue color
                    ))
                    
                    fig.update_layout(
                        title='Mean |SHAP Value| (Feature Importance)',
                        xaxis_title='Mean Absolute SHAP Value',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    progress_text.text("SHAP analysis complete!")
                except Exception as e2:
                    st.warning(f"Could not generate summary plot either: {str(e2)}")
                    st.info("Using only the simple bar chart shown above.")
                    progress_text.text("SHAP analysis completed with limited visualizations.")
            
        except Exception as e:
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