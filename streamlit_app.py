import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json
import os
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
    # Create DataFrame from inputs
        input_df = pd.DataFrame([input_data])
        
        # Apply preprocessor
        input_transformed = preprocessor.transform(input_df)
        
        # Get transformed feature names and convert to DataFrame (fixes the warning)
        transformed_columns = preprocessor.get_feature_names_out()
        input_transformed_df = pd.DataFrame(input_transformed, columns=transformed_columns)
        
        # Predict
        prediction = model.predict(input_transformed_df)
        st.subheader("Prediction")
        st.write(prediction[0])