import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.figure_factory as ff
import plotly.graph_objects as go
import time
from tqdm import tqdm
from pygwalker.api.streamlit import StreamlitRenderer
import xgboost as xgb
import lightgbm as lgb
import shap
from openai import OpenAI

# Streamlit Page Configuration
st.set_page_config(page_title="@data_lemak", layout="wide")

# Sidebar Design
st.sidebar.header("專案流程")
sidebar_options = [
    "1. 加載數據",
    "2. 數據探索和可視化",
    "3. 數據預處理",
    "4. 特徵工程",
    "5. 模型選擇與訓練",
    "6. 模型評估",
    "7. 模型部署模擬",
    "8. What-If 分析"
]
selected_option = st.sidebar.radio("選擇步驟", sidebar_options)

st.sidebar.divider()
st.sidebar.header("Chatbot")
openai_apikey = st.sidebar.text_input("Enter OpenAI API Key", type="password")
if openai_apikey:
    st.sidebar.success("API Key entered!")
    client = OpenAI(api_key=openai_apikey)

# Initialize session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Main content
st.title("用戶流失率預測與聊天助手系統")

# Create tabs
tab1, tab2 = st.tabs(["聊天機器人", "用戶流失率預測系統"])

with tab1:
    st.header("聊天機器人")
    
    # Display chat messages from the session state
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])
        else:
            st.text(message["content"])

    def check_for_personal_info(prompt, openai_api_key):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You're a highly intelligent assistant. Your task is to determine if the following message contains any personal information such as names, email addresses, phone numbers, or any other details that could be used to identify an individual. Please classify it only as 1 or 0 where 1 means it contains personal information and 0 means it doesn't contain any personal information."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            ai_response = response['choices'][0]['message']['content'].strip()
            return ai_response
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # Chat input
    if openai_apikey:
        if prompt := st.chat_input("You:"):
            OpenAI.api_key = openai_apikey
            if check_for_personal_info(prompt, openai_apikey) == '1':
                st.warning("Warning: Please do not share personal information.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)

                response = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                assistant_message = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                st.chat_message("assistant").write(assistant_message)

    # Clear conversation button
    if st.button('Clear Conversation'):
        st.session_state["messages"] = [
            {"role": "system", "content": "You are an AI assistant."}
        ]
        st.experimental_rerun()

with tab2:
    st.header("用戶流失率預測系統")
    
        # Data Loading
    st.header("1. Data Loading")
    data_option = st.radio("選擇數據來源", ["上傳CSV文件", "隨機資料"])

    def load_sample_data(n_samples=1000):
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'tenure': np.random.randint(0, 60, n_samples),
            'balance': np.random.uniform(0, 250000, n_samples),
            'num_products': np.random.randint(1, 5, n_samples),
            'has_credit_card': np.random.choice([0, 1], n_samples),
            'is_active_member': np.random.choice([0, 1], n_samples),
            'estimated_salary': np.random.uniform(30000, 200000, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        return df

    if data_option == "上傳CSV文件":
        uploaded_file = st.file_uploader("上傳CSV文件", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("成功載入上傳的CSV文件！")
            st.session_state.df = df
        else:
            st.warning("Please upload a CSV file @data_lemak")
            st.stop()
    else:
        n_samples = st.number_input("設定樣本數", min_value=1000, value=1000, step=50)
        df = load_sample_data(n_samples)
        st.success("Successfully Loaded Dataset! @data_lemak")
        st.session_state.df = df

    st.write(st.session_state.df.head())

    # Data Exploration and Visualization
    st.header("2. Data Exploration and Visualization")
    with st.expander("展開以查看數據探索"):
        st.write("使用 PyGWalker 進行數據探索:")

        if 'df' in st.session_state:
            vis_spec = r"""{"config":[{"config":{"defaultAggregated":true,"geoms":["auto"],"coordSystem":"generic","limit":-1,"timezoneDisplayOffset":0,"folds":["tenure"]},"encodings":{"dimensions":[{"fid":"num_products","name":"num_products","basename":"num_products","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"churn","name":"churn","basename":"churn","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"age","name":"age","basename":"age","analyticType":"dimension","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"has_credit_card","name":"has_credit_card","basename":"has_credit_card","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"is_active_member","name":"is_active_member","basename":"is_active_member","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"}],"measures":[{"fid":"tenure","name":"tenure","basename":"tenure","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"balance","name":"balance","basename":"balance","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"estimated_salary","name":"estimated_salary","basename":"estimated_salary","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}}],"columns":[{"fid":"age","name":"age","basename":"age","analyticType":"dimension","semanticType":"quantitative","aggName":"sum","offset":0}],"color":[],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[{"fid":"tenure","name":"tenure","basename":"tenure","analyticType":"measure","semanticType":"quantitative","aggName":"mean","offset":0},{"fid":"estimated_salary","name":"estimated_salary","basename":"estimated_salary","analyticType":"measure","semanticType":"quantitative","aggName":"mean","offset":0}],"filters":[],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"full","width":320,"height":200},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false}},"visId":"gw_fRWT","name":"Chart 1"}],"chart_map":{},"workflow_list":[{"workflow":[{"type":"transform","transform":[{"key":"gw_count_fid","expression":{"op":"one","params":[],"as":"gw_count_fid"}}]},{"type":"view","query":[{"op":"aggregate","groupBy":["age"],"measures":[{"field":"gw_count_fid","agg":"sum","asFieldKey":"gw_count_fid_sum"},{"field":"tenure","agg":"mean","asFieldKey":"tenure_mean"},{"field":"estimated_salary","agg":"mean","asFieldKey":"estimated_salary_mean"}]}]}]}],"version":"0.4.9.7"}"""
            pyg_app = StreamlitRenderer(df, vis_spec=vis_spec)
            pyg_app.explorer()
        else:
            st.warning("Please load the data first! @data_lemak")

    # Data Preprocessing
    st.header("2. Data Preprocessing")
    with st.expander("展開以查看數據預處理"):
        st.write("Missing Value Imputation")
        fill_methods = {col: st.selectbox(f"選擇 {col} 的填補方式", ['mean', 'mode', '0'], key=f"fill_{col}") for col in st.session_state.df.columns}

        if st.button("執行數據預處理"):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            for i, column in tqdm(enumerate(st.session_state.df.columns), desc="執行數據預處理", total=len(st.session_state.df.columns), ncols=100, bar_format="{l_bar}{bar} [時間剩餘: {remaining}]"):
                progress_text.text(f"正在處理: {column}")
                if st.session_state.df[column].dtype != 'object':
                    if fill_methods[column] == 'mean':
                        st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].mean())
                    elif fill_methods[column] == 'mode':
                        st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].mode()[0])
                    else:
                        st.session_state.df[column] = st.session_state.df[column].fillna(0)
                else:
                    st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].mode()[0])
                progress_bar.progress((i + 1) / len(st.session_state.df.columns))
                time.sleep(0.1)  
            progress_text.text("數據預處理完成！")
            st.success("Data preprocessing completed! @data_lemak")
            st.write(st.session_state.df.head())

    # Feature Engineering
    st.header("3. Feature Engineering")
    with st.expander("展開以查看特徵工程"):
        features_to_process = ['age', 'tenure', 'balance', 'num_products', 'has_credit_card', 'is_active_member', 'estimated_salary']

        feature_bins = {}
        for feature in features_to_process:
            if st.session_state.df[feature].dtype != 'object':
                st.subheader(f"{feature} 分組設置")
                num_bins = st.slider(f"Number of bins for {feature}", 2, 10, 4, key=f"bins_{feature}")
                min_val, max_val = float(st.session_state.df[feature].min()), float(st.session_state.df[feature].max())
                bins = st.slider(f"Bin range for {feature}", min_val, max_val, (min_val, max_val), key=f"range_{feature}")
                feature_bins[feature] = np.linspace(bins[0], bins[1], num_bins + 1)

        def process_feature(feature, feature_name):
            if feature_name in ['has_credit_card', 'is_active_member']:
                return feature.map({0: 'No', 1: 'Yes'})
            elif feature_name in feature_bins:
                return pd.cut(feature, bins=feature_bins[feature_name], labels=[f'bin_{i+1}' for i in range(len(feature_bins[feature_name])-1)])
            else:
                return feature

        if st.button("執行特徵工程"):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            for i, feature in tqdm(enumerate(features_to_process), desc="執行特徵工程", total=len(features_to_process), ncols=100, bar_format="{l_bar}{bar} [時間剩餘: {remaining}]"):
                progress_text.text(f"正在處理: {feature}")
                st.session_state.df[feature + '_processed'] = process_feature(st.session_state.df[feature], feature)
                progress_bar.progress((i + 1) / len(features_to_process))
                time.sleep(0.1)
            progress_text.text("特徵工程完成！")
            st.success("Feature engineering completed! @data_lemak")
            st.write(st.session_state.df.head())

    # Select features
    st.header("4. Select Features for Training")
    features_for_model = st.multiselect(
        "選擇特徵",
        options=[col for col in st.session_state.df.columns if col.endswith('_processed') or col in features_to_process],
        default=[col for col in st.session_state.df.columns if col.endswith('_processed') or col in features_to_process]
    )

    # Model Selection and Training
    st.header("5. 模型選擇與訓練")
    tuning_method = st.radio("選擇超參數調優方法", ["手動調整", "GridSearchCV", "RandomizedSearchCV"])
    if tuning_method == "手動調整":
        n_splits = st.slider("K-Fold 交叉驗證折數", 2, 10, 5)
    model_option = st.selectbox("選擇要使用的模型", ["Random Forest", "XGBoost", "LightGBM"])


    def get_model_params(model_option):
        if model_option == "Random Forest":
            return {
                'n_estimators': st.slider("樹的數量", 10, 1000, 100),
                'max_depth': st.slider("最大深度", 1, 50, 10),
                'min_samples_split': st.slider("最小分裂樣本數", 2, 20, 2),
                'min_samples_leaf': st.slider("最小葉子樣本數", 1, 20, 1),
                'max_features': st.selectbox("最大特徵數", ["sqrt", "log2", None])
            }
        elif model_option == "XGBoost":
            return {
                'n_estimators': st.slider("樹的數量", 10, 1000, 100),
                'learning_rate': st.slider("學習率", 0.01, 1.0, 0.1),
                'max_depth': st.slider("最大深度", 1, 20, 6),
                'min_child_weight': st.slider("最小子權重", 1, 10, 1),
                'gamma': st.slider("修剪閾值", 0.0, 1.0, 0.0, 0.1),
                'subsample': st.slider("子採樣比例", 0.1, 1.0, 1.0, 0.1),
                'colsample_bytree': st.slider("特徵採樣比例", 0.1, 1.0, 1.0, 0.1)
            }
        else:  # LightGBM
            return {
                'n_estimators': st.slider("樹的數量", 10, 1000, 100),
                'learning_rate': st.slider("學習率", 0.01, 1.0, 0.1),
                'num_leaves': st.slider("葉子數量", 20, 3000, 31),
                'max_depth': st.slider("最大深度", -1, 20, -1),
                'min_child_samples': st.slider("最小子樣本數", 1, 100, 20),
                'subsample': st.slider("子採樣比例", 0.1, 1.0, 1.0, 0.1),
                'colsample_bytree': st.slider("特徵採樣比例", 0.1, 1.0, 1.0, 0.1)
            }

    def get_model(model_option, model_params):
        if model_option == "Random Forest":
            return RandomForestClassifier(**model_params, random_state=42, n_jobs=-1)
        elif model_option == "XGBoost":
            return xgb.XGBClassifier(**model_params, random_state=42, n_jobs=-1)
        else:  # LightGBM
            return lgb.LGBMClassifier(**model_params, random_state=42, n_jobs=-1)

    if tuning_method == "手動調整":
        model_params = get_model_params(model_option)
        model = get_model(model_option, model_params)

        if st.button("訓練模型"):
            X = st.session_state.df[features_for_model]
            y = st.session_state.df['churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            progress_bar = st.progress(0)
            progress_text = st.empty()
            cv_scores = []
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            with tqdm(total=n_splits, desc="總體進度") as pbar_outer:
                for fold, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
                    X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
                    y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
                    with tqdm(total=100, desc=f"Fold {fold+1}/{n_splits}", leave=False) as pbar_inner:
                        for i in range(10):  # 模擬訓練過程
                            model.fit(X_train_cv, y_train_cv)
                            pbar_inner.update(10)
                            time.sleep(0.01)  # 模擬訓練時間
                    score = model.score(X_val_cv, y_val_cv)
                    cv_scores.append(score)
                    pbar_outer.update(1)
                    progress = (fold + 1) / n_splits
                    progress_bar.progress(progress)
                    progress_text.text(f"總體進度: {progress*100:.0f}% (Fold {fold+1}/{n_splits})")

            mean_cv_score = np.mean(cv_scores)
            with tqdm(total=100, desc="最終模型訓練") as pbar_final:
                for i in range(10):  # 模擬最終訓練過程
                    model.fit(X_train_scaled, y_train)
                    pbar_final.update(10)
                    time.sleep(0.01)  # 模擬訓練時間

            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            st.session_state.update({
                'model': model,
                'X_test': X_test,
                'X_test_scaled': X_test_scaled,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'scaler': scaler,
                'features': X.columns,
                'cv_scores': cv_scores
            })

            st.success("Model training completed! You can proceed with model evaluation and interpretation. @data_lemak")

    elif tuning_method in ["GridSearchCV", "RandomizedSearchCV"]:
        param_grid = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [6, 10, 15],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'max_depth': [-1, 10, 20],
                'min_child_samples': [20, 30, 40],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }

        model_params = get_model_params(model_option)
        model = get_model(model_option, model_params)
        search_method = GridSearchCV if tuning_method == "GridSearchCV" else RandomizedSearchCV
        search = search_method(model, param_grid[model_option], cv=3, n_jobs=-1)

        if st.button("訓練模型"):
            X = st.session_state.df[features_for_model]
            y = st.session_state.df['churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            progress_bar = st.progress(0)
            progress_text = st.empty()

            n_iter = 10  # 模擬搜索過程的迭代次數
            with tqdm(total=n_iter, desc="超參數搜索進度") as pbar_search:
                for i in range(n_iter):
                    search.fit(X_train_scaled, y_train)
                    progress = (i + 1) / n_iter
                    progress_bar.progress(progress)
                    progress_text.text(f"超參數搜索進度: {progress*100:.0f}% @data_lemak")
                    pbar_search.update(1)
                    time.sleep(0.1)  # 模擬搜索時間

            best_model = search.best_estimator_

            y_pred = best_model.predict(X_test_scaled)
            y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

            st.session_state.update({
                'model': best_model,
                'X_test': X_test,
                'X_test_scaled': X_test_scaled,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'scaler': scaler,
                'features': X.columns,
                'cv_scores': []  # Clear cv_scores to avoid errors
            })

            st.success("Model training completed! You can proceed with model evaluation and interpretation. @data_lemak")

    # Model Evaluation
    st.header("6. 模型評估")

    if 'model' in st.session_state:
        if tuning_method == "手動調整":
            st.subheader("交叉驗證結果")
            cv_scores = st.session_state.get('cv_scores', [])
            if cv_scores:
                col1, col2 = st.columns(2)
                col1.metric("交叉驗證平均得分:", f"{np.mean(cv_scores):.4f}")
                col2.metric("標準差", f"{np.std(cv_scores):.4f}")

                color_option = st.selectbox("選擇圖表顏色", ["lightblue", "lightgreen", "lightcoral", "plum", "peachpuff"])
                color_map = {
                    "lightblue": "#ADD8E6",
                    "lightgreen": "#90EE90",
                    "lightcoral": "#F08080",
                    "plum": "#DDA0DD",
                    "peachpuff": "#FFDAB9"
                }
                selected_color = color_map[color_option]
                fig = go.Figure(data=[go.Bar(y=cv_scores, x=[f"Fold {i+1}" for i in range(len(cv_scores))], marker_color=selected_color)])
                fig.update_layout(title="交叉驗證各折得分", xaxis_title="Fold", yaxis_title="得分")
                st.plotly_chart(fig)
            else:
                st.write("未找到交叉驗證結果。")

        st.subheader("最終模型評估指標")
        col1, col2, col3, col4 = st.columns(4)
        
        # 使用 zero_division 參數來避免警告
        accuracy = accuracy_score(st.session_state['y_test'], st.session_state['y_pred'])
        precision = precision_score(st.session_state['y_test'], st.session_state['y_pred'], zero_division=0)
        recall = recall_score(st.session_state['y_test'], st.session_state['y_pred'], zero_division=0)
        f1 = f1_score(st.session_state['y_test'], st.session_state['y_pred'], zero_division=0)
        
        col1.metric("準確率", f"{accuracy:.4f}")
        col2.metric("精確率", f"{precision:.4f}")
        col3.metric("召回率", f"{recall:.4f}")
        col4.metric("F1 分數", f"{f1:.4f}")

        # ROC curve and PR curve
        fpr, tpr, _ = roc_curve(st.session_state['y_test'], st.session_state['y_prob'])
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(st.session_state['y_test'], st.session_state['y_prob'])

        # Allow user to adjust Plotly style
        st.subheader("Plotly 圖表樣式調整")
        plotly_template = st.selectbox("選擇 Plotly 樣式模板", ["plotly", "ggplot2", "seaborn", "simple_white", "plotly_white", "plotly_dark", "presentation", "xgridoff", "ygridoff", "gridon", "none"])

        col1, col2 = st.columns([1, 1], gap="small")

        with col1:
            fig_roc = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(
                title_text='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                height=500,
                template=plotly_template
            )
            fig_roc.add_annotation(
                x=0.5, y=0.5,
                text=f'AUC = {roc_auc:.4f}',
                showarrow=False,
                yshift=10
            )
            st.plotly_chart(fig_roc)

        with col2:
            fig_pr = go.Figure(data=go.Scatter(x=recall, y=precision, mode='lines', name='PR curve'))
            fig_pr.update_layout(
                title_text='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                height=500,
                template=plotly_template
            )
            st.plotly_chart(fig_pr)

        # Confusion Matrix
        st.subheader("混淆矩陣")
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        
        # Allow user to adjust colorscale
        st.subheader("顏色比例調整")
        colorscale = st.selectbox("選擇顏色比例", ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'])
        
        fig = ff.create_annotated_heatmap(cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'], colorscale=colorscale)
        fig.update_layout(title_text='Confusion Matrix', xaxis_title='Predicted label', yaxis_title='True label')
        st.plotly_chart(fig)

        # SHAP values for model interpretability
        st.subheader("模型解釋性 (SHAP 值)")
        with st.spinner('計算 SHAP 值中...'):
            explainer = shap.TreeExplainer(st.session_state['model'])
            shap_values = explainer.shap_values(st.session_state['X_test_scaled'])
            
            # 檢查 shap_values 的形狀
            if isinstance(shap_values, list):
                # 對於二元分類，我們通常關注正類的 SHAP 值
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                # 如果是三維數組，我們取最後一個維度（通常對應於正類）
                shap_values = shap_values[:, :, 1]
            
            # 現在 shap_values 應該是二維的了
            shap_df = pd.DataFrame(shap_values, columns=st.session_state['features'])
            shap_importance = shap_df.abs().mean().sort_values(ascending=False)
            
            colorscale = st.selectbox("選擇顏色比例", ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'],key="shap_importance")

            fig = go.Figure(go.Bar(
                y=shap_importance.index,
                x=shap_importance.values,
                orientation='h',
                marker=dict(
                    color=shap_importance.values,
                    colorscale=colorscale,
                    colorbar=dict(title="SHAP 值")
                )
            ))
            
            fig.update_layout(
                title='特徵重要性 (基於 SHAP 值)',
                xaxis_title='平均 |SHAP 值|',
                yaxis_title='特徵',
                height=500,
                width=700
            )
            
            st.plotly_chart(fig)
            
            st.write("""
            SHAP (SHapley Additive exPlanations) 值表示每個特徵對模型預測的貢獻度。
            正值表示該特徵增加了預測為正類（流失）的可能性，負值則相反。
            柱狀圖顯示了每個特徵的平均絕對 SHAP 值，代表該特徵對模型預測的整體重要性。
            """)

            # 顯示 SHAP 值的統計信息
            st.write("SHAP 值統計信息：")
            st.write(shap_df.describe())
        
        # 特徵重要性（如果模型支持）
        if hasattr(st.session_state['model'], 'feature_importances_'):
            st.subheader("特徵重要性")
            feature_importance = pd.DataFrame({
                'feature': st.session_state['features'],
                'importance': st.session_state['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            chart_type = st.selectbox("選擇圖表類型", ['Horizontal Bar', 'Vertical Bar'], key="feature_importance_chart_type")
            colorscale = st.selectbox("選擇顏色比例", ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'], key="feature_importance_colorscale")

            if chart_type == 'Horizontal Bar':
                fig = go.Figure(go.Bar(
                    y=feature_importance['feature'],
                    x=feature_importance['importance'],
                    orientation='h',
                    marker=dict(
                        color=feature_importance['importance'],
                        colorscale=colorscale,
                        colorbar=dict(title="重要性")
                    )
                ))
            elif chart_type == 'Vertical Bar':
                fig = go.Figure(go.Bar(
                    x=feature_importance['feature'],
                    y=feature_importance['importance'],
                    orientation='v',
                    marker=dict(
                        color=feature_importance['importance'],
                        colorscale=colorscale,
                        colorbar=dict(title="重要性")
                    )
                ))

            fig.update_layout(
                title='特徵重要性',
                xaxis_title='重要性' if chart_type == 'Vertical Bar' else '特徵',
                yaxis_title='特徵' if chart_type == 'Vertical Bar' else '重要性',
                height=500,
                width=700
            )

            st.plotly_chart(fig)

    else:
        st.warning("Please train the model first! @data_lemak")

    # Model Deployment Simulation
    st.header("7. Model Deployment Simulation")
    st.write("輸入新的數據，看看模型的預測結果：")

    if 'features' in st.session_state:
        input_data = {}
        for feature in st.session_state['features']:
            if st.session_state.df[feature].dtype == 'object':
                input_data[feature] = st.selectbox(f"Select {feature}", st.session_state.df[feature].unique())
            else:
                input_data[feature] = st.number_input(f"Enter {feature}", value=float(st.session_state.df[feature].mean()))

        if 'prediction_value' not in st.session_state:
            st.session_state['prediction_value'] = None

        if st.button("預測"):
            if 'model' in st.session_state:
                input_df = pd.DataFrame([input_data])
                input_scaled = st.session_state['scaler'].transform(input_df)
                prediction = st.session_state['model'].predict_proba(input_scaled)[0]
                st.session_state['prediction_value'] = f"{prediction[1]:.2%}"
            else:
                st.warning("請先訓練模型！")

        if st.session_state['prediction_value'] is not None:
            st.metric(label="客戶流失的機率", value=st.session_state['prediction_value'])
    else:
        st.warning("Please train the model first! @data_lemak")

    # What-If Analysis
    st.header("8. What-If Analysis")
    st.write("調整特徵值，看看如何影響模型的預測：")

    if 'features' in st.session_state:
        feature_to_change = st.selectbox("選擇要調整的特徵", st.session_state['features'])
        if feature_to_change in input_data:
            original_value = input_data[feature_to_change]
            new_value = st.slider(f"調整 {feature_to_change} 的值", 
                                float(st.session_state.df[feature_to_change].min()), 
                                float(st.session_state.df[feature_to_change].max()), 
                                float(original_value))

            if 'original_prediction' not in st.session_state:
                st.session_state['original_prediction'] = None
            if 'new_prediction' not in st.session_state:
                st.session_state['new_prediction'] = None
            if 'change' not in st.session_state:
                st.session_state['change'] = None

            if st.button("比較預測結果"):
                if 'model' in st.session_state:
                    original_input = pd.DataFrame([input_data])
                    original_scaled = st.session_state['scaler'].transform(original_input)
                    st.session_state['original_prediction'] = st.session_state['model'].predict_proba(original_scaled)[0][1]
                    
                    new_input = original_input.copy()
                    new_input[feature_to_change] = new_value
                    new_scaled = st.session_state['scaler'].transform(new_input)
                    st.session_state['new_prediction'] = st.session_state['model'].predict_proba(new_scaled)[0][1]
                    
                    st.session_state['change'] = float(st.session_state['new_prediction'] - st.session_state['original_prediction'])
                    
            if st.session_state['original_prediction'] is not None and st.session_state['new_prediction'] is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="原始預測", value=f"{float(st.session_state['original_prediction']):.2%}")
                with col2:
                    st.metric(label="新預測", value=f"{float(st.session_state['new_prediction']):.2%}")
                with col3:
                    if st.session_state['change'] > 0:
                        st.metric(label="變化", value=f"{st.session_state['change']:.2%}", delta=f"{st.session_state['change']:.2%}", delta_color="inverse")
                    else:
                        st.metric(label="變化", value=f"{st.session_state['change']:.2%}", delta=f"{st.session_state['change']:.2%}", delta_color="normal")
    else:
        st.warning("Please train the model first! @data_lemak")