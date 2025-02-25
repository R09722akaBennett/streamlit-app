import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="廣告變現收益預測", layout="wide")


@st.cache_data
def load_initial_data(dataset_name="kdan_android"):
    if dataset_name == "kdan_android":
        data = {
            'date': ['2022-01-01','2022-02-01','2022-03-01','2022-04-01','2022-05-01','2022-06-01','2022-07-01','2022-08-01','2022-09-01','2022-10-01','2022-11-01','2022-12-01','2023-01-01','2023-02-01','2023-03-01','2023-04-01','2023-05-01','2023-06-01','2023-07-01','2023-08-01','2023-09-01','2023-10-01','2023-11-01','2023-12-01'],
            'cost': [
                704177, 762384, 812837, 904768, 1013294, 1217421,
                1328718, 1530757, 1547773, 1548895, 1452694, 1095080,
                897250, 842486, 1036517, 1154801, 1042375, 1263188,
                727369, 494465, 382925, 353211, 509009, 506131
            ],
            'active_user': [
                1487546, 1468368, 1464235, 1402852, 1386879, 1369241, 1356332, 1364901, 1347618, 1294489, 1287219, 1199877, 1262118, 1188010, 1221980, 1135310, 1116841, 1099087, 944065, 969298, 946241, 892729, 823957, 759620
            ],
            'revenue': [
                1665937, 1513545, 1831731, 1937624, 1874419, 1723995,
                1979887, 1998035, 1746071, 1331042, 1258247, 1121431,
                1059160, 999901, 1076458, 943998, 1077483, 1162024,
                1073448, 1023352, 848734, 749857, 749430, 792460
            ]
        }
    elif dataset_name == "cs_android":
        data = {
    'date': [
        '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01',
        '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01',
        '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01',
        '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01'
    ],
    'cost': [
        411166.0, 459678.0, 467154.0, 358090.0, 321809.0, 375642.0,
        516304.0, 389143.0, 325003.0, 286079.0, 356050.0, 293915.0,
        112422.0, 109266.0, 113934.0, 139135.0, 129609.0, 141700.0,
        172153.0, 198878.0, 159169.0, 173970.0, 194594.0, 181865.0
    ],
    'active_user': [
        780453.0, 869452.0, 938582.0, 794291.0, 794872.0, 751335.0,
        747692.0, 769719.0, 790245.0, 778229.0, 798234.0, 698742.0,
        618280.0, 586202.0, 583201.0, 561235.0, 548519.0, 512903.0,
        500318.0, 496642.0, 482258.0, 489265.0, 476973.0, 450954.0
    ],
    'revenue': [
        233243.0, 265104.0, 345975.0, 307883.0, 309512.0, 296876.0,
        307495.0, 285060.0, 276729.0, 132227.0, 174468.0, 141109.0,
        95627.0, 70130.0, 65817.0, 53972.0, 54981.0, 102854.0,
        115993.0, 104805.0, 103678.0, 94063.0, 91066.0, 56464.0
    ]
}
    else:
        raise ValueError("Unknown dataset name")

    return pd.DataFrame(data)

def prepare_future_regressor(historical_data, column_name, forecast_periods, months=6):
    """
    使用最後6個月的平均值作為未來預測值
    """
    last_6_months = historical_data[column_name].tail(months)
    mean_value = last_6_months.mean()
    return [mean_value] * forecast_periods



def prepare_data(df):
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        raise ValueError(f"Error converting 'date' column to datetime: {e}")
    
    for column in ['revenue', 'cost', 'active_user']:
        try:
            # Ensure colum type
            if df[column].dtype != 'object':
                df[column] = df[column].astype(str)
            df[column] = pd.to_numeric(df[column].str.replace(',', ''), errors='coerce')
        except Exception as e:
            raise ValueError(f"Error converting '{column}' column to float: {e}")
    
    # Check Null Value
    if df[['date', 'revenue', 'cost', 'active_user']].isnull().any().any():
        raise ValueError("Data contains NaN values after conversion. Please check your data for invalid entries.")
        
    # Preparing model data
    prophet_df = df.rename(columns={
        'date': 'ds',
        'revenue': 'y'
    }).copy()
    
    # Create regressors
    prophet_df['cost'] = df['cost']
    prophet_df['active_user'] = df['active_user']
    
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    min_cap = max(0, df['revenue'].min() * 0.5)  
    max_cap = df['revenue'].max() * 1.5          
    prophet_df['floor'] = min_cap
    prophet_df['cap'] = max_cap
    
    return prophet_df

def get_model_parameters(model_name="model_kdan_android"):
    # Config Model Parameters after testing
    model_configs = {
        "model_kdan_android": {
            "model_params": {
                'seasonality_mode': 'multiplicative',
                'growth': 'logistic',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.0005,
                'seasonality_prior_scale': 0.01,
                'interval_width': 0.67,
                'n_changepoints': 6
            },
            "seasonality_params": {
                'period': 30.5,
                'fourier_order': 2,
                'prior_scale': 0.01
            },
            "regressor_params": {
                'cost': {
                    'prior_scale': 0.25,
                    'mode': 'additive'
                },
                'active_user': {
                    'prior_scale': 0.95,
                    'mode': 'additive'
                }
            }
        },
        "custom": {
            "model_params": {
                'seasonality_mode': 'multiplicative',
                'growth': 'logistic',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.001,
                'seasonality_prior_scale': 0.1,
                'interval_width': 0.95,
                'n_changepoints': 10
            },
            "seasonality_params": {
                'period': 30.5,
                'fourier_order': 3,
                'prior_scale': 0.1
            },
            "regressor_params": {
                'cost': {
                    'prior_scale': 0.5,
                    'mode': 'additive'
                },
                'active_user': {
                    'prior_scale': 0.5,
                    'mode': 'additive'
                }
            }
        },
        "model_cs_android": {
            "model_params": {
                'seasonality_mode': 'multiplicative',
                'growth': 'logistic',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.0003,
                'seasonality_prior_scale': 0.005,
                'interval_width': 0.58,
                'n_changepoints': 8
            },
            "seasonality_params": {
                'period': 30.5,
                'fourier_order': 3,
                'prior_scale': 0.02
            },
            "regressor_params": {
                'cost': {
                    'prior_scale': 0.4,
                    'mode': 'multiplicative'
                },
                'active_user': {
                    'prior_scale': 0.85,
                    'mode': 'multiplicative'
                }
            }
        }
    }
    
    return model_configs.get(model_name)

def customize_model_parameters(st, base_params):
    st.markdown("### 模型參數設置")
    
    with st.expander("基本參數設置"):
        model_params = base_params["model_params"].copy()
        model_params['seasonality_mode'] = st.selectbox(
            "季節性模式",
            ['multiplicative', 'additive'],
            index=0 if model_params['seasonality_mode'] == 'multiplicative' else 1
        )
        model_params['growth'] = st.selectbox(
            "成長模式",
            ['logistic', 'linear', 'flat'],
            index=0 if model_params['growth'] == 'logistic' else 1
        )
        model_params['changepoint_prior_scale'] = st.number_input(
            "變點先驗尺度",
            min_value=0.0001,
            max_value=0.5,
            value=float(model_params['changepoint_prior_scale']),
            format='%f'
        )
        model_params['seasonality_prior_scale'] = st.number_input(
            "季節性先驗尺度",
            min_value=0.01,
            max_value=10.0,
            value=float(model_params['seasonality_prior_scale']),
            format='%f'
        )
        model_params['interval_width'] = st.slider(
            "預測區間寬度",
            min_value=0.5,
            max_value=0.95,
            value=float(model_params['interval_width'])
        )
        model_params['n_changepoints'] = st.slider(
            "變點數量",
            min_value=1,
            max_value=20,
            value=int(model_params['n_changepoints'])
        )
    
    with st.expander("季節性參數設置"):
        seasonality_params = base_params["seasonality_params"].copy()
        seasonality_params['fourier_order'] = st.slider(
            "傅立葉階數",
            min_value=1,
            max_value=10,
            value=int(seasonality_params['fourier_order'])
        )
        seasonality_params['prior_scale'] = st.number_input(
            "季節性先驗尺度",
            min_value=0.01,
            max_value=10.0,
            value=float(seasonality_params['prior_scale']),
            format='%f'
        )
    
    with st.expander("Regressor 參數設置"):
        regressor_params = base_params["regressor_params"].copy()
        for regressor in ['cost', 'active_user']:
            st.markdown(f"#### {regressor} 設置")
            regressor_params[regressor]['prior_scale'] = st.number_input(
                f"{regressor} 先驗尺度",
                min_value=0.01,
                max_value=10.0,
                value=float(regressor_params[regressor]['prior_scale']),
                format='%f',
                key=f"{regressor}_prior_scale"
            )
            regressor_params[regressor]['mode'] = st.selectbox(
                f"{regressor} 模式",
                ['additive', 'multiplicative'],
                index=0 if regressor_params[regressor]['mode'] == 'additive' else 1,
                key=f"{regressor}_mode"
            )
    
    return {
        "model_params": model_params,
        "seasonality_params": seasonality_params,
        "regressor_params": regressor_params
    }

def train_model(df_prepared, model_params, seasonality_params, regressor_params):
    model = Prophet(**model_params)
    
    # add seasonality
    model.add_seasonality(
        name='monthly',
        **seasonality_params
    )
    
    # add regressors
    for regressor_name, params in regressor_params.items():
        if regressor_name in df_prepared.columns:
            model.add_regressor(regressor_name, **params)
    
    model.fit(df_prepared)
    
    return model


def predict_revenue(model, future_costs, periods, historical_data):
    # create future dates
    future_dates = model.make_future_dataframe(
        periods=periods,
        freq='MS'
    )
    
    historical_costs = model.history['cost']
    
    future_dates['cost'] = pd.concat([
        historical_costs,
        pd.Series(future_costs[:periods])  
    ]).reset_index(drop=True)
    
    future_user_values = prepare_future_regressor(historical_data, 'active_user', len(future_dates))  # Use the length of future_dates

    future_dates['active_user'] = future_user_values  

    # 添加 floor 和 cap 列
    min_cap = max(0, historical_data['revenue'].min() * 0.5)  
    max_cap = historical_data['revenue'].max() * 1.5          
    future_dates['floor'] = min_cap
    future_dates['cap'] = max_cap

    # Check Null Value
    if future_dates['cost'].isna().any() or future_dates['active_user'].isna().any():
        raise ValueError("成本或活躍用戶數據中存在缺失值，請確保所有時間點都有對應的數據")
    
    # Forecasting
    forecast = model.predict(future_dates)
    
    # Get only forecast data
    forecast_results = forecast.tail(periods)
    
    # Preparing final dataframe
    results = pd.DataFrame({
        'date': range(1, periods + 1),
        'cost': future_costs[:periods],
        'predict_revenue': forecast_results['yhat'],
        'lower_bound': forecast_results['yhat_lower'],
        'upper_bound': forecast_results['yhat_upper']
    })
    
    results['roi_lower'] = round(results['predict_revenue'] / results['cost'],2)
    results['roi_upper'] = round(results['upper_bound'] / results['cost'],2)
    
    results['predict_revenue'] = round(results['predict_revenue'],2)
    results['upper_bound'] = round(results['upper_bound'],2)
    results['date'] = pd.to_datetime(future_dates['ds']).dt.strftime('%Y-%m') 
    return results, forecast

def main():
    st.title('廣告變現收益預測')
    
    st.sidebar.subheader("選擇數據集和預算")
    dataset_choice = st.sidebar.radio(
        "選擇數據集",
        ["kdan_android", "cs_android"]
    )

    # Define default budgets for each type
    default_budgets_kdan = {
        1: 496675.0, 
        2: 646544.0, 
        3: 631547.0, 
        4: 730672.0, 
        5: 1192148.0, 
        6: 813243.0, 
        7: 782203.0, 
        8: 780915.0, 
        9: 780966.0, 
        10: 794793.0, 
        11: 794793.0, 
        12: 794793.0
    }
    default_budgets_cs = {
        1: 26584.0, 
        2: 27790.0, 
        3: 33841.0, 
        4: 43701.0, 
        5: 38353.0, 
        6: 33748.0, 
        7: 46247.0, 
        8: 47568.0, 
        9: 49241.0, 
        10: 60630.0,
        11: 60630.0,
        12: 60630.0
    }

    # Set default budgets based on dataset choice
    if dataset_choice == "kdan_android":
        default_budgets = default_budgets_kdan
    else:
        default_budgets = default_budgets_cs

    # Set monthly budgets based on the selected default
    monthly_budget = [
        float(st.sidebar.text_input(
            f'第 {month} 月預算',
            value=str(default_budgets[month])
        ))
        for month in range(1, 13)
    ]
    
    # Load initial data based on dataset choice
    df = load_initial_data(dataset_choice)
    

    data_source = st.radio("選擇數據來源", ("使用預設資料", "上傳 CSV 文件"))
    if data_source == "上傳 CSV 文件":
        uploaded_file = st.file_uploader("上傳 CSV 文件", type=["csv"])
        st.markdown("**文件格式要求:**")
        st.markdown("上傳的 CSV 文件必須包含以下四個欄位: ")
        st.markdown("- `date`: 日期，格式為 `YYYY-MM-DD`")
        st.markdown("- `cost`: 歷史投遞金額")
        st.markdown("- `revenue`: 歷史變現收益")
        st.markdown("- `active_user`: 歷史活躍用戶數")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            expected_columns = ['date', 'cost', 'revenue', 'active_user']
            if not all(col in df.columns for col in expected_columns):
                st.error("上傳的文件格式不正確，請確保包含以下列: " + ", ".join(expected_columns))
                return
            df['date'] = pd.to_datetime(df['date'])
            st.subheader("上傳的數據")
            df_display = df.rename(columns={
                'date': '日期',
                'cost': '歷史投遞金額',
                'active_user': '歷史活躍用戶數',
                'revenue': '歷史變現收益'
            })
            df_display['日期'] = pd.to_datetime(df_display['日期']).dt.strftime('%Y-%m')
            st.dataframe(df_display, use_container_width=True)

    else:
        df['date'] = pd.to_datetime(df['date'])
        
        df_display = df.rename(columns={
            'date': '日期',
            'cost': '歷史投遞金額',
            'active_user': '歷史活躍用戶數',
            'revenue': '歷史變現收益'
        })
        st.dataframe(df_display, use_container_width=True)

    st.sidebar.subheader("選擇模型參數")
    model_param_choice = st.sidebar.radio(
        "選擇模型參數",
        ["kdan_android", "cs_android"]
    )
    # Prepare data and train model
    df_prepared = prepare_data(df)
    selected_params = get_model_parameters(f"model_{model_param_choice}")
    model = train_model(
        df_prepared, 
        selected_params["model_params"],
        selected_params["seasonality_params"],
        selected_params["regressor_params"]
    )
    
    # Display selected model parameters
    st.subheader("使用的模型參數")
    params_df = pd.DataFrame([
        {'參數類型': param_type, '參數名稱': param_name, '參數值': str(param_value)}
        for param_type, params in [
            ('基本參數', selected_params["model_params"].items()),
            ('季節性參數', selected_params["seasonality_params"].items()),
            ('Cost參數', selected_params["regressor_params"]['cost'].items()),
            ('Active User參數', selected_params["regressor_params"]['active_user'].items())
        ]
        for param_name, param_value in params
    ])
    st.dataframe(params_df, use_container_width=True)
    
    st.subheader("預測結果")
    
    results, forecast = predict_revenue(model, monthly_budget, 12, df)
    results_display = results[['date', 'cost', 'predict_revenue', 'upper_bound', 'roi_lower', 'roi_upper']].rename(columns={
        'date': '預測日期',
        'cost': '預期預算',
        'predict_revenue':'預測下限',
        'upper_bound':'預測上限',
        'roi_lower': 'ROAS下限',
        'roi_upper': "ROAS上限"})   
    metrics = [
        ("未來12個月預測總預算", results['cost'].sum().round(0)),
        ("未來12個月預測總收益（下限）", results['predict_revenue'].sum().round(0)),
        ("未來12個月預測總收益（上限）", results['upper_bound'].sum().round(0)),
        ("未來12個月預測總ROAS（下限）", results['roi_lower'].mean().round(2)),
        ("未來12個月預測總ROAS（上限）", results['roi_upper'].mean().round(2))
    ]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("未來12個月預測總預算", int(metrics[0][1]))
    col2.metric("未來12個月預測總收益（下限）", int(metrics[1][1]))
    col3.metric("未來12個月預測總收益（上限）", int(metrics[2][1]))
    col4.metric("未來12個月預測總ROAS（下限）", float(metrics[3][1]))
    col5.metric("未來12個月預測總ROAS（上限）", float(metrics[4][1]))

    st.dataframe(results_display, use_container_width=True) 
    future_active_user = prepare_future_regressor(df, 'active_user', 12)
    st.metric("預測活躍用戶數",int(future_active_user[0].round(0)))
    st.write("使用最近6個月平均活躍用戶為估值")
    
    components = ['trend', 'seasonal', 'cost', 'active_user']
    component_importance = {}
    for comp in components:
        if comp in forecast.columns:
            component_importance[comp] = abs(forecast[comp]).mean()
    st.write("各變數平均影響力：")
    st.bar_chart(component_importance)
    


    
    # 1. 收入預測圖
    st.markdown("### 收入預測圖")
    st.markdown("此圖顯示了歷史收入數據以及未來預測的收入範圍。紅色虛線表示預測的下限，橙色虛線表示預測的上限，紅色區域顯示了預測的信賴區間(0.95)。")
    
    fig1 = make_subplots(rows=1, cols=1, subplot_titles=("收入預測圖",))
    fig1.add_trace(
        go.Scatter(x=df['date'], y=df['revenue'], name='歷史收入', line=dict(color='blue', width=2)),
    )
    fig1.add_trace(
          go.Scatter(
            x=df['date'], 
            y=df['active_user'], 
            name='活躍使用者', 
            line=dict(color='green', width=2),
            hovertemplate='活躍使用者: %{y:.2f}<br>日期: %{x}<extra></extra>'
        ),
    )
    fig1.add_trace(
        go.Scatter(
            x=results['date'], 
            y=results['predict_revenue'], 
            name='預測下限', 
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='預測下限: %{y:.2f}<br>日期: %{x}<extra></extra>'
        ),
    )
    fig1.add_trace(
        go.Scatter(
            x=results['date'], 
            y=results['upper_bound'], 
            name='預測上限', 
            line=dict(color='orange', dash='dash', width=2),
            hovertemplate='預測上限: %{y:.2f}<br>日期: %{x}<extra></extra>'
        ),
    )
    fig1.add_trace(
        go.Scatter(
            x=pd.concat([results['date'], results['date'][::-1]]), 
            y=pd.concat([results['upper_bound'], results['predict_revenue'][::-1]]), 
            fill='toself', 
            fillcolor='rgba(255,0,0,0.2)', 
            name='預測區間', 
            line=dict(color='rgba(255,0,0,0)'),
            hovertemplate='預測區間: %{y:.2f}<br>日期: %{x}<extra></extra>'
        )
    )
    fig1.update_layout(
        title='收入預測與歷史數據比較',
        xaxis_title='日期',
        yaxis_title='收入',
        legend_title='圖例',
        template='plotly_white'
    )
    st.plotly_chart(fig1)

    # 2. ROAS 分析
    st.markdown("### ROAS 分析")
    st.markdown("此圖顯示了預測的投資回報率（ROAS）。ROAS 是預測收入與預算的比率，幫助評估投資的效益。")
    
    fig2 = make_subplots(rows=1, cols=1, subplot_titles=("ROAS 分析",))
    fig2.add_trace(
        go.Scatter(x=results['date'], y=results['roi_lower'], name='預測 ROAS', line=dict(color='purple', width=2),
                   hovertemplate='日期: %{x}<br>預測 ROAS: %{y:.2f}<extra></extra>'),
    )
    fig2.update_layout(
        title='預測 ROAS 分析',
        xaxis_title='日期',
        yaxis_title='ROAS',
        legend_title='圖例',
        template='plotly_white'
    )
    st.plotly_chart(fig2)

    # 3. 趨勢分解
    st.markdown("### 趨勢分解")
    st.markdown("此圖顯示了預測模型中的趨勢成分，幫助理解收入隨時間的變化趨勢。")
    
    fig3 = make_subplots(rows=1, cols=1, subplot_titles=("趨勢分解",))
    fig3.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['trend'], name='趨勢', line=dict(color='blue', width=2)),
    )
    fig3.update_layout(
        title='趨勢分解',
        xaxis_title='日期',
        yaxis_title='趨勢',
        legend_title='圖例',
        template='plotly_white'
    )
    st.plotly_chart(fig3)

if __name__ == "__main__":
    main()