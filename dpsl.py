import streamlit as st
import pandas as pd
import numpy as np
import io
import xlsxwriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

def preprocess_weight_data(data):
    data['order_date'] = pd.to_datetime(data['order_date'])
    data['month'] = data['order_date'].dt.month
    data['year'] = data['order_date'].dt.year
    data['day'] = data['order_date'].dt.day
    data['dayofweek'] = data['order_date'].dt.dayofweek
    data['quarter'] = data['order_date'].dt.quarter
    data['month_year'] = data['order_date'].dt.to_period('M')
    min_period = data['month_year'].min()
    data['month_year_encoded'] = data['month_year'].apply(lambda x: (x - min_period).n)
    data.set_index('order_date', inplace=True)
    for col in ['Beverages', 'Snacks']:
        data[f'{col}_rolling_mean_3m'] = data[col].rolling('90D').mean()
        data[f'{col}_expanding_max'] = data[col].expanding().max()
    data.reset_index(inplace=True)
    return data

def preprocess_drops_data(data):
    data['order_date'] = pd.to_datetime(data['order_date'])
    data['year'] = data['order_date'].dt.year
    data['month'] = data['order_date'].dt.month
    data['day'] = data['order_date'].dt.day
    data['day_of_week'] = data['order_date'].dt.dayofweek

    if 'drops_percentage' in data.columns:
        data['moving_avg_drops_3'] = data.groupby('city')['drops_percentage'].transform(lambda x: x.rolling(window=3).mean())
    else:
        data['moving_avg_drops_3'] = np.nan

    city_gmv_totals = data.groupby(['city', 'order_date'])['category_gmv'].sum().reset_index()
    city_gmv_totals.rename(columns={'category_gmv': 'city_total_gmv'}, inplace=True)
    data = pd.merge(data, city_gmv_totals, on=['city', 'order_date'])
    data['category_to_city_gmv_ratio'] = data['category_gmv'] / data['city_total_gmv']
    data['log_category_gmv'] = np.log1p(data['category_gmv'])

    return data

st.sidebar.title('Model Selection')
app_mode = st.sidebar.radio("Choose the Mode", ["Weight Modeling", "Drops Modeling"])

if app_mode == "Weight Modeling":
    st.title("Weight Cohorts Predictive Modeling")

    train_file = st.file_uploader("Upload Training Data (.xlsx)", type='xlsx', key="train")
    test_file = st.file_uploader("Upload Testing Data (.xlsx)", type='xlsx', key="test")

    if train_file and test_file:
        submit = st.button('Submit for Weight Modeling')
        if submit:
            with st.spinner('Processing... Please wait'):
                train_data = pd.read_excel(train_file)
                train_data = preprocess_weight_data(train_data)
                test_data = pd.read_excel(test_file)
                test_data = preprocess_weight_data(test_data)

                features = [col for col in train_data.columns if col not in ['order_date', 'month_year', 'city', 'drops_0_200', 'drops_200_800', 'drops_800_']]
                targets = ['drops_0_200', 'drops_200_800', 'drops_800_']
                models = {}

                for target in targets:
                    X_train = train_data[features]
                    y_train = train_data[target]
                    model = LGBMRegressor(learning_rate=0.1, max_depth=5, n_estimators=100, num_leaves=31, random_state=42)
                    model.fit(X_train, y_train)
                    models[target] = model

                    X_test = test_data[features]
                    predictions = models[target].predict(X_test)
                    test_data[target + '_predictions'] = predictions

                output_buffer = io.BytesIO()
                with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                    test_data.to_excel(writer, sheet_name='Predictions', index=False)
                output_buffer.seek(0)

                st.write(test_data)

                st.download_button(
                    label="Download prediction results as Excel",
                    data=output_buffer,
                    file_name="predictions_weight_modeling.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success('Processing complete!')

elif app_mode == "Drops Modeling":
    st.title("Drops Predictive Modeling")

    train_file = st.file_uploader("Upload Training Data (.xlsx)", type='xlsx', key="train_drops")
    test_file = st.file_uploader("Upload Testing Data (.xlsx)", type='xlsx', key="test_drops")

    if train_file and test_file:
        submit = st.button('Submit for Drops Modeling')
        if submit:
            with st.spinner('Processing... Please wait'):
                train_data = pd.read_excel(train_file)
                train_data = preprocess_drops_data(train_data)
                test_data = pd.read_excel(test_file)
                test_data_processed = preprocess_drops_data(test_data)

                categorical_features = ['city', 'core_category']
                numeric_features = ['category_gmv', 'log_category_gmv', 'category_to_city_gmv_ratio', 'moving_avg_drops_3', 'year', 'month', 'day', 'day_of_week']
                categorical_transformer = OneHotEncoder(handle_unknown='ignore')
                numeric_transformer = StandardScaler()

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])
                
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', LGBMRegressor())
                ])

                X_train = train_data.drop(['drops_percentage', 'order_date'], axis=1)
                y_train = train_data['drops_percentage']
                pipeline.fit(X_train, y_train)

                X_test = test_data_processed.drop('order_date', axis=1)
                predictions = pipeline.predict(X_test)
                test_data['predicted_drops_percentage'] = predictions

                output_buffer = io.BytesIO()
                with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                    test_data.to_excel(writer, sheet_name='Predictions', index=False)
                output_buffer.seek(0)

                st.write(test_data)

                st.download_button(
                    label="Download prediction results as Excel",
                    data=output_buffer,
                    file_name="predictions_drops_modeling.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success('Processing complete!')
