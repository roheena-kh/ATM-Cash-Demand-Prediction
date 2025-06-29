import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import numpy as np
# Load model and scalers
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(MODEL_DIR, 'atm_weekly_cash_forecast_model.pkl'))
model_features = model.get_booster().feature_names
joblib.dump(model.get_booster().feature_names, 'atm_weekly_model_features.pkl')



def preprocess_input(df):
    """Preprocess data for weekly prediction"""
    try:
        df = df[df['TXN_CCY_CODE'] == 'AFN'].drop('TXN_CCY_CODE', axis=1)
        df['TERM_ID'] = df['TERM_ID'].astype(str)
        df['TRN_DT'] = pd.to_datetime(df['TRN_DT'], format='%Y-%m-%d')

        df = df.sort_values(by='TRN_DT').reset_index(drop=True)
        all_weeks = pd.date_range(start=df['TRN_DT'].min(), end=df['TRN_DT'].max(), freq='W-SAT')
        
        weekly_data = (
            df.set_index('TRN_DT')
            .groupby('TERM_ID')
            .resample('W-SAT', label='left', closed='left')
            .agg({'TRN_AMOUNT': 'sum'})
            .reset_index()
        )

        all_atms = df['TERM_ID'].unique()
        filled_data = []

       # ...existing code...
        for term_id in all_atms:
            atm_data = weekly_data[weekly_data['TERM_ID'] == term_id].set_index('TRN_DT')
            atm_data = atm_data.reindex(all_weeks, fill_value=np.nan)  # Use np.nan, not 0
            atm_data['TERM_ID'] = term_id
            atm_data = atm_data.reset_index().rename(columns={'index': 'TRN_DT'})
            filled_data.append(atm_data)

        weekly_data = pd.concat(filled_data, ignore_index=True)

        first_date = weekly_data['TRN_DT'].min()
        weekly_data['WEEK'] = (weekly_data['TRN_DT'] - first_date).dt.days // 7
        weekly_data['WEEK_OF_YEAR'] = weekly_data['TRN_DT'].dt.isocalendar().week

        grouped = weekly_data.groupby('TERM_ID')
        weekly_data["lag_trn_1"] = grouped['TRN_AMOUNT'].shift(1)
        weekly_data["mean_trn_2"] = grouped['TRN_AMOUNT'].transform(lambda x: x.rolling(2, min_periods=1).mean())
        weekly_data["mean_trn_4"] = grouped['TRN_AMOUNT'].transform(lambda x: x.rolling(4, min_periods=1).mean())
        weekly_data["mean_trn_6"] = grouped['TRN_AMOUNT'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        weekly_data['diff_trn_1'] = grouped['TRN_AMOUNT'].diff(1)

        # Drop dates and missing values
        processed = weekly_data.drop(columns=['TRN_DT'])
        # Optionally, only drop rows where TRN_AMOUNT is NaN
        processed = processed[~processed['TRN_AMOUNT'].isna()]

        # One-hot encode TERM_ID
        processed = pd.get_dummies(processed, columns=['TERM_ID'])

        # Align with model's features
        for col in model_features:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[model_features]

        return processed, weekly_data

    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")

def predict_future():
    """Generate predictions for next week"""
    try:
        DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'base_data.csv')
        df = pd.read_csv(DATA_PATH)

        processed_data, weekly_data = preprocess_input(df)
        atm_list = weekly_data['TERM_ID'].unique()
        predictions = []

        # Build list of all TERM_ID dummy columns seen during training
        all_dummy_cols = [col for col in model_features if col.startswith("TERM_ID_")]

        last_date = weekly_data['TRN_DT'].max()
        # Find the next Saturday after the last date in your data
        days_until_sat = (5 - last_date.weekday()) % 7
        next_sat = last_date + pd.Timedelta(days=days_until_sat)

        print("ATM list:", atm_list)
        for term_id in atm_list:
            term_history = weekly_data[weekly_data['TERM_ID'] == term_id]
            print(f"ATM {term_id} term_history:\n", term_history.tail())
            if term_history.empty:
                continue

            valid_history = term_history[~term_history['TRN_AMOUNT'].isna()]
            if valid_history.empty:
                print(f"ATM {term_id} has no data, skipping.")
                continue
            last_entry = valid_history.sort_values('TRN_DT').iloc[-1]

            predicted_week = last_entry['TRN_DT'] + timedelta(weeks=1)
            start_of_month = pd.Timestamp(predicted_week).to_period('M').start_time
            first_week = start_of_month.isocalendar().week
            week_of_year = predicted_week.isocalendar().week
            week_of_month = week_of_year - first_week + 1

            feature_row = dict.fromkeys(model_features, 0)
            feature_row.update({
                'WEEK': last_entry['WEEK'] + 1,
                'WEEK_OF_YEAR': week_of_year,
                'FIRST_WEEK_OF_MONTH': first_week,
                'WEEK_OF_MONTH': week_of_month,
                'lag_trn_1': last_entry['TRN_AMOUNT'],
                'mean_trn_2': term_history['TRN_AMOUNT'].tail(2).mean(),
                'mean_trn_4': term_history['TRN_AMOUNT'].tail(4).mean(),
                'mean_trn_6': term_history['TRN_AMOUNT'].tail(6).mean(),
                'diff_trn_1': last_entry['TRN_AMOUNT'] - term_history['TRN_AMOUNT'].iloc[-2] if len(term_history) > 1 else 0,
            })

            for col in all_dummy_cols:
                feature_row[col] = 0

            dummy_col = f"TERM_ID_{term_id}"
            if dummy_col in all_dummy_cols:
                feature_row[dummy_col] = 1
            else:
                print(f"[Warning] TERM_ID {term_id} not seen during training. Skipping.")
                continue

            feature_df = pd.DataFrame([feature_row])
            for col in model_features:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            feature_df = feature_df[model_features]

            predicted_amount = model.predict(feature_df)[0]
            predictions.append({
                'ATM_ID': term_id,
                'NEXT_WEEK_START': predicted_week.strftime('%Y-%m-%d'),
                'PREDICTED_AMOUNT': int(round(predicted_amount, 0))
            })

        return pd.DataFrame(predictions)

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    predictions = predict_future()
    predictions.to_csv('weekly_predictions.csv', index=False)
    print("Weekly predictions generated successfully:")
    print(predictions.head())
