import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

# Load model and dependencies
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'atm_weekly_usd_cash_forecast_model.pkl')
saved_model = joblib.load(MODEL_PATH)
model = saved_model['model']

label_encoder = saved_model['label_encoder']

def preprocess_input(df):
    """Preprocess data for weekly prediction"""
    try:
        #select only usd transactions
        
        df = df[df['TXN_CCY_CODE'] == "USD"].drop('TXN_CCY_CODE', axis=1)
        df = df[df['TRN_AMOUNT'] != 0]
        
        # Convert TERM_ID using the saved LabelEncoder
        df['TERM_ID'] = label_encoder.transform(df['TERM_ID'])
        df['TRN_DT'] = pd.to_datetime(df['TRN_DT'], format='%Y-%m-%d')
        print(df.head(), df.shape)
        # today = pd.Timestamp.today().normalize()
        # last_fri = today - pd.Timedelta(days=(today.weekday() -5) % 7)

        df = df.sort_values(by='TRN_DT').reset_index(drop=True)

        # all_weeks = pd.date_range(start=df['TRN_DT'].min(), end=last_fri, freq='W-SAT')
        
        all_weeks = pd.date_range(start=df['TRN_DT'].min(), end=df['TRN_DT'].max())
        # Group daily transactions to weekly
        weekly_data = (
            df.set_index('TRN_DT')
            .groupby('TERM_ID')
            .resample('W-SAT', label='left', closed='right')
            .agg({'TRN_AMOUNT': 'sum'})
            .reset_index()
        )
        print("Weekly data after resample:", weekly_data.shape)
        all_atms = df['TERM_ID'].unique()
        filled_data = []

        for term_id in all_atms:
            atm_data = weekly_data[weekly_data['TERM_ID'] == term_id].set_index('TRN_DT')
            atm_data = atm_data.reindex(all_weeks, fill_value=0)
            atm_data['TERM_ID'] = term_id
            atm_data = atm_data.reset_index().rename(columns={'index': 'TRN_DT'})
            filled_data.append(atm_data)

        weekly_data = pd.concat(filled_data, ignore_index=True)
        
        # Calculate week features
        first_date = weekly_data['TRN_DT'].min()
        weekly_data['WEEK'] = (weekly_data['TRN_DT'] - first_date).dt.days // 7

        weekly_data['WEEK_OF_YEAR'] = weekly_data['TRN_DT'].dt.isocalendar().week
        
        # Generate temporal features
        grouped = weekly_data.groupby('TERM_ID')
        
        # Lag features
        weekly_data["lag_trn_1"] = grouped['TRN_AMOUNT'].shift(1)
        
        # Rolling means
        weekly_data["mean_trn_2"] = grouped['TRN_AMOUNT'].transform(
            lambda x: x.rolling(2, min_periods=1).mean()
        )
        weekly_data["mean_trn_4"] = grouped['TRN_AMOUNT'].transform(
            lambda x: x.rolling(4, min_periods=1).mean()
        )
        weekly_data["mean_trn_6"] = grouped['TRN_AMOUNT'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        print("Weekly data after feature engineering:", weekly_data.isna().sum())
        # Drop dates and missing values
        processed = weekly_data.drop(columns=['TRN_DT']).dropna()
        
        return processed, weekly_data
    
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")

def predict_future():
    """Generate predictions for next week"""
    try:
        # Get historical data with absolute path
        DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'base_data.csv')
        df = pd.read_csv(DATA_PATH)
        print(df.head(), df.shape)
        # Preprocess data (including TERM_ID encoding)
        processed_data, weekly_data = preprocess_input(df)
        
        # Get list of ATMs from training data
        atm_list = weekly_data['TERM_ID'].unique()
        
        # Prepare predictions container
        predictions = []

        today = pd.Timestamp.today().normalize()
        next_sat = today + pd.Timedelta(days=(5 - today.weekday()) % 7)
        
        for term_id in atm_list:
            term_history = weekly_data[weekly_data['TERM_ID'] == term_id]
            valid_history = term_history[~term_history['TRN_AMOUNT'].isna()]
            if valid_history.empty:
                continue
            last_entry = valid_history.sort_values('TRN_DT').iloc[-1]
            predicted_week = last_entry['TRN_DT'] + timedelta(weeks=1)
           
            
            # ====== START of additional logic for FIRST_WEEK_OF_MONTH and WEEK_OF_MONTH ======
            start_of_month = pd.Timestamp(predicted_week).to_period('M').start_time
            first_week = start_of_month.isocalendar().week
            week_of_year = predicted_week.isocalendar().week
            week_of_month = week_of_year - first_week + 1
            
            # ====== END of additional logic ======
            if predicted_week < next_sat - timedelta(weeks=1):
                continue
            feature_row = {
                'TERM_ID': term_id,
                'WEEK': last_entry['WEEK'] + 1,
                'WEEK_OF_YEAR': week_of_year,
                'FIRST_WEEK_OF_MONTH': first_week,
                'WEEK_OF_MONTH': week_of_month,
                'lag_trn_1': last_entry['TRN_AMOUNT'],
                'mean_trn_2': term_history['TRN_AMOUNT'].tail(2).mean(),
                'mean_trn_4': term_history['TRN_AMOUNT'].tail(4).mean(),
                'mean_trn_6': term_history['TRN_AMOUNT'].tail(6).mean()
            }
            
            features = pd.DataFrame([feature_row]).astype({
                'TERM_ID': 'int32',
                'WEEK': 'int32',
                'WEEK_OF_YEAR': 'int32',
                'FIRST_WEEK_OF_MONTH': 'int32',
                'WEEK_OF_MONTH': 'int32',
                'lag_trn_1': 'float32',
                'mean_trn_2': 'float32',
                'mean_trn_4': 'float32',
                'mean_trn_6': 'float32'
            })[model.get_booster().feature_names]
            
            # Make prediction
            predicted_amount = model.predict(features)[0]
            
            predictions.append({
                'ATM_ID': label_encoder.inverse_transform([term_id])[0],
                'NEXT_WEEK_START': predicted_week.strftime('%Y-%m-%d'),
                'PREDICTED_AMOUNT': int(round(predicted_amount, 0))
            })
        
        return pd.DataFrame(predictions)
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    predictions = predict_future()
    predictions.to_csv('weekly_usd_predictions.csv', index=False)
    print("Weekly predictions generated successfully:")
    print(predictions.head())