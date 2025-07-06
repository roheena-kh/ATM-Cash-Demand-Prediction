from flask import Flask, render_template
from daily_routes import daily_bp
from weekly_routes import weekly_bp
from db_connection import fetch_and_save_data
from available_cash import get_available_cash
from flask import jsonify
import os

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))

# Register blueprints
app.register_blueprint(daily_bp, url_prefix='/daily')
app.register_blueprint(weekly_bp, url_prefix='/weekly')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/available-cash')
def api_available_cash():
    data = get_available_cash()
    return jsonify(data)

if __name__ == '__main__':
    fetch_and_save_data()
    app.run(host='0.0.0.0', port=5000, debug=True)