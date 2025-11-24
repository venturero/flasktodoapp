from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import sqlite3

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_path/flasktodoapp2/todo.db'
db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80))
    complete = db.Column(db.Boolean)

with app.app_context():
    db.create_all()
    print("Database and table created successfully.")
    
    try:
        connection = sqlite3.connect(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
        connection.close()
        print("Connection successful.")
    except sqlite3.Error as e:
        print("Connection failed:", e)
