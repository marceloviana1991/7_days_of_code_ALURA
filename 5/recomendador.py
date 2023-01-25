import pandas as pd
from flask import Flask, render_template, request
from modelo import recomendador


app = Flask(__name__)


@app.route('/')
def inicio():
    return render_template('inicio.html')


@app.route('/lista')
def lista():
    return pd.read_csv('movies.csv').to_html()


@app.route('/recomendacao', methods=['POST',])
def recomendacao():
    nome = request.form['nome']
    nome = nome.strip()
    return recomendador(nome).to_html()


app.run()

