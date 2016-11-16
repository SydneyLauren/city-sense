from __future__ import division
from math import sqrt
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    pagetitle = {'name': "CitySense"}
    intro = {'txt': 'Some intro text will go here'}
    print 'hey!'
    return render_template('index.html',
                           intro=intro,
                           pagetitle=pagetitle)


@app.route('/solve', methods=['POST'])
def solve():
    print "in solve"
    user_data = request.json
    a, b, c = user_data['a'], user_data['b'], user_data['c']
    print user_data


# from app import app
app.run(debug=True)
