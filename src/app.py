# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, redirect, url_for

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__, template_folder='templates')


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def home():
    return render_template('home.html')


@app.route('/explore')
def explore():
    return render_template('explore.html')


@app.route('/discriminate')
def discriminate():
    return render_template('discriminate.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
