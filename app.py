from flask import Flask, render_template, request, redirect

from flask_sqlalchemy import SQLAlchemy

from flask_bcrypt import Bcrypt

from flask_login import LoginManager, UserMixin, login_user


# CREATE APP

app = Flask(__name__)


# SECRET KEY

app.config['SECRET_KEY'] = 'thrive_secret_key'


# DATABASE

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'


# INITIALIZE DATABASE

db = SQLAlchemy(app)


# PASSWORD HASHING

bcrypt = Bcrypt(app)


# LOGIN MANAGER

login_manager = LoginManager(app)


# USER MODEL

class User(db.Model, UserMixin):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(20),
        unique=True,
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )


# USER LOADER

@login_manager.user_loader
def load_user(user_id):

    return User.query.get(int(user_id))


# HOME ROUTE

@app.route("/")
def home():

    return redirect("/login")


# REGISTER ROUTE

@app.route("/register", methods=['GET', 'POST'])

def register():

    if request.method == 'POST':

        username = request.form.get("username")

        email = request.form.get("email")

        password = request.form.get("password")


        # HASH PASSWORD

        hashed_password = bcrypt.generate_password_hash(
            password
        ).decode('utf-8')


        # CREATE USER

        user = User()
        user.username = username
        user.email = email
        user.password = hashed_password


        # SAVE USER

        db.session.add(user)

        db.session.commit()


        return redirect("/login")


    return render_template("register.html")


# LOGIN ROUTE

@app.route("/login", methods=['GET', 'POST'])

def login():

    if request.method == 'POST':

        email = request.form.get("email")

        password = request.form.get("password")


        # FIND USER

        user = User.query.filter_by(email=email).first()


        # CHECK PASSWORD

        if user and bcrypt.check_password_hash(
            user.password,
            password
        ):

            login_user(user)

            return redirect("/dashboard")


    return render_template("login.html")


# DASHBOARD

@app.route("/dashboard")

def dashboard():

    return "WELCOME TO THRIVE SPACE DASHBOARD"

@app.route("/users")

def users():

    all_users = User.query.all()

    data = ""

    for user in all_users:

        data += f"""
        <h1>{user.username}</h1>

        <p>{user.email}</p>

        <p>{user.password}</p>

        <hr>
        """

    return data
# RUN APP

if __name__ == "__main__":

    app.run(debug=True)