from flask import Flask,render_template,request,redirect
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager,UserMixin,login_user,login_required,current_user

app=Flask(__name__)

app.config['SECRET_KEY']='thrive_secret_key'

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///site.db'

db=SQLAlchemy(app)

bcrypt=Bcrypt(app)

login_manager=LoginManager(app)

class User(db.Model,UserMixin):
    id=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String(150),unique=True,nullable=False)
    password=db.Column(db.String(150),nullable=False)
    email=db.Column(db.String(150),unique=True,nullable=False)

class Journal(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    title = db.Column(
        db.String(200),
        nullable=False
    )

    content = db.Column(
        db.Text,
        nullable=False
    )

    user_id = db.Column(
    db.Integer,
    db.ForeignKey('user.id'),
    nullable=False
)
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user=User()
        user.username = username
        user.email = email
        user.password = hashed_password
        db.session.add(user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect('/journals')
    return render_template('login.html')

@app.route('/create-journal', methods=['GET', 'POST'])
@login_required
def create_journal():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        journal=Journal()
        journal.title = title
        journal.content = content
        journal.user_id = current_user.id
        db.session.add(journal)
        db.session.commit()
        return redirect('/journals')
    return render_template('create_journal.html')

@app.route('/journals')
@login_required
def journals():
    all_journals = Journal.query.filter_by(user_id=current_user.id).all()
    return render_template('journals.html', journals=all_journals)

@app.route("/users")
def users():
    all_users = User.query.all()
    data=""
    for user in all_users:
      data += f"""
<h1>{user.username}</h1>

<p>{user.email}</p>

<p>{user.password}</p>

<hr>
"""
    return data

if __name__ == '__main__':
    app.run(debug=True) 
      