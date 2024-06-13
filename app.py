from flask import Flask, flash, redirect, render_template, jsonify, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import os
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, ValidationError
from wtforms.validators import InputRequired
import glob
import secrets
import time

# your DataScience and Graph scripts
import WCA_script

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Define the folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'DATA', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'Outputs')

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)

# Add a secret key for CSRF protection
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def file_type_check(form, field):
    if not allowed_file(field.data.filename):
        raise ValidationError(
            'Invalid file type. Only .txt files are allowed.')


class UploadForm(FlaskForm):
    file = FileField('File', validators=[InputRequired(), file_type_check])
    submit = SubmitField('Upload')


@app.route("/", methods=['GET', 'POST'])
def home():
    print("\nBase dir: ", BASE_DIR)
    print(f"\nupload: {UPLOAD_FOLDER}")
    print(f"\noutput: {OUTPUT_FOLDER}\napp.py ends\n")
    # clear_output_folder()  # Only clear if necessary
    form = UploadForm()
    return render_template("Home.html", form=form, max_content_length=app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024))


@app.route("/upload", methods=['POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            session['uploaded_file'] = filepath
            return jsonify({'success': True, 'message': f'"{filename}" uploaded successfully!'}), 200
        except Exception as e:
            return jsonify({'success': False, 'message': f'An error occurred while saving the file: {str(e)}'}), 500

    return jsonify({'success': False, 'message': 'Invalid file type.'}), 400


@app.route("/execute", methods=['POST'])
def execute_script():
    if 'uploaded_file' in session:
        filepath = session['uploaded_file']
        try:
            res = WCA_script.main(filepath, BASE_DIR)
            if res:
                return jsonify({'success': True, 'message': 'Script executed successfully!'}), 200
            else:
                return jsonify({'success': False, 'message': 'Script execution failed.'}), 500
        except Exception as e:
            return jsonify({'success': False, 'message': f'An error occurred during script execution: {str(e)}'}), 500

    return jsonify({'success': False, 'message': 'No file uploaded.'}), 400


@app.route("/plots", methods=['GET'])
def plots():
    plot_files = []
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'])
    try:
        for filename in sorted(os.listdir(output_folder)):  # Sort filenames
            print(f"\n{filename}")
            if filename.endswith(".html"):  # Adjust if you have different file formats
                plot_files.append(filename)
                # plot_files.append(f'Outputs/{filename}')
        
        if plot_files:
            return render_template("Result.html", plot_files=plot_files)
        else:
            flash('No plot found. Please upload a file first.', 'error')
            return redirect(url_for('home'))
    except Exception as e:
        flash(f'An error occurred during processing: {str(e)}', 'error')
        return redirect(url_for('home'))
    
@app.route('/static/Outputs/<path:filename>')
def serve_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static', 'Outputs'), filename)


# @app.route("/cleanup", methods=['POST'])
# def cleanup():
#     try:
#         clear_output_folder()
#         return jsonify({'success': True, 'message': 'Cleanup successful.'}), 200
#     except Exception as e:
#         return jsonify({'success': False, 'message': f'Cleanup failed: {str(e)}'}), 500


# def clear_output_folder():

#     # Clear UPLOAD_FOLDER
#     files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
#     for file_path in files:
#         try:
#             os.remove(file_path)
#             print(f"\n\nUPLOAD_FOLDER - Deleted file: {file_path}\n\n\n")
#         except Exception as e:
#             print(f"Error deleting file {file_path}: {e}")

#     # Clear OUTPUT_FOLDER
#     files = glob.glob(os.path.join(OUTPUT_FOLDER, '*'))
#     for file_path in files:
#         try:
#             os.remove(file_path)
#             print(f"\n\nOUTPUT_FOLDER - Deleted file: {file_path}\n\n\n")
#         except Exception as e:
#             print(f"Error deleting file {file_path}: {e}")


if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])
    # clear_output_folder()
    # os.system("cls")
    # Get the port from the environment variable PORT or use 5000 as the default
    app.run(debug=True)
