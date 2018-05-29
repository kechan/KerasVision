# https://stackoverflow.com/questions/48547660/attributeerror-module-pil-image-has-no-attribute-register-extensions
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# workaround 
from PIL import Image
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
  for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions

def download_data_from_gdrive(ids, filenames):

    for id, filename in zip(ids, filenames):
        uploaded = drive.CreateFile({'id': id})
        uploaded.GetContentFile(filename)

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


