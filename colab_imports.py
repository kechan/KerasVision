# https://stackoverflow.com/questions/48547660/attributeerror-module-pil-image-has-no-attribute-register-extensions
from subprocess import call


# workaround 
from PIL import Image
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
  for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions

def download_data_from_gdrive(drive, ids, filenames):

    for id, filename in zip(ids, filenames):
        uploaded = drive.CreateFile({'id': id})
        uploaded.GetContentFile(filename)

