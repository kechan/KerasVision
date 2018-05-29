# https://stackoverflow.com/questions/48547660/attributeerror-module-pil-image-has-no-attribute-register-extensions
from subprocess import call


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

#download_data_from_gdrive(['1Zdt10Q1Jn-hrq2o1mmvQ1j4DgBTxxGIq', '1FgVh2oGqH9Pr4Ze2NETyLnBTPtC0hTui', '1X6ijkgbWCzATPCJLx0rBCy5jtUkjo2KG'],
                          #['train_224_224.hdf5.gz', 'validation_224_224.hdf5.gz', 'test_224_224.hdf5.gz'])

call(["mkdir", "/content/data"])
call(["mkdir", "/content/data/224x224_cropped_merged_heads_hdf5"])
