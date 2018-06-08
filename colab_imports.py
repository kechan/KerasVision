# https://stackoverflow.com/questions/48547660/attributeerror-module-pil-image-has-no-attribute-register-extensions
from subprocess import call

google_drive_filename_id = {
'train_224_224.hdf5.gz': '1Zdt10Q1Jn-hrq2o1mmvQ1j4DgBTxxGIq',
'validation_224_224.hdf5.gz': '1FgVh2oGqH9Pr4Ze2NETyLnBTPtC0hTui',
'test_224_224.hdf5.gz': '1X6ijkgbWCzATPCJLx0rBCy5jtUkjo2KG',
'150x150_cropped_merged_heads_hdf5.tar.gz': '1vXZGV3WUmGPrUTaO5NhdVndc36McTdbj',
'resnet101_weights_tf.h5': '1tFWWXxctzWzcjfvuotZLYemXdw7zxDKQ',
'keras_resnet50_fc1_weights_acc_0.9783.h5': '1Kkaay-LN7u8DYTWGm2XlHrDnKR3218CL',
'keras_resnet50_fc1_d0.7_weights_acc_0.9796.h5': '1sCf9MszmL41oKfXM3OxY7MKmlP3N7KVN',
'keras_resnet101_weights_acc_0.9620.h5': '1HLY_EOY1wbzHIvD3W29uMIHq8pIx-HWe',
'deep_learning_sentences.pickle': '1vn-pBVbFC0GAd7E5foa5SM_T_RT5PR-I',
'coursera_dl_lang_model.h5': '1HWJtRtNuRHN312zoc8AnFS4NlVTFqOIb',
'coursera_dl_lang_model_5.h5': '1Yb89t-szy4qidAhXeS5j8BuYAQoNiUre',
'coursera_dl_lang_model_8.h5': '1KTMt_C63Anuai52lGwvy3QSBQUjq9eI1',
'cat_dog.zip': '1986mlvS7r-cCCc0l9yy4DBPoUTxkYed2',
'keras_resnet50_fc1_d0.75_weights_acc_0.9660.h5': '1k__rNgrUo6XrbIb6vIFIUbp3iwrGrGTC',
'keras_cyc_resnet50_fc1_d0.5_weights_acc_0.9837.h5': '16A7CfcNuTvJYVvE_dgQs1p3R9cElRrES',
'keras_cyc_resnet50_fc1_d0.5_weights_acc_0.9878.h5': '1SZRnrPr0Ibaty3v9XYquQ2m7MAAAQwUZ'
}

# workaround 
from PIL import Image
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
  for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions
