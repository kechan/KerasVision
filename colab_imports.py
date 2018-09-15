# https://stackoverflow.com/questions/48547660/attributeerror-module-pil-image-has-no-attribute-register-extensions
from subprocess import call

google_drive_filename_id = {
'train_224_224.hdf5.gz': '1Zdt10Q1Jn-hrq2o1mmvQ1j4DgBTxxGIq',
'validation_224_224.hdf5.gz': '1FgVh2oGqH9Pr4Ze2NETyLnBTPtC0hTui',
'test_224_224.hdf5.gz': '1X6ijkgbWCzATPCJLx0rBCy5jtUkjo2KG',

'224x224_original_and_cropped_merged_heads.train_224_224.hdf5': '1ShQU-ASlFviY-N2abN4PZEl0E4gATIIr',
'224x224_original_and_cropped_merged_heads.validation_224_224.hdf5': '1vfOXhk_lgEUFS7hDgaaX8Os2comYuGL8',
'224x224_original_and_cropped_merged_heads.test_224_224.hdf5': '1EhauFTRd8QUddBAGHl5mgGWBEZplxjWK',

'224x224_original_and_cropped_merged_heads_bbox.train_224_224.hdf5': '1l3AqcXaFoJDX1lPmOO2gIUQt83Rp0NHS',
'224x224_original_and_cropped_merged_heads_bbox.validation_224_224.hdf5': '1KYrLnz00LWIawV_lbMtsJfTFqVJwL1it',
'224x224_original_and_cropped_merged_heads_bbox.test_224_224.hdf5': '1kR1QioJsYRhba2jK_51u1nOItuEaU5Co',

'150x150_cropped_merged_heads_hdf5.tar.gz': '1vXZGV3WUmGPrUTaO5NhdVndc36McTdbj',
'cropped_merged_heads_resized_224.tar.gz': '1SHVVSWY3H7P6VpqE-g3VVpmNdz_299AT',
'cropped_merged_heads_resized_512.tar.gz': '1LARMS0JatAEf8RcBztYU3EMntEdZ-8kP',
'cropped_merged_heads_resized_1024.tar.gz': '1D0xOnbWa94KdxMSI59RfL4oYz3UV072w',
'resnet101_weights_tf.h5': '1tFWWXxctzWzcjfvuotZLYemXdw7zxDKQ',
'keras_resnet50_fc1_weights_acc_0.9783.h5': '1Kkaay-LN7u8DYTWGm2XlHrDnKR3218CL',
'keras_resnet50_fc1_d0.7_weights_acc_0.9796.h5': '1sCf9MszmL41oKfXM3OxY7MKmlP3N7KVN',
'keras_resnet50_fc1_d0.7_weights_acc_0.9810.h5': '1PTikV_JkZOndZCPfcgk6uGLUpVwY55Du',
'keras_resnet50_conv1_d0.7_weights_acc_0.9823.h5': '1K9jcXyBSRQCjqrn_-JAryTvU7Y4eRl54',
'keras_resnet101_weights_acc_0.9620.h5': '1HLY_EOY1wbzHIvD3W29uMIHq8pIx-HWe',
'deep_learning_sentences.pickle': '1vn-pBVbFC0GAd7E5foa5SM_T_RT5PR-I',
'coursera_dl_lang_model.h5': '1HWJtRtNuRHN312zoc8AnFS4NlVTFqOIb',
'coursera_dl_lang_model_5.h5': '1Yb89t-szy4qidAhXeS5j8BuYAQoNiUre',
'coursera_dl_lang_model_8.h5': '1KTMt_C63Anuai52lGwvy3QSBQUjq9eI1',
'coursera_dl_lang_model_10.h5': '1JptpjvB_eHx4E1BUxqD9TvH5d3XNfLGW',
'cat_dog.zip': '1986mlvS7r-cCCc0l9yy4DBPoUTxkYed2',
'keras_resnet50_fc1_d0.75_weights_acc_0.9660.h5': '1k__rNgrUo6XrbIb6vIFIUbp3iwrGrGTC',
'keras_cyc_resnet50_fc1_d0.5_weights_acc_0.9837.h5': '16A7CfcNuTvJYVvE_dgQs1p3R9cElRrES',
'keras_cyc_resnet50_fc1_d0.5_weights_acc_0.9878.h5': '1SZRnrPr0Ibaty3v9XYquQ2m7MAAAQwUZ',
'keras_resnet50_far_less_aug_fc1_d0.5_weights_acc_0.9040.h5': '1E_b6BrQwdWYDKy7reVoWtwU-duYzfb2k',
'keras_resnet50_far_less_aug_fc1_d0.5_weights_acc_0.9204.h5': '1SXVJlDbT6ij_qLrXOxIAPkCHvq8TcI21',
'keras_resnet50_far_less_aug_conv1_d0.5_weights_acc_0.9204.h5': '1lpHe8LuP97LyoL8ceVrrSf84JDYAs9QF',
'keras_resnet50_far_less_aug_conv1_stride1_d0.5_weights_acc_0.9204.h5': '16DUjzKQ-lN-_439vy2PAO6PZyhZhyRa5',
'keras_resnet50_336_far_less_aug_conv1_stride1_d0.5_weights_acc_0.9204.h5': '1wAu1w4ARr8iYIL7vobxF5FIEZBvcyv_A',

'train_336_336.hdf5': '1dlXOsMb_8_aFu_znCIh3gn3EylDe0uIP',
'validation_336_336.hdf5': '1y8BaHz_HliYnmdQevdsC8xA7QcFzBO3C',
'test_336_336.hdf5': '1jqGWC88Nm3zkoPMPjnjoLy5ADb1E6ecF',

'336x336_cropped_merged_heads.validation_336_336.hdf5': '13QjZZnTEWxh7DEXuNPZE7Be-h1-bCA0u',
'336x336_cropped_merged_heads.test_336_336.hdf5': '1kEpTxBF-W0klUsd9rAUEtuCAhF7F5qby',

'validation_672_672.hdf5': '1F6GwyDud2yuvq650QHYJ4Zdyn4wUN68q',
'validation_672_672.hdf5.aa': '1l562_r8Qq0VyA5Kj0FwqFr81eyHsDyvB',
'validation_672_672.hdf5.ab': '1Fm0oazHER3W7QpGCP_qTPqrSSP775Rqz',
'validation_672_672.hdf5.ac': '1fCS_r2Sy_bZoe14yeDhPyYFRiVvuOkH4',
'validation_672_672.hdf5.ad': '1W7gC4HCaYkkvh9Rv3yGp-nz_jul4tzqP',
'validation_672_672.hdf5.ae': '1AgojsW9aXhMam3AwjPa6xd3L-E7eunBY',
'validation_672_672.hdf5.af': '1dln3a0mGp_FBWXtWrLWvz_5l0UBGYcQ7',
'validation_672_672.hdf5.ag': '1hX3sw9vFztjmB-xOTSwCJm747cfHJVIq',
'validation_672_672.hdf5.ah': '1_6aieOpvhwt9UEcCjIbcXz36YpitJnkC',
'validation_672_672.hdf5.ai': '1UWu94GHKqn0l2TpX_HVinpeFR1NZZQbl',
'validation_672_672.hdf5.aj': '1oMd-q3AOfez31vm-gNBOEhxafK9HCr_Q',
'validation_672_672.hdf5.ak': '1AP37Byb_mYwyR1lkom9K_GYFbXFbZQhQ',
'validation_672_672.hdf5.al': '1GzHbGd3njhWHy7rhdm2lKpU9ShzYWcGZ',
'validation_672_672.hdf5.am': '1KtbCTNUb6vFsAzfPCn9N3rdjK72ot1t2',
'validation_672_672.hdf5.an': '10-foci1vuiPkGGeTji0ub1IxFvxulbII',
'validation_672_672.hdf5.ao': '1PfNeRAYNqVcKccVbZFetT_klygBbw9Jd',
'validation_672_672.hdf5.ap': '1h2Q1YYmYESMl16tkXS8DGfgz2tmW90TS',
'validation_672_672.hdf5.aq': '1SmU_jlRgg6TGVm6ScaRzWC0L4wh1M5RT',
'validation_672_672.hdf5.ar': '1D7UO122XNKC7J1a_nBm46sWMO76pNpC2',

'yolo.h5': '1hisBnWBFIAtPk4oZHAQ5h7XK7NVAGdcl',
'test.jpg': '1K8QDVMWdFPNRnaxRwtOz5BgrNyQCLcGt',
'giraffe.jpg': '1eYxEcakDfV6nILK6xVLJc1XNJRCSfT20',
'yad2k.tar': '1qmndyzf5cXcL1sQ0b9QG80UwTquowzyq',
'coco_classes.txt': '17-s22pkaOI-MGQwtYb_07d-JoSWsOXQM',
'pascal_classes.txt': '180-ufE9INr1derRcenRZyTbCzHI9cKME',
'yolo_anchors.txt': '1fH2IEQRyV7s9S7TB0Wu1IqusO5s1X2Ni',
'car_detection_for_auto_driving_images.tar': '1o_qpLpKX-d2kjCEJKl5Z0r7OlD_5sy0x',
'pascal_voc_07_12.hdf5': '1cwy6FIDKOK-fHqJrGEUtlBl4j1d5Xdjy',
'overfit_yad2k_yolo_model.h5': '14WkdnTU1u2oxONO-JnPpmycwOR9x9j72',
'coursera_yolo_model_random_weights.h5': '1rIppSsbZg_QFFSGH9v5r-boNc9oAo11w',
'coursera_yolo_model_overfit.h5': '1dG_mGP-jQfr0xga7vBJeimm0yPRVjjZH',
'coursera_tensorflow_keras_yolo_model_overfit.h5': '13YhMOd4r392Dzk9wShkg9BTta04CelLa',

'dev_keras_resnet50_localization.h5': '1ibW7wET4Dvp9WdAXTfZT2vVtR6xPHgO9'
}

# workaround 
from PIL import Image
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
  for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions

import matplotlib.pyplot as plt

def plot_loss_accuracy(history):
  acc = history['acc']
  val_acc = history['val_acc']
  loss = history['loss']
  val_loss = history['val_loss']

  epochs = range(len(acc))

  # Plotting Accuracy vs Epoch curve
  plt.figure(figsize=(8, 8))

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  #plt.legend(loc='upper left')
  plt.legend(loc='best')
  plt.grid()


  # Plotting Loss vs Epoch curve
  plt.figure(figsize=(8, 8))

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  #plt.legend()
  plt.legend(loc='best')
  plt.grid()
