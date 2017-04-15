import numpy as np
import numpy.random
import os.path
import scipy.misc
import tensorflow as tf

import srez_model
import srez_input

FLAGS = tf.app.flags.FLAGS

def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):

    if image_size is None:
        image_size = FLAGS.sample_size
    
    # Read each JPEG file
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    channels = 3
    image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
    image.set_shape([None, None, channels])

    # Crop and other random augmentations
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, .95, 1.05)
    #image = tf.image.random_brightness(image, .05)
    #image = tf.image.random_contrast(image, .95, 1.05)

    image = tf.reshape(image, [1, 256,256, 3])
    image = tf.cast(image, tf.float32)/255.0

    image = tf.image.resize_area(image, [64, 64])
    feature = tf.reshape(image, [64, 64, 3])

    # Using asynchronous queues
    features = tf.train.batch([feature],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name='features')

    tf.train.start_queue_runners(sess=sess)
      
    return features

def infere(sess):
    """Generate image based on model"""

    # Setup async input queues
    filenames = tf.gfile.ListDirectory(FLAGS.source_fake_dir)
    filenames = sorted(filenames)
    filenames = [os.path.join(FLAGS.source_fake_dir, f) for f in filenames if f[-4:]=='.jpg']
    features = setup_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, features, features)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    chk_filename = 'checkpoint_new.txt'
    chk_filename = os.path.join(FLAGS.checkpoint_dir, chk_filename)
    saver.restore(sess, chk_filename)

    # Infere on an image
    feature, _ = sess.run([features, features])
    feed_dict = {gene_minput: feature}
    gene_output = sess.run(gene_moutput, feed_dict=feed_dict)

    size = [64, 64] #[label.shape[1], label.shape[2]]

    nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    bicubic = tf.image.resize_bicubic(feature, size)
    bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    image   = tf.concat(2, [nearest, bicubic, clipped])
    #image   = tf.concat(0, [image[0,:,:,:]])
    print("There are ", len(filenames), "files")
    image = tf.concat(0, [image[i,:,:,:] for i in range(len(filenames))])

    image = sess.run(image)

    res_filename = 'new_results.png'
    res_filename = os.path.join(FLAGS.result_dir, res_filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(res_filename)
    print("    Saved result")    

def retro_infere(sess):
    """Generate image based on model"""

    # Setup async input queues
    filenames = tf.gfile.ListDirectory(FLAGS.source_fake_dir)
    filenames = sorted(filenames)
    filenames = [os.path.join(FLAGS.source_dir, f) for f in filenames if f[-4:]=='.jpg']

    labelnames = tf.gfile.ListDirectory(FLAGS.source_real_dir)
    labelnames = sorted(labelnames)
    labelnames = [os.path.join(FLAGS.source_real_dir, f) for f in labelnames if f[-4:]=='.jpg']
    features, labels = srez_input.setup_inputs(sess, filenames, labelnames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, features, labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    chk_filename = 'checkpoint_new.txt'
    chk_filename = os.path.join(FLAGS.checkpoint_dir, chk_filename)
    saver.restore(sess, chk_filename)

    # Infere on an image
    feature, label = sess.run([features, labels])
    feed_dict = {gene_minput: feature}
    gene_output = sess.run(gene_moutput, feed_dict=feed_dict)

    size = [label.shape[1], label.shape[2]]

    nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    bicubic = tf.image.resize_bicubic(feature, size)
    bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    image   = tf.concat(2, [nearest, bicubic, clipped, label])
    #image   = tf.concat(0, [image[0,:,:,:]])
    print("There are ", len(filenames), "files")
    image = tf.concat(0, [image[i,:,:,:] for i in range(len(filenames))])

    image = sess.run(image)

    res_filename = 'check_results.png'
    res_filename = os.path.join(FLAGS.result_dir, res_filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(res_filename)
    print("    Saved result")  
    
