import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def feature_inputs(filenames, image_size=None):

    if image_size is None:
        image_size = FLAGS.sample_size

    # Read each JPEG file
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    channels = 3
    image = tf.image.decode_jpeg(value, channels=channels, name="coco_fake_image")
    image.set_shape([None, None, channels])

    # Crop and other random augmentations
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, .95, 1.05)
    #image = tf.image.random_brightness(image, .05)
    #image = tf.image.random_contrast(image, .95, 1.05)

    #wiggle = 8
    #off_x, off_y = 25-wiggle, 60-wiggle
    #crop_size = 128
    #crop_size_plus = crop_size + 2*wiggle
    #image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
    #image = tf.random_crop(image, [crop_size, crop_size, 3])

    #image = tf.reshape(image, [1, crop_size, crop_size, 3])
    #image = tf.cast(image, tf.float32)/255.0

    #if crop_size != image_size:
    #    image = tf.image.resize_area(image, [image_size, image_size])

    image = tf.image.crop_to_bounding_box(image, 50, 50, image_size, image_size)

    image = tf.reshape(image, [1, image_size, image_size, 3])
    image = tf.cast(image, tf.float32)/255.0

    # The feature is simply a Kx downscaled version

    K = 4
    downsampled = tf.image.resize_area(image, [image_size//K, image_size//K])

    feature = tf.reshape(downsampled, [image_size//K, image_size//K, 3])

    '''
    image = tf.reshape(image, [1, 256,256, 3])
    image = tf.cast(image, tf.float32)/255.0

    image = tf.image.resize_area(image, [64, 64])
    feature = tf.reshape(image, [64, 64, 3])
    '''
    return feature

def label_inputs(labelnames, image_size=None):

    if image_size is None:
        image_size = FLAGS.sample_size

    # Read each JPEG file
    reader = tf.WholeFileReader()
    labelname_queue = tf.train.string_input_producer(labelnames)
    key, value = reader.read(labelname_queue)
    channels = 3
    image = tf.image.decode_jpeg(value, channels=channels, name="coco_real_image")
    image.set_shape([None, None, channels])

    #wiggle = 8
    #off_x, off_y = 25-wiggle, 60-wiggle
    #crop_size = 128
    #crop_size_plus = crop_size + 2*wiggle
    #image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
    #image = tf.random_crop(image, [crop_size, crop_size, 3])

    #image = tf.reshape(image, [1, crop_size, crop_size, 3])
    #image = tf.cast(image, tf.float32)/255.0

    #if crop_size != image_size:
    #    image = tf.image.resize_area(image, [image_size, image_size])

    #label = tf.reshape(image, [64, 64, 3])

    image = tf.image.crop_to_bounding_box(image, 50, 50, image_size, image_size)

    image = tf.reshape(image, [1, image_size, image_size, 3])
    image = tf.reshape(image, [image_size, image_size, 3])
    label = tf.cast(image, tf.float32)/255.0

    return label

def setup_inputs(sess, filenames, labelnames, image_size=None, capacity_factor=3):

    if image_size is None:
        image_size = FLAGS.sample_size

    feature = feature_inputs(filenames)
    label = label_inputs(labelnames)

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)
      
    return features, labels
