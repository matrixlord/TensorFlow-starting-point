import tensorflow as tf
from images import get_all_image_data, get_test_images

input_n = 40000
# Hidden layers.
l1_n = 1200
l2_n = 1200
l3_n = 1200
l4_n = 1200

# Output layer.
n_classes = 2

# Input.
x = tf.placeholder('float')

# Output.
y = tf.placeholder('float')

# Neural network model.
def create_model(data):
    # First layer.
    hidden_1_layer = {
        'weights': tf.Variable(tf.random_normal([input_n, l1_n])),
        'biases': tf.Variable(tf.random_normal([l1_n]))
    }

    # Second layer.
    hidden_2_layer = {
        'weights': tf.Variable(tf.random_normal([l1_n, l2_n])),
        'biases': tf.Variable(tf.random_normal([l2_n]))
    }

    # Third layer.
    hidden_3_layer = {
        'weights': tf.Variable(tf.random_normal([l2_n, l3_n])),
        'biases': tf.Variable(tf.random_normal([l3_n]))
    }

    # Forth layer.
    hidden_4_layer = {
        'weights': tf.Variable(tf.random_normal([l3_n, l4_n])),
        'biases': tf.Variable(tf.random_normal([l4_n]))
    }

    # Output layer.
    output_layer = {
        'weights': tf.Variable(tf.random_normal([l4_n, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)

    output = tf.matmul(l4, output_layer['weights']) + output_layer['biases']

    return output

# Train model and test accuracy.
def train_model(x):
    # Load images and labels.
    train_data = get_all_image_data()
    train_images = [item[1] for item in train_data]
    train_labels = [item[0] for item in train_data]

    # Create prediction plan.
    prediction = create_model(x)
    # Get the cost.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    # Optimize based on local minimum.
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # Train based on model.
    epochs = 100
    with tf.Session() as sess:
        # Intialize variables.
        sess.run(tf.global_variables_initializer())

        # Run training for each epoch.
        epoch_loss_list = []
        for epoch in range(epochs):
            # Get the epoch loss.
            _, epoch_loss = sess.run([optimizer, cost], feed_dict = {x: train_images, y: train_labels})
            print('Epoch: ' + str(epoch) + ' Epoch loss: ' +  str(epoch_loss))
            # If epoch_loss is zero, break the loop.
            epoch_loss_list.append(epoch_loss)
            if (len(epoch_loss_list) > 5 and len(set(epoch_loss_list[-5:])) == 1):
                break
            # if (epoch_loss == 0):
            #     break

        # Check accuracy.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        test_images = get_test_images()
        for item in test_images:
            # Set [1, 0] for hot dog and [0, 1] for pizza.
            if (accuracy.eval({x: [item[0]], y: [[1, 0]]})):
                print('SocialNerds bot found a Hot Dog!' + '(' + item[1] + ')')
            elif (accuracy.eval({x: [item[0]], y: [[0, 1]]})):
                print('SocialNerds bot found a pizza!' + '(' + item[1] + ')')
            else:
                print('Oops!')
        
        # Print all predictions.
        print('Predictions: ', prediction.eval(feed_dict = {x: [item[0] for item in test_images]}, session = sess))

# Train model and test accuracy.
train_model(x)
