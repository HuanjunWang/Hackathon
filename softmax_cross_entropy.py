import tensorflow as tf

labels = [[0.2, 0.3, 0.5],
          [0.1, 0.6, 0.3]]
logits = [[2, 0.5, 1],
          [0.1, 1, 3]]
logits_scaled = tf.nn.softmax(logits)

labels_scaled = tf.nn.softmax(labels)

test_logits = [[[2, 0.5, 1],
                [0.1, 1, 3]],
               [[2, 0.5, 1],
                [0.1, 1, 3]]]

test_logits_scaled = tf.nn.softmax(test_logits)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)
result3 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels_scaled, logits=logits_scaled)

with tf.Session() as sess:
    # print(sess.run(test_logits_scaled))
    print(sess.run(logits_scaled))
    print(sess.run(result1))
    print(sess.run(result2))
    print(sess.run(result3))
    print(sess.run(result4))
