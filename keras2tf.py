from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import os
import os.path 

MODEL_NAME='BANGLA_CONVENT'
# Load existing model.
with open("model.json",'r') as f:
    modelJSON = f.read()

model = model_from_json(modelJSON)
model.load_weights("model_weight.hdf5")


# All new operations will be in test mode from now on.
K.set_learning_phase(0)

# Serialize the model and get its weights, for quick re-building.
config = model.get_config()
weights = model.get_weights()

# Re-build a model where the learning phase is now hard-coded to 0.
new_model = Sequential.from_config(config)
def train(new_model,weights):
    new_model.set_weights(weights)


def export_model(saver, model, input_node_names, output_node_name):
    
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


def main():
    
    train(new_model,weights)
    export_model(tf.train.Saver(),new_model, ["conv2d_1_input"], "dense_2/Softmax")


if __name__ == '__main__':
    main()