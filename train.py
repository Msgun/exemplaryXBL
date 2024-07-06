from keras.callbacks import EarlyStopping,ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import helpers
import prepare_data

class CustomModel(keras.Model):
    global grad_model
    
    def train_step(self, data):
        x, y, x_exp = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self._compute_loss(x, y, y_pred, x_exp)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def _compute_loss(self, x, y, y_pred, x_exp):
        class_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        gcam =  self._compute_gcam(x, y_pred) 
        if(x_exp.dtype != tf.float32):
            x_exp = tf.cast(x_exp, tf.float32)
        gcam = tf.multiply(gcam, x_exp)
        
        gcam = tf.reshape(gcam, [batch_size, 49])
        
        gg_distance = mse(gcam, good_e)
        gb_distance = mse(gcam, bad_e)
        triplet = tf.math.subtract(gg_distance, gb_distance) + tf.constant(1.)
        triplet_loss = tf.maximum(triplet, 0.0)
        
        valid_triplets = tf.math.greater(triplet_loss, 0)
        loss = tf.reduce_sum(triplet_loss)/(tf.reduce_sum(tf.cast(valid_triplets, tf.float32))+1e-16)
        
        loss = tf.math.multiply(loss, exp_rate)
        exp_loss = tf.math.divide(loss, batch_size)
        
        exp_loss = tf.math.divide(loss, batch_size)
        loss = tf.math.add(class_loss, exp_loss)
        return loss
    
    def _compute_gcam(self, x, y_pred):
        layer = 153        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model([x])
            loss = predictions[0]

        output = conv_outputs[0]

        grads = tape.gradient(loss, conv_outputs)[0]
        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        return cam
def train_model(model, good_exp, bad_exp):
    global grad_model
    
    EarlyStopping = EarlyStopping(monitor='val_categorical_accuracy',
                                  min_delta=.001,
                                  patience=6,
                                  verbose=1,
                                  mode='auto',
                                  baseline=None,
                                  restore_best_weights=True)
    rlr = ReduceLROnPlateau( monitor="val_categorical_accuracy",
                                factor=0.1,
                                patience=4,
                                verbose=0,
                                mode="max",
                                min_delta=0.01)
    model_save = ModelCheckpoint('./models/refined_model',
                                 save_best_only = True,
                                 save_weights_only = False,
                                 monitor = 'val_categorical_accuracy', 
                                 mode = 'max', verbose = 1)
    
    model = tf.keras.models.load_model('./models/resnet_scratch', compile=False)
    model = CustomModel(model.input, model.output)
    grad_model = tf.keras.models.Model([model.input], [model.layers[153].output, m.output])
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                  metrics=['categorical_accuracy'])
    
    train_dataset, val_dataset, test_dataset = prepare_data.prepare_data()
    
    history = model.fit(train_dataset, validation_data = val_dataset, epochs=100,
                   callbacks=[EarlyStopping, model_save, rlr])
    
    return train_history

def main():
    good_exp, bad_exp = helpers.get_exemplary_exps()

    train_history = train_model(model, good_exp, bad_exp)
    
if __name__ == "__main__":
    main()