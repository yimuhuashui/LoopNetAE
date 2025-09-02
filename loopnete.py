# -- coding: gbk --
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import resample
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv1D
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import Activation

from attention import EnhancedTemporalAttention

class loopnet():
    def __init__(self, learning_rate, epochs, train_x, train_y, test_x, test_y, chromname, kernel_size, save_dir):
        """
        Initialize loopnet model
        :param save_dir: Directory for saving model files
        """
        self.rate = learning_rate
        self.epochs = epochs
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.chromname = chromname
        self.kernel_size = kernel_size
        self.save_dir = save_dir  # Store the save directory

    def evaluateModel(self, trainX, trainY):
        input_shape = (trainX.shape[1], trainX.shape[2])
        model = Sequential()

        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.5))
        
        model.add(Conv1D(64, kernel_size=self.kernel_size, padding='same', activation='relu', input_shape=input_shape,
            kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.5))
        
        model.add(Conv1D(32, kernel_size=self.kernel_size, padding='same', activation='relu', input_shape=input_shape,
            kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.5))
        
        model.add(EnhancedTemporalAttention()) 
        
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))) 
        model.add(Dense(2))
        model.add(Activation('softmax'))  
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        model.summary()
        return model


    def train_model(self):
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,     # Halve the learning rate
            patience=3,     # Reduce LR after 3 epochs without improvement
            min_lr=1e-6     # Minimum learning rate
        )  

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,            # Increased from 3 to 10, giving model more recovery opportunity
            min_delta=0.003,        # Increased from 0.001 to 0.005, ignoring minor fluctuations
            mode='min',             # Explicitly monitor minimum value
            restore_best_weights=True,
            verbose=1,  
        )    
    
        n_split = 10
        scores = []
        for m in range(n_split):
            ix = [i for i in range(len(self.train_x))]
            train_ix = resample(ix, replace=True)
            trainX, trainY = self.train_x[train_ix], self.train_y[train_ix]
            model = self.evaluateModel(trainX, trainY)
            model.fit(
                trainX, 
                trainY, 
                epochs=self.epochs, 
                batch_size=128, 
                callbacks=[early_stop, lr_scheduler],
                validation_data=(self.test_x, self.test_y)
            )
            # Use the passed save path
            save_path = self.save_dir + self.chromname + '_' + str(m) + '.h5'
            model.save(save_path)
            scores.append(model.evaluate(self.test_x, self.test_y, verbose=0))
        return scores