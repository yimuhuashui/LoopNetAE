from keras.layers import Layer
from keras import backend as K

class EnhancedTemporalAttention(Layer):
    def __init__(self, return_attention=False, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        self.timesteps = input_shape[1]
        self.features = input_shape[2]
        
        # Dual-path attention mechanism
        self.W_query = self.add_weight(
            name='W_query', 
            shape=(self.features, self.features),
            initializer='glorot_normal',
            trainable=True
        )
        self.W_key = self.add_weight(
            name='W_key', 
            shape=(self.features, self.features),
            initializer='glorot_normal',
            trainable=True
        )
        self.W_value = self.add_weight(
            name='W_value', 
            shape=(self.features, self.features),
            initializer='glorot_normal',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # Compute Q, K, V - Using different variable names to avoid conflicts
        Q_tensor = K.dot(x, self.W_query)  # (batch, timesteps, features)
        K_tensor = K.dot(x, self.W_key)    # Using K_tensor instead of K
        V_tensor = K.dot(x, self.W_value)   # (batch, timesteps, features)
        
        # Compute attention scores
        attn_scores = K.batch_dot(Q_tensor, K_tensor, axes=[2,2]) / K.sqrt(K.cast(self.features, 'float32'))
        attn_scores = K.softmax(attn_scores, axis=-1)  # (batch, timesteps, timesteps)
        
        # Apply attention
        weighted = K.batch_dot(attn_scores, V_tensor)  # (batch, timesteps, features)
        
        # Residual connection
        weighted += x
        
        # Global average pooling
        output = K.mean(weighted, axis=1)  # (batch, features)

        # Global average pooling (fixed output)
        return K.mean(weighted, axis=1)  # (batch, features)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])  # Fixed output (batch, features)    

      