import numpy as np


class ImageReconstructorRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_ih = np.random.randn(hidden_size, input_size) * np.sqrt(1. / input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1. / hidden_size)
        self.W_ho = np.random.randn(output_size, hidden_size) * np.sqrt(1. / hidden_size)

        self.b_h = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((output_size, 1))

        self.m, self.v = {}, {}
        for param in ['W_ih', 'W_hh', 'W_ho', 'b_h', 'b_o']:
            self.m[param] = np.zeros_like(getattr(self, param))
            self.v[param] = np.zeros_like(getattr(self, param))

        self.beta1, self.beta2, self.epsilon, self.t = 0.9, 0.999, 1e-8, 0

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        outputs = []
        for i in range(len(inputs)):
            x = inputs[i].reshape(-1, 1)
            self.hidden_states[i] = np.tanh(
                np.dot(self.W_ih, x) + np.dot(self.W_hh, self.hidden_states[i - 1]) + self.b_h)
            output = np.dot(self.W_ho, self.hidden_states[i]) + self.b_o
            outputs.append(output)
        return outputs

    def backward(self, outputs, targets):
        dW_ih, dW_hh, dW_ho = np.zeros_like(self.W_ih), np.zeros_like(self.W_hh), np.zeros_like(self.W_ho)
        db_h, db_o = np.zeros_like(self.b_h), np.zeros_like(self.b_o)
        dh_next = np.zeros_like(self.hidden_states[0])
        total_loss = 0

        for i in reversed(range(len(self.inputs))):
            error = outputs[i] - targets[i].reshape(-1, 1)
            total_loss += np.sum(error ** 2)
            dW_ho += np.dot(error, self.hidden_states[i].T)
            db_o += error
            dh = np.dot(self.W_ho.T, error) + dh_next
            dh_raw = dh * (1 - self.hidden_states[i] ** 2)
            dW_hh += np.dot(dh_raw, self.hidden_states[i - 1].T)
            dW_ih += np.dot(dh_raw, self.inputs[i].reshape(1, -1))
            db_h += dh_raw
            dh_next = np.dot(self.W_hh.T, dh_raw)

        for dparam in [dW_ih, dW_hh, dW_ho, db_h, db_o]:
            np.clip(dparam, -5, 5, out=dparam)

        return {'W_ih': dW_ih, 'W_hh': dW_hh, 'W_ho': dW_ho, 'b_h': db_h, 'b_o': db_o}, total_loss / len(self.inputs)

    def update_weights_adam(self, grads, learning_rate):
        self.t += 1
        for key in ['W_ih', 'W_hh', 'W_ho', 'b_h', 'b_o']:
            param = getattr(self, key)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            m_corr = self.m[key] / (1 - self.beta1 ** self.t)
            v_corr = self.v[key] / (1 - self.beta2 ** self.t)
            param -= learning_rate * m_corr / (np.sqrt(v_corr) + self.epsilon)

    def train_step(self, inputs, targets, learning_rate):
        outputs = self.forward(inputs)
        grads, loss = self.backward(outputs, targets)
        self.update_weights_adam(grads, learning_rate)
        return loss
