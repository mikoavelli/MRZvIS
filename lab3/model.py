# Индивидуальная лабораторная работа 1 по дисциплине МРЗвИС вариант 6
# "Реализовать модель линейной рециркуляционной сети с адаптивным коэффициентом обучения с ненормированными весами"
# Выполнена студентом группы 221702 БГУИР Целуйко Дмитрием Александровичем
# Файл реализации линейной рециркуляционной сети
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач. Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import numpy as np


class ImageReconstructorNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.inputs = None
        self.hidden_states = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        init_range = 1.0
        self.W_ih = np.random.uniform(-init_range, init_range, (hidden_size, input_size)).astype(np.float32)
        self.W_ho = np.random.uniform(-init_range, init_range, (output_size, hidden_size)).astype(np.float32)

        self.m, self.v = {}, {}
        for param in ['W_ih', 'W_ho']:
            self.m[param] = np.zeros_like(getattr(self, param))
            self.v[param] = np.zeros_like(getattr(self, param))

        self.beta1, self.beta2, self.epsilon, self.t = 0.9, 0.999, 1e-8, 0

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_states = np.dot(inputs, self.W_ih.T)

        outputs = np.dot(self.hidden_states, self.W_ho.T)

        return outputs

    def backward(self, outputs, targets, compute_loss=False):
        N = self.inputs.shape[0]
        error = outputs - targets

        total_loss = 0.0
        if compute_loss:
            total_loss = np.mean(error ** 2) * self.input_size

        dW_ho = np.dot(error.T, self.hidden_states)

        dh = np.dot(error, self.W_ho)

        dW_ih = np.dot(dh.T, self.inputs)

        return {
            'W_ih': dW_ih / N,
            'W_ho': dW_ho / N
        }, total_loss

    def update_weights_adam(self, grads, learning_rate):
        self.t += 1

        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t

        for key in ['W_ih', 'W_ho']:
            param = getattr(self, key)
            grad = grads[key]

            m = self.m[key]
            v = self.v[key]

            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            m_corr = m / correction1
            v_corr = v / correction2

            param -= learning_rate * m_corr / (np.sqrt(v_corr) + self.epsilon)

    def train_step(self, inputs, targets, learning_rate, compute_loss=False):
        outputs = self.forward(inputs)
        grads, loss = self.backward(outputs, targets, compute_loss)
        self.update_weights_adam(grads, learning_rate)
        return loss

    def train(self, inputs, targets, config):
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)

        max_epochs = config.get('max_epochs', 10000)
        target_loss = config.get('target_loss', 0.0)
        initial_lr = config['initial_lr']
        log_frequency = config['log_frequency']

        enable_scheduler = config.get('enable_scheduler', False)
        patience_threshold = config.get('lr_scheduler_patience_epochs', 20)
        factor = config['lr_factor']
        min_lr = config['min_lr']

        patience_counter = 0
        best_loss = float('inf')
        loss = float('inf')
        current_lr = initial_lr

        print(f"Starting optimized training (float32). Target Loss: {target_loss}...")

        for epoch in range(max_epochs):
            loss = self.train_step(inputs, targets, current_lr, compute_loss=True)

            if loss <= target_loss:
                print(f"\n✅ Target loss {target_loss} reached at epoch {epoch}! Loss: {loss:.6f}")
                break

            if epoch % log_frequency == 0:
                print(f'Epoch {epoch}, Loss: {loss:.6f}, LR: {current_lr:.6f}')

            if enable_scheduler:
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience_threshold:
                    new_lr = current_lr * factor
                    if new_lr >= min_lr:
                        print(f"--- No new best loss for {patience_threshold} epochs. Reducing LR to {new_lr:.6f} ---")
                        current_lr = new_lr
                        patience_counter = 0
        else:
            print(f"\n⚠️ Max epochs ({max_epochs}) reached. Final Loss: {loss:.6f}")
