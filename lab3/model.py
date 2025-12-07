import numpy as np


class ImageReconstructorNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.inputs = None
        self.hidden_states = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        init_range = 1.0
        self.W_ih = np.random.uniform(
            -init_range, init_range, (hidden_size, input_size)
        ).astype(np.float32)
        self.W_ho = np.random.uniform(
            -init_range, init_range, (output_size, hidden_size)
        ).astype(np.float32)

        self.m, self.v = {}, {}
        for param in ["W_ih", "W_ho"]:
            self.m[param] = np.zeros_like(getattr(self, param))
            self.v[param] = np.zeros_like(getattr(self, param))
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.t = 0

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_states = np.dot(inputs, self.W_ih.T)
        outputs = np.dot(self.hidden_states, self.W_ho.T)
        return outputs

    def backward(self, outputs, targets, compute_loss=False):
        """
        SSE (Sum of Squared Errors).
        Градиенты не усредняются.
        """
        error = outputs - targets

        total_loss = 0.0
        if compute_loss:
            # ТРЕБОВАНИЕ: Сумма квадратичных ошибок
            total_loss = np.sum(error**2)

        # Вычисляем градиенты
        dW_ho = np.dot(error.T, self.hidden_states)
        dh = np.dot(error, self.W_ho)
        dW_ih = np.dot(dh.T, self.inputs)

        # ТРЕБОВАНИЕ: Нет деления на N. Чистая сумма градиентов.
        return {"W_ih": dW_ih, "W_ho": dW_ho}, total_loss

    def update_weights_adam(self, grads, learning_rate):
        self.t += 1
        correction1 = 1 - self.beta1**self.t
        correction2 = 1 - self.beta2**self.t

        for key in ["W_ih", "W_ho"]:
            param = getattr(self, key)
            grad = grads[key]
            m = self.m[key]
            v = self.v[key]

            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad**2)

            m_corr = m / correction1
            v_corr = v / correction2

            param -= learning_rate * m_corr / (np.sqrt(v_corr) + self.epsilon)

    def train_step(self, inputs, targets, learning_rate, compute_loss=False):
        # inputs здесь - это ОДИН z (размер 1x192)
        outputs = self.forward(inputs)
        grads, loss = self.backward(outputs, targets, compute_loss)
        self.update_weights_adam(grads, learning_rate)
        return loss

    def train(self, inputs, targets, config):
        """
        Цикл обучения с по-чанковым проходом (SGD).
        """
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)

        max_epochs = config.get("max_epochs", 1000)
        target_loss = config.get("target_loss", 100.0)
        initial_lr = config["initial_lr"]
        log_frequency = config["log_frequency"]

        enable_scheduler = config.get("enable_scheduler", False)
        patience_threshold = config.get("lr_scheduler_patience_epochs", 5)
        factor = config["lr_factor"]
        min_lr = config["min_lr"]

        patience_counter = 0
        best_loss = float("inf")
        current_lr = initial_lr

        num_chunks = inputs.shape[0]

        print(
            f"Starting Pure SGD (One chunk update). Total chunks: {num_chunks}, LR: {current_lr}"
        )

        for epoch in range(max_epochs):
            # Перемешиваем индексы, чтобы порядок подачи чанков был случайным (Stochastic)
            indices = np.arange(num_chunks)
            np.random.shuffle(indices)

            epoch_loss_accum = 0.0

            # --- ТРЕБОВАНИЕ: ГЛАВНЫЙ ЦИКЛ ПО ЧАНКАМ ---
            # Мы идем по одному чанку за раз.
            for i in indices:
                # Берем срез [i:i+1], чтобы сохранить размерность (1, 192)
                single_chunk_in = inputs[i : i + 1]
                single_chunk_target = targets[i : i + 1]

                # Этот вызов делает forward -> backward -> update ДЛЯ ОДНОГО ЧАНКА
                loss = self.train_step(
                    single_chunk_in, single_chunk_target, current_lr, compute_loss=True
                )

                epoch_loss_accum += loss

            # Проверки делаем раз в эпоху (суммируя ошибки всех чанков)
            if epoch_loss_accum <= target_loss:
                print(
                    f"\nTarget SSE loss {target_loss} reached at epoch {epoch}! Loss: {epoch_loss_accum:.2f}"
                )
                break

            if epoch % log_frequency == 0:
                print(
                    f"Epoch {epoch}, Total SSE Loss: {epoch_loss_accum:.2f}, LR: {current_lr:.8f}"
                )

            if enable_scheduler:
                if epoch_loss_accum < best_loss:
                    best_loss = epoch_loss_accum
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience_threshold:
                    new_lr = current_lr * factor
                    if new_lr >= min_lr:
                        print(f"--- Reducing LR to {new_lr:.8f} ---")
                        current_lr = new_lr
                        patience_counter = 0
