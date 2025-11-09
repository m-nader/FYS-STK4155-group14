import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        x_data,
        y_data,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der_func,
        L1=False,
        L2=False,
        lmbda=0.0,
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.network_input_size = x_data.shape[1]
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der_func = cost_der_func
        self.layers = self.create_layers()
        self.L1 = L1
        self.L2 = L2
        self.lmbda = lmbda

    def create_layers(self):
        layers = []

        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size)
            b = np.zeros(layer_output_size) + 0.01
            layers.append((W, b))

            i_size = layer_output_size
        return layers

    def predict(self, inputs=None):
        a = self.x_data if inputs is None else inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = np.dot(a, W) + b
            a = activation_func(z)
        return a

    def cost(self):
        predicts = self.predict()
        cost = self.cost_fun(predicts, self.y_data)
        if self.L1:
            L1_sum = 0
            for W, _ in self.layers:
                L1_sum += np.sum(np.abs(W))
            cost += self.lmbda * L1_sum / self.x_data.shape[0]
        if self.L2:
            L2_sum = 0
            for W, _ in self.layers:
                L2_sum += np.sum(W**2)
            cost += self.lmbda * L2_sum / (2 * self.x_data.shape[0])
        return cost

    def cost_der(self, predicts, targets):
        cost_der = self.cost_der_func(predicts, targets)
        return cost_der

    def _feed_forward_saver(self, inputs=None):
        layer_inputs = []
        zs = []
        a = self.x_data if inputs is None else inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = np.dot(a, W) + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a

    def backpropagation(self, inputs=None, targets=None):
        inputs = self.x_data if inputs is None else inputs
        targets = self.y_data if targets is None else targets

        layer_inputs, zs, predicts = self._feed_forward_saver(inputs)
        layer_grads = [() for _ in self.layers]

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input = layer_inputs[i]
            z = zs[i]
            activation_der = self.activation_ders[i]
            W, b = self.layers[i]

            if i == len(self.layers) - 1:
                targets_one_hot = np.eye(10)[targets]
                dC_dz = self.activation_funcs[-1](z) - targets_one_hot
            else:
                # For other layers we build on previous z derivative: dC_da_prev = dC_dz @ W_next.T
                next_W, _ = self.layers[i + 1]
                dC_da = dC_dz @ next_W.T                  
                dC_dz = dC_da * activation_der(z)
                # print(next_W.shape, dC_dz.shape)
                # dC_da = next_W @ dC_dz.T
                # dC_dz = activation_der(z) * dC_da.T
            dC_dW = layer_input.T @ dC_dz
            batch_size = layer_input.shape[0]
            dC_dW = dC_dW / batch_size
            # add regularization to the weight gradient (use full dataset size to match cost())
            n = self.x_data.shape[0]
            if self.L2:
                # cost() uses lambda * sum(W^2) / (2n) so gradient is lambda * W / n
                dC_dW = dC_dW + (self.lmbda / n) * W
            if self.L1:
                # cost() used lambda * sum(|W|) / (2n) so gradient is lambda * sign(W) / (2n)
                dC_dW = dC_dW + (self.lmbda / n) * np.sign(W)

            # return bias gradient as 1D array (matches stored b shape)
            dC_db = np.mean(dC_dz, axis=0)



            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def update_weights(self, layer_grads, learning_rate=0.001):
        for idx, ((W, b), (W_g, b_g)) in enumerate(
            zip(self.layers, layer_grads)
        ):
            W = W - learning_rate * W_g
            b = b - learning_rate * b_g
            self.layers[idx] = (W, b)

    def update_weights_RMSProp(
        self, layer_grads, G_iter, learning_rate=0.001, delta=1e-8, rho=0.9
    ):
        for idx, ((W, b), (W_g, b_g)) in enumerate(
            zip(self.layers, layer_grads)
        ):
            G_iter[idx] = (
                rho * G_iter[idx][0] + (1 - rho) * W_g**2,
                rho * G_iter[idx][1] + (1 - rho) * b_g**2,
            )
            W = W - learning_rate * W_g / (np.sqrt(G_iter[idx][0]) + delta)
            b = b - learning_rate * b_g / (np.sqrt(G_iter[idx][1]) + delta)
            self.layers[idx] = (W, b)

    def update_weights_ADAM(
        self,
        layer_grads,
        first_moment,
        second_moment,
        i,
        learning_rate=0.001,
        delta=1e-8,
        beta1=0.9,
        beta2=0.999,
    ):
        for idx, ((W, b), (W_g, b_g)) in enumerate(
            zip(self.layers, layer_grads)
        ):
            first_moment[idx] = (
                beta1 * first_moment[idx][0] + (1 - beta1) * W_g,
                beta1 * first_moment[idx][1] + (1 - beta1) * b_g,
            )
            second_moment[idx] = (
                beta2 * second_moment[idx][0] + (1 - beta2) * W_g**2,
                beta2 * second_moment[idx][1] + (1 - beta2) * b_g**2,
            )
            first_term_corr_W = first_moment[idx][0] / (1 - beta1**i)
            first_term_corr_b = first_moment[idx][1] / (1 - beta1**i)
            second_term_corr_W = second_moment[idx][0] / (1 - beta2**i)
            second_term_corr_b = second_moment[idx][1] / (1 - beta2**i)
            update_W = (
                learning_rate
                * first_term_corr_W
                / (np.sqrt(second_term_corr_W) + delta)
            )
            update_b = (
                learning_rate
                * first_term_corr_b
                / (np.sqrt(second_term_corr_b) + delta)
            )
            W = W - update_W
            b = b - update_b
            self.layers[idx] = (W, b)
        

    def train_network_stochastic_gd(
        self,
        learning_rate=0.001,
        epochs=100,
        minibatch_size=128,
        lr_method=None,
        delta=1e-8,
        rho=0.9,
        beta1=0.9,
        beta2=0.999,
    ):
        if lr_method == "RMSProp":
            G_iter = [
                (np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.layers
            ]
        elif lr_method == "ADAM":
            first_moment = [
                (np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.layers
            ]
            second_moment = [
                (np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.layers
            ]
        change = [
            (np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.layers
        ]
        n_data = self.x_data.shape[0]
        m = int(n_data / minibatch_size)
        for epoch in range(epochs):
            epoch += 1
            indices = np.random.permutation(n_data)
            x_shuffled = self.x_data[indices]
            y_shuffled = self.y_data[indices]
            for i in range(m):
                xi = x_shuffled[i : i + minibatch_size]
                yi = y_shuffled[i : i + minibatch_size]
                layer_grads = self.backpropagation(xi, yi)
                prev_layers = self.layers.copy()
                if lr_method == "RMSProp":
                    self.update_weights_RMSProp(
                        layer_grads, G_iter, learning_rate, delta, rho
                    )
                elif lr_method == "ADAM":
                    self.update_weights_ADAM(
                        layer_grads,
                        first_moment,
                        second_moment,
                        epoch,
                        learning_rate,
                        delta,
                        beta1,
                        beta2,
                    )
                else:
                    self.update_weights(layer_grads, learning_rate)

    def train_network_plain_gd(
        self,
        learning_rate=0.001,
        max_iter=1000,
        stopping_criteria=1e-10,
        lr_method=None,
        delta=1e-8,
        rho=0.9,
        beta1=0.9,
        beta2=0.999,
    ):
        if lr_method == "RMSProp":
            G_iter = [
                (np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.layers
            ]
        elif lr_method == "ADAM":
            first_moment = [
                (np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.layers
            ]
            second_moment = [
                (np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.layers
            ]
        last_cost = None
        for i in range(max_iter):
            i += 1
            layer_grads = self.backpropagation()
            # prev_layers = self.layers.copy()
            if lr_method == "RMSProp":
                self.update_weights_RMSProp(
                    layer_grads, G_iter, learning_rate, delta, rho
                )
            elif lr_method == "ADAM":
                self.update_weights_ADAM(
                    layer_grads,
                    first_moment,
                    second_moment,
                    i,
                    learning_rate,
                    delta,
                    beta1,
                    beta2,
                )
            else:
                self.update_weights(layer_grads, learning_rate)
            # Compute previous and current cost and stop if improvement below threshold
            # prev_layers was the network state before the weight update

                if i % 50 == 0:  # compute cost every 50 iters
                    curr_cost = self.cost()
                    if last_cost is not None and abs(last_cost - curr_cost) <= stopping_criteria:
                        print(f"Early stopping at iter {i}, delta {abs(last_cost - curr_cost):.2e}")
                        break
                    last_cost = curr_cost



