import torch
import torch.nn as nn

from networks import Conv4


class MAML(nn.Module):

    def __init__(self, num_ways, input_size, T=1, second_order=True, inner_lr=0.4, **kwargs):
        super().__init__()
        self.num_ways = num_ways
        self.input_size = input_size
        self.num_updates = T
        self.second_order = second_order
        self.inner_loss = nn.CrossEntropyLoss()
        self.inner_lr = inner_lr

        self.network = Conv4(self.num_ways, img_size=int(input_size ** 0.5))

    def apply(self, x_supp, y_supp, x_query, y_query, training=False):
        """
        Performs the inner-level learning procedure of MAML: adapt to the given task
        using the support set. It returns the predictions on the query set, as well as the loss
        on the query set (cross-entropy).
        You may want to set the gradients manually for the base-learner parameters 

        :param x_supp (torch.Tensor): the support input images of shape (num_support_examples, num channels, img width, img height)
        :param y_supp (torch.Tensor): the support ground-truth labels
        :param x_query (torch.Tensor): the query inputs images of shape (num_query_inputs, num channels, img width, img height)
        :param y_query (torch.Tensor): the query ground-truth labels

        :returns:
          - query_preds (torch.Tensor): the predictions of the query inputs
          - query_loss (torch.Tensor): the cross-entropy loss on the query inputs
        """
        params = self.network.parameters()
        fast_weights = [param.clone() for param in params]

        for _ in range(self.num_updates):
            preds = self.network.forward(x_supp, weights=fast_weights)

            partial_loss = self.inner_loss(preds, y_supp)

            grad_w = torch.autograd.grad(partial_loss, fast_weights, create_graph=self.second_order)

            fast_weights = [
                weights - self.inner_lr * gradients
                for weights, gradients in zip(fast_weights, grad_w)
            ]

        # outer gradient descent on query set
        query_preds = self.network.forward(x_query, weights=fast_weights)

        query_loss = self.inner_loss(query_preds, y_query)

        if training:
            query_loss.backward()

        return query_preds, query_loss
