def train_loop(model, *args, loss_fn=None, optimizer=None):
    """
    Implements a training loop mostly for unit tests.

    :param model: a model
    :param args: its inputs
    :param loss_fn: a loss function, `MSELoss` if not specified
    :param optimizer: an optimizer, `SGD` it not specified
    :return: gradients
    """
    import torch

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    # Compute prediction and loss
    pred = model(*args)
    if isinstance(pred, tuple):
        v = pred[0]
    elif hasattr(pred, "last_hidden_state"):
        v = pred.last_hidden_state
    else:
        v = pred
    loss = loss_fn(v, torch.ones_like(v))

    # Backpropagation
    loss.backward()
    optimizer.step()
    # skip that part to retrieve the gradients
    # optimizer.zero_grad()

    # returns the gradients
    res = tuple(p.grad for p in model.parameters() if p.grad is not None)
    assert len(res) > 0, f"No gradient, loss is {loss}"
    return res


def train_loop_mixed_precision(model, *args, loss_fn=None, optimizer=None):
    """
    Implements a training loop mostly for unit tests
    using mixed precision.

    :param model: a model
    :param args: its inputs
    :param loss_fn: a loss function, `MSELoss` if not specified
    :param optimizer: an optimizer, `SGD` it not specified
    :return: gradients
    """
    import torch

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()

        # Compute prediction and loss
        pred = model(*args)
        if isinstance(pred, tuple):
            v = pred[0]
        elif hasattr(pred, "last_hidden_state"):
            v = pred.last_hidden_state
        else:
            v = pred
        loss = loss_fn(v, torch.ones_like(v))

        # Backpropagation
        loss.backward()
        optimizer.step()
        # skip that part to retrieve the gradients
        # optimizer.zero_grad()

    # returns the gradients
    res = tuple(p.grad for p in model.parameters() if p.grad is not None)
    assert len(res) > 0, f"No gradient, loss is {loss}"
    return res
