""" The training loop to train the lightweight classifier """

import os
import torch
from torch.optim import Adam
from tqdm import tqdm

from src.utilities.colors import PrintColors
from src.datasets import SplittedDataloader
from src.TrackingModel import store_model
from src.FocalLoss import FocalLoss


def train(directory, model, dataset, training_config):
    """
    Trains the classifier

    :param directory: Directory to store the model and logs
    :param model: The already initiated tracking model
    :param dataset: The already initiated dataset
    :param training_config: Training configuration from the config file
    :return: Training metrics
    """
    ''' Load variables '''
    optimizer_config = training_config.optimizer_config
    drop_factor = training_config.drop_factor
    loss_config = training_config.loss_config
    epochs = training_config.epochs
    accumulate = max(training_config.accumulate_gradients, 1)
    drop_lr_every = training_config.drop_lr_every

    ''' Inititates the optimization framework '''
    assert dataset.sequences_for_training_loop is not None
    print("=== Start Training with sequences ===")
    for seq in dataset.sequences_for_training_loop:
        print("    ", seq["name"])
    model_path = os.path.join(directory, "model.pkl")
    optimizer = Adam(params=model.parameters(), **optimizer_config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=drop_lr_every, gamma=drop_factor)
    loss_function = FocalLoss(**loss_config)

    ''' Run training loop '''
    metrics = {}
    for epoch in range(epochs):
        # Induce that training procedure is running
        os.environ["TRAINING"] = "1"
        # Optimize on all batches
        model.train()
        dataloader = SplittedDataloader(
            dataset=dataset,
            dataset_config=dict(sequences=dataset.sequences_for_training_loop, is_inference=False),
            dataloader_config=dict(shuffle=True, num_workers=0),
            total_parts=35
        )
        step, total_steps, training_loss = 0, len(dataloader), 0.0
        optimizer.zero_grad()
        progress_bar = tqdm(iter(dataloader), desc="Train Epoch {}".format(epoch + 1))
        for data in progress_bar:
            # Skip batches without positive samples
            num_pos_gt, num_gt_edges = data["edges"]["gt_edge_costs"].sum(), data["edges"]["gt_edge_costs"].numel()
            if not 0 < num_pos_gt < num_gt_edges:
                continue
            step += 1
            scheduler.step()
            # Calculate loss
            nodes, edges = data["nodes"], data["edges"]
            costs = model.calculate_costs(nodes, edges)
            output = model.training_step(costs)
            data["prediction"] = output
            loss = loss_function(data)
            loss = loss / accumulate
            loss.backward()
            # Optimize
            if (1 + step) % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
            # Report logs
            training_loss += loss.cpu().detach().numpy()
            report = dict(avg_loss=accumulate * training_loss / step, loss=accumulate * loss.item())
            progress_bar.set_postfix(report)
        # Update of last batch, if not already done
        if (1 + step) % accumulate > 0:
            optimizer.step()
            optimizer.zero_grad()
        # Store model and print logs
        store_model(model_path, model)
        training_loss = accumulate * (training_loss / max(1, step))
        metrics['training_loss'] = training_loss
        report = dict(loss=training_loss)
        info_str = "\t".join([f"{k}={v:.5f}" for k, v in report.items()])
        print(f"\rEpoch: {epoch + 1}: ({step}/{total_steps}), {info_str}", end='\n')

    ''' Report training metrics'''
    print("%sFinished training of model %s" % (PrintColors.OKGREEN, PrintColors.ENDC))
    return metrics




