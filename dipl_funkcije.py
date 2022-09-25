import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
import os
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# roc_auc multi class OvR
def roc_auc_score_mc(true_class, pred_class, average):
  classes = set(true_class)
  roc_auc_dict = {}
  for one_class in classes:
    other_classes = [x for x in classes if x != one_class]
    true_classes = [0 if x in other_classes else 1 for x in true_class]
    pred_classes = [0 if x in other_classes else 1 for x in pred_class]
    roc_auc = roc_auc_score(true_classes, pred_classes, average = average)
    roc_auc_dict[one_class] = roc_auc

  return roc_auc_dict


# roc_auc_curve multi class OvR
def roc_auc_curve_mc(true_class, pred_class, classes):

  from sklearn import metrics

  j_classes = set(true_class)
  roc_auc_dict = {}
  for one_class in j_classes:
    other_classes = [x for x in j_classes if x != one_class]
    true_classes = [0 if x in other_classes else 1 for x in true_class]
    pred_classes = [0 if x in other_classes else 1 for x in pred_class]

    fpr, tpr, threshold = metrics.roc_curve(true_classes, pred_classes)
    roc_auc = metrics.auc(fpr, tpr)

    plt.rcParams["figure.figsize"] = (12,10)
    plt.title(classes[one_class])
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# loss/accuracy per epoch krivulje
def plot_results(results):

    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    train_accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy per epoch")
    plt.xlabel("Epoch")
    plt.plot(epochs, train_accuracy, label="Train accuracy")
    plt.plot(epochs, test_accuracy, label="Test accuracy")
    plt.legend()

# PROVJERENO I KOPIRANO DO OVDJE 
    
# preuzeto iz going_modular; original dostupan na: 
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/engine.py
#
# dodano vraćanje liste stvarnih i predicted klasa

from sklearn.metrics import f1_score, roc_auc_score, classification_report
import numpy as np

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.
    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.
    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:
    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        epoch_true_labels = []
        epoch_pred_labels = []
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
            # rješenje "tensor ili numpy array" zavrzlame
            if device == 'cuda':
              epoch_true_labels.append(y.cpu())
              epoch_pred_labels.append(test_pred_labels.cpu())
            else:
              epoch_true_labels.append(y)
              epoch_pred_labels.append(test_pred_labels)
              
    epoch_true_labels = np.concatenate(epoch_true_labels)
    epoch_pred_labels = np.concatenate(epoch_pred_labels)
    
    #print(classification_report(epoch_true_labels, epoch_pred_labels))
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    #f1score = f1_score(epoch_true_labels, epoch_pred_labels, average = 'macro')
    
    return test_loss, test_acc, epoch_true_labels, epoch_pred_labels    
    
from typing import Dict, List
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


# preuzeto iz going_modular; original dostupan na: 
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/train.py
# dodano early stopping, classification report, roc_auc
# dodano proslijeđivanje best_epoch_true i best_epoch_pred 

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          es_patience: int,
          best_model: str,
          labels: list,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      
    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
      For example if training for epochs=2: 
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [],
               #"test_f1_score": []
               #"test_roc_auc": []
    }

    roc_auc = {}
    best_accuracy = 0; # najbolji acc 
    es_counter = 1; # brojač epoha bez porasta acc 
    best_epoch_true, best_epoch_pred = [], []

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc, epoch_true, epoch_pred = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)


        # Print out what's happening
        print ('============================================================================================')
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f} "
          #f"test_f1_score: {f1score:.4f} "
        )
        
        # ubačeno za early stopping

        if    test_acc > best_accuracy: 
              torch.save(model.state_dict(), best_model)
              print('---------------------------------------------------------------------------------------------')
              print('Epoch ', (epoch + 1), '| Best model saved in ', best_model) 
              best_accuracy = test_acc
              es_counter = 1
              best_epoch_true = epoch_true
              best_epoch_pred = epoch_pred
              print(classification_report(y_true=epoch_true, y_pred=epoch_pred, target_names=labels, zero_division=0))
              roc_auc = roc_auc_score_mc(epoch_true, epoch_pred, average = 'macro')
              print('{:>12}  {:>9}'.format("", "ROC_AUC (OvR)"))

              #for l , v in roc_auc.items(): 
              #    print ('{:>12}  {:>9}'.format(labels[l], round(v, 4)))
              for l , v in roc_auc.items(): 
                  print ('{:>12}  {:>9}'.format(labels[l], round(v, 4)))
        else:
              es_counter = es_counter + 1      


        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        #results["test_f1_score"].append(f1score)
        #results["test_roc_auc"].append(roc_auc)
       
        # provjera za early stopping

        if es_counter > es_patience:
            print ('Test accuracy not improving for ', es_patience ,' epochs - early stopping.')
            print ('Best model saved in ', best_model)
            print ('Best test accuracy: ', best_accuracy)
            break
        
    return results, best_epoch_true, best_epoch_pred    
