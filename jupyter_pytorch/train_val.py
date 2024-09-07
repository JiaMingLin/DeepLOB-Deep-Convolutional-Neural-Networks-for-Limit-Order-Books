from common import *
def train(model, criterion, optimizer, train_loader, val_loader, exp_name, epochs):
    
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in range(epochs):
        print(f"Traning... Epoch:{it+1}/{epochs}")
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in tqdm(train_loader):
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            # print("inputs.shape:", inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            # print("about to get model output")
            # print(inputs.shape) torch.Size([64, 1, 100, 40])
            outputs = model(inputs.squeeze())
            # print("done getting model output")
            # print("outputs.shape:", outputs.shape, "targets.shape:", targets.shape)
            loss = criterion(outputs, targets)
            # Backward and optimize
            # print("about to optimize")
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss) # a little misleading

        print("Validation...")        
        test_loss = test_model(model, criterion, val_loader)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        
        if test_loss < best_test_loss:
            torch.save(model.state_dict(), f'./best_val_model_{exp_name}.pt')
            best_test_loss = test_loss
            best_test_epoch = it+1
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, test_losses

def test_model(model,criterion, test_loader):
    model.eval()
    all_targets = []
    all_predictions = []
    test_loss = []

    for inputs, targets in tqdm(test_loader):
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs.squeeze())

        loss = criterion(outputs, targets)
        test_loss.append(loss.item())
    
        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    test_loss = np.mean(test_loss)
    all_targets = np.concatenate(all_targets)    
    all_predictions = np.concatenate(all_predictions)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')
    print(f'accuracy_score: {accuracy:.4f}, precision_score: {precision:.4f}, recall_score: {recall:.4f}, f1_score: {f1:.4f}')

    # print(classification_report(all_targets, all_predictions, digits=4))

    return test_loss
