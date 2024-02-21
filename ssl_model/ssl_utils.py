import numpy as np
import torch
import torch.nn.functional as F

def inference_by_SSFL_model(SSFL_model, dataloader, device):
    feature_vector = []
    labels_vector = []
    SSFL_model.to(device)
    SSFL_model.eval()
    for step, (x, y) in enumerate(dataloader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = SSFL_model(x)

        h = h.squeeze()
        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector



def create_data_loaders_from_arrays(X, y, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y)
    )
    embedding_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )
    return embedding_loader


def info_nce_loss( features, batch_size, device, n_views=2, temperature=0.07):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     n_views * self.conf.batch_size, n_views * self.conf.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels