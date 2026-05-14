import os
import pdb
import umap
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import cfg
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D


def compute_entropy(similarity_matrix):
    probabilities = F.softmax(similarity_matrix, dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    return entropy


def visualisation_umap(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    center_embeddings,
    low_entropy_nodes,
    predicted_labels,
    hop,
    num_shot,
    dataset_name,
    acc,
):

    save_path = f"visualisation/{dataset_name}/umap_{num_shot}_shot_{hop}_label.pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    umap_model = umap.UMAP(n_components=2, random_state=42)
    all_data = torch.cat(
        [
            train_embeddings.detach().to("cpu"),
            test_embeddings.detach().to("cpu"),
            center_embeddings.detach().to("cpu"),
        ],
        dim=0,
    ).numpy()
    all_data = (all_data - all_data.mean(axis=0)) / (all_data.std(axis=0) + 1e-6)
    embeddings_2d = umap_model.fit_transform(all_data)

    train_size = train_embeddings.shape[0]
    test_size = test_embeddings.shape[0]

    train_2d = embeddings_2d[:train_size]
    test_2d = embeddings_2d[train_size : train_size + test_size]
    center_2d = embeddings_2d[train_size + test_size :]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        test_2d[:, 0],
        test_2d[:, 1],
        c=test_labels.numpy(),
        cmap="tab10",
        alpha=0.2,
        s=10,
        label="Test Nodes",
    )
    plt.scatter(
        train_2d[:, 0],
        train_2d[:, 1],
        c=train_labels.numpy(),
        cmap="tab10",
        marker="o",
        alpha=1,
        edgecolors="black",
        s=50,
        label="Train Nodes",
    )

    # if low_entropy_nodes is not None:
    #     plt.scatter(
    #         test_2d[low_entropy_nodes, 0],
    #         test_2d[low_entropy_nodes, 1],
    #         c=predicted_labels.numpy(),
    #         cmap="tab10",
    #         facecolors="none",
    #         edgecolors="black",
    #         s=20,
    #         marker="s",
    #         label="Lowest Entropy",
    #     )
    plt.scatter(
        center_2d[:, 0],
        center_2d[:, 1],
        c=np.arange(len(center_2d)),
        cmap="tab10",
        edgecolors="black",
        s=100,
        marker="*",
        label="Class Centers",
    )

    # plt.colorbar(label="Class")
    plt.title(f"UMAP {dataset_name} {num_shot} shot {hop} layer acc: {acc:.2f}")
    plt.legend()
    plt.savefig(save_path, format="pdf")
    plt.show()


def visualisation_tsne_train(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    center_embeddings,
    low_entropy_nodes,
    predicted_labels,
    hop,
    num_shot,
    dataset_name,
    acc,
):
    save_path = (
        f"visualisation/{dataset_name}/tsne_{num_shot}_shot_{hop}_center_wo_theta.pdf"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # pdb.set_trace()

    tsne_model = TSNE(n_components=2, perplexity=10, random_state=42)
    all_data = torch.cat(
        [train_embeddings.detach().to("cpu"), center_embeddings.detach().to("cpu")],
        dim=0,
    ).numpy()
    embeddings_2d = tsne_model.fit_transform(all_data)

    train_size = train_embeddings.shape[0]
    train_2d = embeddings_2d[:train_size]
    center_2d = embeddings_2d[train_size:]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        train_2d[:, 0],
        train_2d[:, 1],
        c=train_labels.numpy(),
        cmap="tab10",
        marker="o",
        alpha=0.7,
        edgecolors="black",
        s=50,
        label="Train Nodes",
    )
    plt.scatter(
        center_2d[:, 0],
        center_2d[:, 1],
        c=np.arange(len(center_2d)),
        cmap="tab10",
        edgecolors="black",
        s=100,
        marker="*",
        label="Class Centers",
    )

    # plt.colorbar(label="Class")
    plt.title(f"t-SNE {dataset_name} {num_shot} shot {hop} layer acc: {acc:.2f}")
    plt.legend()
    plt.savefig(save_path, format="pdf")
    plt.show()


def visualisation_tsne(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    center_embeddings,
    low_entropy_nodes,
    predicted_labels,
    hop,
    num_shot,
    dataset_name,
    acc,
):
    save_path = f"visualisation/{dataset_name}/tsne_{num_shot}_shot_{hop}_test2.pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
    all_data = torch.cat(
        [
            train_embeddings.detach().to("cpu"),
            test_embeddings.detach().to("cpu"),
            center_embeddings.detach().to("cpu"),
        ],
        dim=0,
    ).numpy()
    all_data = (all_data - all_data.mean(axis=0)) / (all_data.std(axis=0) + 1e-6)
    embeddings_2d = tsne_model.fit_transform(all_data)

    train_size = train_embeddings.shape[0]
    test_size = test_embeddings.shape[0]

    train_2d = embeddings_2d[:train_size]
    test_2d = embeddings_2d[train_size : train_size + test_size]
    center_2d = embeddings_2d[train_size + test_size :]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        test_2d[:, 0],
        test_2d[:, 1],
        c=test_labels.numpy(),
        cmap="tab10",
        alpha=0.2,
        s=10,
        label="Test Nodes",
    )
    plt.scatter(
        train_2d[:, 0],
        train_2d[:, 1],
        c=train_labels.numpy(),
        cmap="tab10",
        marker="o",
        alpha=1,
        edgecolors="black",
        s=50,
        label="Train Nodes",
    )

    # if low_entropy_nodes is not None:
    #     plt.scatter(
    #         test_2d[low_entropy_nodes, 0],
    #         test_2d[low_entropy_nodes, 1],
    #         c=predicted_labels.numpy(),
    #         cmap="tab10",
    #         facecolors="none",
    #         edgecolors="black",
    #         s=20,
    #         marker="s",
    #         label="Lowest Entropy",
    #     )
    plt.scatter(
        center_2d[:, 0],
        center_2d[:, 1],
        c=np.arange(len(center_2d)),
        cmap="tab10",
        edgecolors="black",
        s=100,
        marker="*",
        label="Class Centers",
    )

    # plt.colorbar(label="Class")
    plt.title(f"t-SNE {dataset_name} {num_shot} shot {hop} layer acc: {acc:.2f}")
    plt.legend()
    plt.savefig(save_path, format="pdf")
    plt.show()


def visualisation_tsne_center(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    stored_center_embeddings,
    low_entropy_nodes,
    predicted_labels,
    hop,
    num_shot,
    dataset_name,
    acc,
):
    save_path = (
        f"visualisation/{dataset_name}/tsne_{num_shot}_shot_{hop}_center_test.pdf"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    center_embeddings = torch.cat(stored_center_embeddings, dim=0)
    # pdb.set_trace()
    num_versions = len(stored_center_embeddings)
    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
    # tsne_model = umap.UMAP(n_components=2, random_state=42)
    all_data = torch.cat(
        [
            train_embeddings.detach().to("cpu"),
            test_embeddings.detach().to("cpu"),
            center_embeddings.detach().to("cpu"),
        ],
        dim=0,
    ).numpy()
    all_data = (all_data - all_data.mean(axis=0)) / (all_data.std(axis=0) + 1e-6)
    embeddings_2d = tsne_model.fit_transform(all_data)

    train_size = train_embeddings.shape[0]
    test_size = test_embeddings.shape[0]

    train_2d = embeddings_2d[:train_size]
    test_2d = embeddings_2d[train_size : train_size + test_size]
    center_2d = embeddings_2d[train_size + test_size :]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        test_2d[:, 0],
        test_2d[:, 1],
        c=test_labels.numpy(),
        cmap="tab10",
        alpha=0.5,
        s=10,
        label="Test Nodes",
    )
    plt.scatter(
        train_2d[:, 0],
        train_2d[:, 1],
        c=train_labels.numpy(),
        cmap="tab10",
        marker="o",
        alpha=1,
        edgecolors="black",
        s=50,
        label="Train Nodes",
    )

    if low_entropy_nodes is not None:
        plt.scatter(
            test_2d[low_entropy_nodes, 0],
            test_2d[low_entropy_nodes, 1],
            c=predicted_labels.numpy(),
            cmap="tab10",
            facecolors="none",
            edgecolors="black",
            s=20,
            marker="s",
            label="Predicted",
        )

    # plt.scatter(
    #     center_2d[:, 0],
    #     center_2d[:, 1],
    #     c=np.arange(len(center_2d)),
    #     cmap="tab10",
    #     edgecolors="black",
    #     s=100,
    #     marker="*",
    #     label="Class Centers",
    # )

    num_classes = stored_center_embeddings[0].shape[0]
    marker_alpha = np.linspace(
        0.6, 1, num_versions
    )  # transparency decreases for older versions
    sizes = np.linspace(
        100, 50, num_versions
    )  # Sizes decrease progressively from 100 to 50
    for version_idx in range(num_versions):
        label = "Train Center" if version_idx == 0 else "Test Center"
        start_idx = version_idx * num_classes
        end_idx = (version_idx + 1) * num_classes

        # Use progressively smaller sizes for each version
        plt.scatter(
            center_2d[start_idx:end_idx, 0],
            center_2d[start_idx:end_idx, 1],
            c=np.arange(num_classes),
            cmap="tab10",
            edgecolors="black",
            s=sizes[version_idx],  # Assign progressively smaller sizes
            marker="*",
            alpha=marker_alpha[version_idx],  # Adjust transparency
            label=label,
        )

    # plt.colorbar(label="Class")
    plt.title(f"t-SNE {dataset_name} {num_shot} shot {hop} layer acc: {acc:.2f}")
    plt.legend()
    plt.savefig(save_path, format="pdf")
    plt.show()


def visualisation_tsne_kmeans(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    stored_center_embeddings,
    low_entropy_nodes,
    predicted_labels,
    hop,
    num_shot,
    dataset_name,
    acc,
):
    save_path = f"visualisation/{dataset_name}/tsne_{num_shot}_shot_{hop}_kmeans.pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_classes = stored_center_embeddings[0].shape[0]
    num_versions = len(stored_center_embeddings)
    center_embeddings = torch.cat(stored_center_embeddings, dim=0)

    all_data = torch.cat(
        [
            train_embeddings.detach().to("cpu"),
            test_embeddings.detach().to("cpu"),
            center_embeddings.detach().to("cpu"),
        ],
        dim=0,
    ).numpy()
    all_data = (all_data - all_data.mean(axis=0)) / (all_data.std(axis=0) + 1e-6)

    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_data)

    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne_model.fit_transform(all_data)

    train_size = train_embeddings.shape[0]
    test_size = test_embeddings.shape[0]

    train_2d = embeddings_2d[:train_size]
    test_2d = embeddings_2d[train_size : train_size + test_size]
    center_2d = embeddings_2d[train_size + test_size :]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        test_2d[:, 0],
        test_2d[:, 1],
        c=cluster_labels[train_size : train_size + test_size],
        cmap="tab10",
        alpha=0.2,
        s=10,
        label="Test Nodes",
    )
    plt.scatter(
        train_2d[:, 0],
        train_2d[:, 1],
        c=cluster_labels[:train_size],
        cmap="tab10",
        marker="o",
        alpha=1,
        edgecolors="black",
        s=50,
        label="Train Nodes",
    )

    if low_entropy_nodes is not None:
        plt.scatter(
            test_2d[low_entropy_nodes, 0],
            test_2d[low_entropy_nodes, 1],
            c=predicted_labels.numpy(),
            cmap="tab10",
            facecolors="none",
            edgecolors="black",
            s=20,
            marker="s",
            label="Predicted",
        )

    marker_alpha = np.linspace(0.6, 1, num_versions)
    sizes = np.linspace(100, 50, num_versions)
    for version_idx in range(num_versions):
        label = "Train Center" if version_idx == 0 else "Test Center"
        start_idx = version_idx * num_classes
        end_idx = (version_idx + 1) * num_classes
        plt.scatter(
            center_2d[start_idx:end_idx, 0],
            center_2d[start_idx:end_idx, 1],
            c=np.arange(num_classes),
            cmap="tab10",
            edgecolors="black",
            s=sizes[version_idx],
            marker="*",
            alpha=marker_alpha[version_idx],
            label=label,
        )

    plt.title(f"t-SNE {dataset_name} {num_shot} shot {hop} layer acc: {acc:.2f}")
    plt.legend()
    plt.savefig(save_path, format="pdf")
    plt.show()


def MultiHopPromptEvaluator(
    loader,
    gnn,
    prompt,
    center_embeddings,
    device,
    num_shot,
    train_loader=None,
    dataset_name=None,
    stored_center_embeddings=None,
):
    correct = 0
    all_train_embeddings = []
    all_train_labels = []
    all_test_embeddings = []
    all_test_labels = []
    all_entropy = []
    all_predicted_labels = []
    layer_accuracies = []

    if train_loader:
        for batch in train_loader:
            batch = batch.to(device)
            out = gnn(batch.x, batch.edge_index, batch.batch)
            all_train_embeddings.append(out.detach().cpu())
            all_train_labels.append(batch.y.detach().cpu())

    correct_consistency = []
    incorrect_consistency = []
    correct_entropy = []
    incorrect_entropy = []
    layer_entropies = []

    use_binary = getattr(cfg, "way", None) == 2
    def _group_centers_to_two(layer_centers):
        # layer_centers: [num_classes, dim]
        num_classes = layer_centers.size(0)
        mid = num_classes // 2
        g0 = layer_centers[:mid].mean(dim=0, keepdim=True)
        g1 = layer_centers[mid:].mean(dim=0, keepdim=True)
        return torch.cat([g0, g1], dim=0), mid

    for batch in loader:
        batch = batch.to(device)
        out = gnn(batch.x, batch.edge_index, batch.batch)
        all_test_embeddings.append(out.detach().cpu())
        all_test_labels.append(batch.y.detach().cpu())

        # Shape: [num_nodes, used_classes]
        used_classes = 2 if use_binary else center_embeddings.size(1)
        accumulated_similarity_matrix = torch.zeros(
            out.size(1), used_classes
        ).to(device)

        layer_preds = []
        if cfg.last_layer:
            embedding = out[-1]
            center_embedding = center_embeddings[-1]

            if use_binary:
                centers_used, mid = _group_centers_to_two(center_embedding)
            else:
                centers_used = center_embedding

            similarity_matrix = F.cosine_similarity(
                embedding.unsqueeze(1), centers_used.unsqueeze(0), dim=-1
            )

            accumulated_similarity_matrix += similarity_matrix * F.relu(
                prompt.gamma[-1]
            )

            pred = similarity_matrix.argmax(dim=1)
            layer_preds.append(pred)
            if use_binary:
                y_bin = (batch.y >= mid).long()
                accuracy = (pred == y_bin).sum().item() / batch.y.size(0)
            else:
                accuracy = (pred == batch.y).sum().item() / batch.y.size(0)
            layer_accuracies.append(accuracy)
            print(f"Last layer accuracy: {accuracy:.4f}")

            probs = F.softmax(similarity_matrix, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            layer_entropies.append(entropy.item())

        else:
            for i, (embedding, center_embedding) in enumerate(
                zip(out, center_embeddings)
            ):
                if use_binary:
                    centers_used, mid = _group_centers_to_two(center_embedding)
                else:
                    centers_used = center_embedding

                similarity_matrix = F.cosine_similarity(
                    embedding.unsqueeze(1), centers_used.unsqueeze(0), dim=-1
                )

                accumulated_similarity_matrix += similarity_matrix * F.relu(
                    prompt.gamma[i]
                )

                pred = similarity_matrix.argmax(dim=1)
                layer_preds.append(pred)
                if use_binary:
                    y_bin = (batch.y >= mid).long()
                    accuracy = (pred == y_bin).sum().item() / batch.y.size(0)
                else:
                    accuracy = (pred == batch.y).sum().item() / batch.y.size(0)
                layer_accuracies.append(accuracy)
                print(f"Layer {i + 1} accuracy: {accuracy:.4f}")

                probs = F.softmax(similarity_matrix, dim=1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
                layer_entropies.append(entropy.item())

                # lowest_pred = similarity_matrix.argmin(dim=1)
                # lowest_correct_mask = lowest_pred == batch.y
                # lowest_correct = lowest_correct_mask.sum().item()
                # lowest_acc = lowest_correct / batch.y.size(0)
                # print(f"Accuracy of comp label: {1 - lowest_acc:.4f}")
            # min_layer = torch.tensor(layer_entropies).argmin().item()
            # print(f"Pivot layer: {min_layer} with min entropy: {layer_entropies[min_layer]:.4f}")

        final_probs = F.softmax(accumulated_similarity_matrix, dim=1)
        final_pred = final_probs.argmax(dim=1)
        all_predicted_labels.append(final_pred.detach().cpu())

        if use_binary:
            y_bin = (batch.y >= mid).long()
            correct_mask = final_pred == y_bin
        else:
            correct_mask = final_pred == batch.y
        correct += correct_mask.sum().item()

    acc = correct / len(loader.dataset)
    print(f"Layer averaged accuracy: {acc:.4f}")
    return acc


def SinglePromptEvaluator(
    loader,
    gnn,
    prompt,
    center_embeddings,
    device,
    num_shot,
    train_loader=None,
    dataset_name=None,
):
    prompt.eval()
    correct = 0
    use_binary = getattr(cfg, "way", None) == 2
    def _group_centers_to_two(centers):
        num_classes = centers.size(0)
        mid = num_classes // 2
        g0 = centers[:mid].mean(dim=0, keepdim=True)
        g1 = centers[mid:].mean(dim=0, keepdim=True)
        return torch.cat([g0, g1], dim=0), mid
    for batch in loader:
        batch = batch.to(device)
        out = gnn(batch.x, batch.edge_index, batch.batch)

        # Compute the cosine similarity between all output embeddings and center embeddings
        if use_binary:
            centers_used, mid = _group_centers_to_two(center_embeddings)
        else:
            centers_used = center_embeddings
        accumulated_similarity_matrix = torch.matmul(out, centers_used.T)

        # Make prediction based on the accumulated similarity matrix
        pred = accumulated_similarity_matrix.argmax(dim=1)

        # Binary evaluation via class merge when cfg.way == 2
        if use_binary:
            y_bin = (batch.y >= mid).long()
            correct += int((pred == y_bin).sum())
            accuracy = (pred == y_bin).sum().item() / batch.y.size(0)
        else:
            correct += int((pred == batch.y).sum())
            # Compute per-batch accuracy for logging
            accuracy = (pred == batch.y).sum().item() / batch.y.size(0)
        print(f"Accuracy: {accuracy:.2f}")

    acc = correct / len(loader.dataset)
    return acc


def Evaluator(
    loader, gnn, prompt, device, num_shot, train_loader=None, dataset_name=None
):
    prompt.eval()
    correct = 0

    all_train_embeddings = []
    all_train_labels = []
    all_test_embeddings = []
    all_test_labels = []
    all_entropy = []
    all_predicted_labels = []

    if train_loader:
        class_embeddings = {}
        class_counts = {}

        for batch in train_loader:
            batch = batch.to(device)
            out = gnn(batch.x, batch.edge_index, batch.batch)
            all_train_embeddings.append(out.detach().cpu())
            all_train_labels.append(batch.y.detach().cpu())
            embeddings = out.detach().cpu()
            labels = batch.y.detach().cpu()

            for embedding, label in zip(embeddings, labels):
                if label.item() not in class_embeddings:
                    class_embeddings[label.item()] = torch.zeros_like(embedding)
                    class_counts[label.item()] = 0
                class_embeddings[label.item()] += embedding
                class_counts[label.item()] += 1

        center_embeddings = []
        for label in sorted(class_embeddings.keys()):
            center_embeddings.append(class_embeddings[label] / class_counts[label])
        center_embeddings = torch.stack(center_embeddings).to(device)
    else:
        center_embeddings = None

    for batch in loader:
        batch = batch.to(device)
        out = gnn(batch.x, batch.edge_index, batch.batch)
        all_test_embeddings.append(out.detach().cpu())
        all_test_labels.append(batch.y.detach().cpu())

        if center_embeddings is not None:
            accumulated_similarity_matrix = torch.zeros(
                out.size(1), center_embeddings.size(1)
            ).to(device)
            for i, (embedding, center_embedding) in enumerate(
                zip(out, center_embeddings)
            ):
                similarity_matrix = F.cosine_similarity(
                    embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1
                )
                accumulated_similarity_matrix += similarity_matrix * prompt.gamma[i]
            # i = center_embeddings.size(0) - 1
            # similarity_matrix = F.cosine_similarity(out[i].unsqueeze(1), center_embeddings[i].unsqueeze(0), dim=-1)
            # accumulated_similarity_matrix = similarity_matrix * prompt.gamma[i]

            entropy = compute_entropy(accumulated_similarity_matrix)
            all_entropy.append(entropy.detach().cpu())
            pred = accumulated_similarity_matrix.argmax(dim=1)
            if getattr(cfg, "way", None) == 2:
                num_classes = accumulated_similarity_matrix.size(1)
                mid = num_classes // 2
                pred_bin = (pred >= mid).long()
                y_bin = (batch.y >= mid).long()
                correct += int((pred_bin == y_bin).sum())
            else:
                correct += int((pred == batch.y).sum())
            all_predicted_labels.append(pred.detach().cpu())

    acc = correct / len(loader.dataset)
    print(f"Accuracy for {num_shot} shot: {acc:.2f}")

    if train_loader and center_embeddings is not None:
        all_train_embeddings = (
            torch.cat(all_train_embeddings, dim=0)
            if all_train_embeddings
            else torch.empty(0, device="cpu")
        )
        all_train_labels = (
            torch.cat(all_train_labels, dim=0)
            if all_train_labels
            else torch.empty(0, device="cpu")
        )
        if len(loader) > 1:
            all_test_embeddings = torch.cat(all_test_embeddings, dim=1)
        else:
            all_test_embeddings = torch.cat(all_test_embeddings, dim=0)
        all_test_labels = torch.cat(all_test_labels, dim=0)
        all_predicted_labels = torch.cat(all_predicted_labels, dim=0)
        all_entropy = torch.cat(all_entropy, dim=0)

        k = 3
        low_entropy_nodes = []
        predicted_labels = []
        for cls in range(center_embeddings.size(1)):
            cls_mask = all_predicted_labels == cls
            if cls_mask.any():
                cls_entropy = all_entropy[cls_mask]
                topk_indices = torch.argsort(cls_entropy)[:k]
                selected_nodes = torch.where(cls_mask)[0][topk_indices]
                low_entropy_nodes.extend(selected_nodes.tolist())
                predicted_labels.extend([cls] * len(selected_nodes))

        low_entropy_nodes = torch.tensor(low_entropy_nodes, dtype=torch.long)
        predicted_labels = torch.tensor(predicted_labels, dtype=torch.long)
        hop = center_embeddings.shape[0] - 1

    return acc
