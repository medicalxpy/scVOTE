import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


from . import _plot
from ._fastopic import fastopic
from ._utils import Logger, assert_fitted, check_fitted, ScDataset, get_top_words

from typing import Dict, List, Optional


logger = Logger("WARNING")


class FASTopic:
    def __init__(
        self,
        num_topics: int,
        num_top_words: int = 15,
        device: str = None,
        DT_alpha: float = 3.0,
        TW_alpha: float = 2.0,
        theta_temp: float = 1.0,
        # Structural alignment (Laplacian + CKA)
        align_enable: bool = True,
        align_alpha: float = 1e-3,
        align_beta: float = 1e-3,
        align_knn_k: int = 48,
        align_cka_sample_n: int = 2048,
        align_max_kernel_genes: int = 4096,
        # Legacy GenePT contrastive alignment
        genept_loss_weight: float = 0.0,
        topic_diversity_weight: float = 0.0,
        low_memory: bool = False,
        low_memory_batch_size: int = None,
        verbose: bool = False,
        log_interval: int = 10,
    ):
        """FASTopic initialization (single-cell focused).

        Args:
            num_topics: The number of topics.
            num_top_words: Number of top words to be returned in topics.
            DT_alpha: Sinkhorn alpha between document (cell) embeddings and topic embeddings.
            TW_alpha: Sinkhorn alpha between topic embeddings and word (gene) embeddings.
            theta_temp: Temperature parameter when computing topic distributions.
            device: The device (cuda/cpu).
            log_interval: Interval to print logs during training.
            verbose: Verbosity.
        """

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.num_top_words = num_top_words
        self.low_memory = low_memory
        self.low_memory_batch_size = low_memory_batch_size

        self.beta = None
        self.train_theta = None
        self.model = fastopic(
            num_topics,
            theta_temp,
            DT_alpha,
            TW_alpha,
            genept_loss_weight=genept_loss_weight,
            topic_diversity_weight=topic_diversity_weight,
            align_enable=align_enable,
            align_alpha=align_alpha,
            align_beta=align_beta,
            align_knn_k=align_knn_k,
            align_cka_sample_n=align_cka_sample_n,
            align_max_kernel_genes=align_max_kernel_genes,
        )

        self.log_interval = log_interval
        self.verbose = verbose
        if verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

        logger.info(f'use device: {device}')

    def make_optimizer(self, learning_rate: float):
        args_dict = {
            'params': self.model.parameters(),
            'lr': learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def fit_transform_sc(
        self,
        cell_embeddings: np.ndarray,
        gene_names: List[str],
        expression_bow: sp.csr_matrix = None,
        epochs: int = 200,
        learning_rate: float = 0.002,
        patience: int = 10,
        min_delta: float = 1e-4,
        init_word_embeddings: Optional[Dict[str, np.ndarray]] = None,
        reinit_topics: bool = False,
    ):
        """
        Single-cell version of fit_transform that works directly with cell embeddings.
        
        Args:
            cell_embeddings: Precomputed cell embeddings of shape (N, D)
            gene_names: List of gene names for vocabulary
            epochs: The number of epochs
            learning_rate: The learning rate
        """
        data_size = cell_embeddings.shape[0]
        if self.low_memory:
            logger.info("Using low memory mode.")
            assert self.low_memory_batch_size is not None
            self.batch_size = self.low_memory_batch_size
            dataset_device = 'cpu'
        else:
            self.batch_size = data_size
            dataset_device = self.device

        # Determine (re)initialization strategy.
        # If reinit_topics is True, we will reinitialize topics even if the model was fitted.
        previously_fitted = check_fitted(self)
        if previously_fitted and not reinit_topics:
            logger.info("Fine-tuning the model (reuse topics).")
            _fitted = True
        elif previously_fitted and reinit_topics:
            logger.info("Fine-tuning with topic reinitialization (reuse words via init_word_embeddings if provided).")
            _fitted = False
        else:
            logger.info("First fit the model.")
            _fitted = False

        # Create the dataset for single-cell data
        dataset = ScDataset(
            cell_embeddings=cell_embeddings,
            gene_names=gene_names,
            expression_bow=expression_bow,
            batch_size=self.batch_size,
            device=dataset_device,
            low_memory=self.low_memory,
        )

        self.train_doc_embeddings = torch.as_tensor(dataset.cell_embeddings)
        if not self.low_memory:
            self.train_doc_embeddings = self.train_doc_embeddings.to(self.device)

        vocab_size = dataset.vocab_size
        doc_embed_size = dataset.cell_embed_size

        if not _fitted:
            self.model.init(
                vocab_size,
                doc_embed_size,
                cell_embeddings=cell_embeddings,
                vocab=dataset.vocab
            )
            if init_word_embeddings is not None:
                self._apply_initial_word_embeddings(
                    dataset.vocab,
                    init_word_embeddings
                )
        else:
            pre_vocab = self.vocab
            self.model.init(
                vocab_size,
                doc_embed_size,
                _fitted,
                pre_vocab,
                dataset.vocab,
                cell_embeddings
            )
            if init_word_embeddings is not None:
                logger.warning("Initial word embeddings provided but model is already fitted; ignoring override.")

        self.vocab = dataset.vocab
        self.model = self.model.to(self.device)

        optimizer = self.make_optimizer(learning_rate)

        # Start training.
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(1, epochs + 1), desc="Training scFASTopic"):
            # Accumulate scalar losses on CPU to avoid keeping the whole
            # computation graph in memory across the epoch.
            loss_rst_dict = defaultdict(float)

            for batch_bow, batch_cell_embed in dataset.dataloader:
                if self.low_memory:
                    batch_cell_embed = batch_cell_embed.to(self.device)
                    batch_bow = batch_bow.to(self.device)

                rst_dict = self.model(batch_bow, batch_cell_embed)
                batch_loss = rst_dict["loss"]

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                batch_size = batch_bow.shape[0]
                # Detach per-batch losses and move to CPU so we do not
                # retain computational graphs and GPU tensors.
                for key, value in rst_dict.items():
                    with torch.no_grad():
                        val = value.detach()
                        # Ensure scalar; if a tensor has shape, take mean.
                        if hasattr(val, "dim") and val.dim() > 0:
                            val = val.mean()
                        loss_rst_dict[key] += float(val) * batch_size

                # Explicitly release batch tensors and clear CUDA cache to
                # mitigate OOM on large datasets (see upstream FASTopic
                # discussions / issue about GPU memory).
                del batch_bow, batch_cell_embed, batch_loss, rst_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_losses = {key: loss_rst_dict[key] / data_size for key in loss_rst_dict}
            current_loss = avg_losses['loss']
            
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % self.log_interval == 0:
                if 'loss_ETP' in avg_losses and 'loss_DSR' in avg_losses:
                    output_log = f"Epoch: {epoch:03d} | Total: {avg_losses['loss']:.3f} = ETP: {avg_losses['loss_ETP']:.3f} + DSR: {avg_losses['loss_DSR']:.3f}"
                    if 'loss_DT' in avg_losses and 'loss_TW' in avg_losses:
                        output_log += f" | ETP = DT: {avg_losses['loss_DT']:.3f} + TW: {avg_losses['loss_TW']:.3f}"
                else:
                    output_log = f"Epoch: {epoch:03d}"
                    for key in avg_losses:
                        output_log += f" {key}: {avg_losses[key]:.3f}"
                
                logger.info(output_log)
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.beta = self.get_beta()

        self.top_words = self.get_top_words(self.num_top_words)
        self.train_theta = self.transform_sc(self.train_doc_embeddings)

        return self.top_words, self.train_theta
    
    def transform_sc(self, cell_embeddings: np.ndarray):
        """
        Single-cell version of transform that works directly with cell embeddings.
        
        Args:
            cell_embeddings: Cell embeddings array of shape (N, D)
        
        Returns:
            theta: Cell-topic distribution matrix
        """
        if isinstance(cell_embeddings, np.ndarray):
            cell_embeddings = torch.as_tensor(cell_embeddings)
        
        if not self.low_memory:
            cell_embeddings = cell_embeddings.to(self.device)

        with torch.no_grad():
            self.model.eval()
            theta = self.model.get_theta(cell_embeddings, self.train_doc_embeddings)
            theta = theta.detach().cpu().numpy()

        return theta

    def get_beta(self):
        """
            return beta: topic-word distributions matrix, $K \times V$
        """
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def get_top_words(self, num_top_words=15, verbose=None):
        if verbose is None:
            verbose = self.verbose
        beta = self.get_beta()
        top_words = get_top_words(beta, self.vocab, num_top_words, verbose)
        return top_words

    @property
    def topic_embeddings(self):
        """
            return topic embeddings $K \times L$
        """
        return self.model.topic_embeddings.detach().cpu().numpy()

    @property
    def word_embeddings(self):
        """
            return word embeddings $V \times L$
        """
        return self.model.word_embeddings.detach().cpu().numpy()

    def _apply_initial_word_embeddings(
        self,
        vocab: List[str],
        init_data: Dict[str, np.ndarray],
    ) -> None:
        """Inject pretrained gene embeddings prior to training.

        The ``init_data`` dictionary must contain an ``embeddings`` array of
        shape ``(V_prev, D)`` and the corresponding ``vocab`` list.  An optional
        ``weights`` array with shape ``(V_prev, 1)`` can also be provided.  The
        embeddings are matched by gene name and copied into the model tensor.
        Genes that do not appear in ``init_data['vocab']`` keep their initial
        random vectors.
        """

        required_keys = {"embeddings", "vocab"}
        missing = required_keys - set(init_data)
        if missing:
            raise ValueError(f"init_word_embeddings is missing keys: {missing}")

        prev_vocab = init_data["vocab"]
        prev_embeddings = init_data["embeddings"]
        if len(prev_vocab) != len(prev_embeddings):
            raise ValueError("Length of vocab and embeddings mismatch in initial word embeddings")

        vocab_to_idx = {gene: idx for idx, gene in enumerate(prev_vocab)}

        with torch.no_grad():
            word_tensor = self.model.word_embeddings.data
            device = word_tensor.device
            dtype = word_tensor.dtype

            matched = 0
            for i, gene in enumerate(vocab):
                prev_idx = vocab_to_idx.get(gene)
                if prev_idx is None:
                    continue
                vec = torch.as_tensor(prev_embeddings[prev_idx], device=device, dtype=dtype)
                if vec.shape != word_tensor[i].shape:
                    raise ValueError("Embedding dimensionality mismatch for gene '%s'" % gene)
                word_tensor[i] = vec
                matched += 1

            if matched:
                word_tensor[:] = F.normalize(word_tensor, dim=1)

            if "weights" in init_data:
                weights = init_data["weights"]
                if len(weights) != len(prev_vocab):
                    raise ValueError("Length of weights must match vocab in initial word embeddings")
                weight_tensor = self.model.word_weights.data
                weight_dtype = weight_tensor.dtype
                for i, gene in enumerate(vocab):
                    prev_idx = vocab_to_idx.get(gene)
                    if prev_idx is None:
                        continue
                    weight_tensor[i] = torch.as_tensor(weights[prev_idx], device=device, dtype=weight_dtype)

        if matched == 0:
            logger.warning("No genes matched between provided initial embeddings and current vocabulary.")

    @property
    def transp_DT(self):
        """
            return transp_DT $N \times K$
        """
        return self.model.get_transp_DT(self.train_doc_embeddings)

    def save(
        self,
        path: str
    ):
        """Saves the FASTopic model and its PyTorch model weights to the specified path, like `./fastopic.zip`.

        This method saves the dict attributes of the FASTopic object (`self`).

        Args:
            path (str): The path to save the model files. If the directory doesn't exist, it will be created.

        Returns:
            None
        """
        assert_fitted(self)

        path = Path(path)
        parent_dir = path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

        instance_dict = {k: v for k, v in self.__dict__.items()}

        state = {
            "instance_dict": instance_dict
        }
        torch.save(state, path)

    @classmethod
    def from_pretrained(
            cls,
            path: str,
            low_memory: bool = None,
            low_memory_batch_size: int = None,
            device: str=None
        ):
        """Loads a pre-trained FASTopic model from a saved file (single-cell variant)."""

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        state = torch.load(path, weights_only=False)

        instance_dict = state["instance_dict"]
        instance_dict["device"] = device
        if low_memory:
            instance_dict["low_memory"] = low_memory
            instance_dict["low_memory_batch_size"] = low_memory_batch_size

        for key, val in instance_dict.items():
            if key != "train_doc_embeddings" and isinstance(val, torch.Tensor):
                instance_dict[key] = val.to(device)

        if not instance_dict["low_memory"]:
            # Move train_doc_embeddings to the device.
            instance_dict["train_doc_embeddings"] = instance_dict["train_doc_embeddings"].to(device)

        instance = cls.__new__(cls)
        instance.__dict__.update(instance_dict)

        if instance.verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

        return instance

    def get_topic(
            self,
            topic_idx: int,
            num_top_words: int=5
        ):

        assert_fitted(self)
        words = self.top_words[topic_idx].split()[:num_top_words]
        scores = np.sort(self.beta[topic_idx])[:-(num_top_words + 1):-1]

        return tuple(zip(words, scores))

    def get_topic_weights(self):
        assert_fitted(self)
        topic_weights = self.transp_DT.sum(0)
        return topic_weights

    def visualize_topic(self, **args):
        assert_fitted(self)
        return _plot.visualize_topic(self, **args)

    def visualize_topic_hierarchy(self, **args):
        assert_fitted(self)
        return _plot.visualize_hierarchy(self, **args)

    def topic_activity_over_time(self,
                                 time_slices: List[int],
                                ):
        assert_fitted(self)
        topic_activity = self.transp_DT
        topic_activity *= self.transp_DT.shape[0]

        assert len(time_slices) == topic_activity.shape[0]

        df = pd.DataFrame(topic_activity)
        df['time_slices'] = time_slices
        topic_activity = df.groupby('time_slices').mean().to_numpy().transpose()

        return topic_activity

    def visualize_topic_activity(self, **args):
        assert_fitted(self)
        return _plot.visualize_activity(self, **args)

    def visualize_topic_weights(self, **args):
        assert_fitted(self)
        return _plot.visualize_topic_weights(self, **args)
