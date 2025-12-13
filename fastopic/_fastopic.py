import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import pickle
import warnings

from ._ETP import ETP
from ._model_utils import pairwise_euclidean_distance
from ._alignment import GeneAlignmentRef, AlignmentConfig



class fastopic(nn.Module):
    def __init__(self,
                 num_topics: int,
                 theta_temp: float = 1.0,
                 DT_alpha: float = 3.0,
                 TW_alpha: float = 2.0,
                 genept_proj_dim: int = 512,
                 genept_proj_hidden: int = 1024,
                 genept_temperature: float = 0.1,
                 genept_loss_weight: float = 0.0,
                 # Structural alignment (Laplacian + CKA)
                 align_enable: bool = True,
                 align_alpha: float = 1e-3,
                 align_beta: float = 1e-3,
                 align_knn_k: int = 48,
                 align_rbf_quantiles: tuple = (0.1, 0.3, 0.6, 0.9),
                 align_cka_sample_n: int = 2048,
                 align_max_kernel_genes: int = 4096,
                ):
        super().__init__()

        self.num_topics = num_topics
        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.theta_temp = theta_temp
        self.genept_proj_dim = genept_proj_dim
        self.genept_proj_hidden = genept_proj_hidden
        self.genept_temperature = genept_temperature
        self.genept_loss_weight = genept_loss_weight
        # Structural alignment config
        self.align_enable = align_enable
        self.align_alpha = align_alpha
        self.align_beta = align_beta
        self.align_knn_k = align_knn_k
        self.align_rbf_quantiles = align_rbf_quantiles
        self.align_cka_sample_n = align_cka_sample_n
        self.align_max_kernel_genes = align_max_kernel_genes

        self.epsilon = 1e-12
        self.word_projector = None
        self.genept_projector = None
        self._genept_embeddings = None
        self._gene_alignment_mask = None
        self._align_ref = None

    def init(self,
             vocab_size: int,
             embed_size: int,
             _fitted: bool = False,
             pre_vocab: list=None,
             vocab: list=None,
             cell_embeddings: np.ndarray=None
            ):

        if _fitted:
            topic_embeddings = self.topic_embeddings.data
            assert topic_embeddings.shape == (self.num_topics, embed_size)
            topic_weights = self.topic_weights.data
            del self.topic_weights
        else:
            topic_embeddings = F.normalize(nn.init.trunc_normal_(torch.empty((self.num_topics, embed_size))))
            topic_weights = (torch.ones(self.num_topics) / self.num_topics).unsqueeze(1)

        self.topic_embeddings = nn.Parameter(topic_embeddings)
        self.topic_weights = nn.Parameter(topic_weights)

        # Initialize word embeddings with random values
        word_embeddings = F.normalize(nn.init.trunc_normal_(torch.empty(vocab_size, embed_size)))
        
        if _fitted:
            pre_word_embeddings = self.word_embeddings.data
            word_weights = torch.zeros(vocab_size, 1)
            pre_norm_word_weights = F.softmax(self.word_weights.data, dim=0)
            del self.word_embeddings
            del self.word_weights

            for i, word in enumerate(vocab):
                if word in pre_vocab:
                    pre_word_idx = pre_vocab.index(word)
                    word_embeddings[i] = pre_word_embeddings[pre_word_idx]
                    word_weights[i] = pre_norm_word_weights[pre_word_idx]

            left_avg = (1.0 - word_weights.sum()) / word_weights.nonzero().size(0)
            word_weights[word_weights == 0] = left_avg

            word_weights = torch.log(word_weights)
            word_weights = word_weights - word_weights.mean()

        else:
            word_weights = (torch.ones(vocab_size) / vocab_size).unsqueeze(1)

        self.word_embeddings = nn.Parameter(word_embeddings)
        self.word_weights = nn.Parameter(word_weights)

        # Store vocab_size for weighting calculations
        self.vocab_size = vocab_size

        # Store vocabulary for GenePT/structural alignment
        self._vocab = vocab

        if not _fitted or self.word_projector is None:
            self.word_projector = self._build_projector(embed_size, self.genept_proj_dim)

        self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
        self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)

        # Build reference for structural alignment if enabled and vocab available
        try:
            if self.align_enable and self._vocab is not None:
                cfg = AlignmentConfig(
                    knn_k=int(self.align_knn_k),
                    rbf_quantiles=tuple(self.align_rbf_quantiles),
                    cka_sample_n=int(self.align_cka_sample_n),
                    max_kernel_genes=int(self.align_max_kernel_genes),
                    random_state=42,
                )
                genept_path = 'GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle'
                self._align_ref = GeneAlignmentRef(self._vocab, genept_path, config=cfg)
        except Exception as e:
            print(f"⚠️ Failed to build structural alignment reference: {e}")

    def get_transp_DT(self, doc_embeddings):
        topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
        _, transp = self.DT_ETP(doc_embeddings, topic_embeddings)

        return transp.detach().cpu().numpy()

    # only for testing
    def get_beta(self):
        with torch.no_grad():
            _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
            # use transport plan as beta
            beta = transp_TW * transp_TW.shape[0]

            return beta

    # only for testing
    def get_theta(self,
            doc_embeddings,
            train_doc_embeddings
        ):
        with torch.no_grad():
            topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
            dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)
            train_dist = pairwise_euclidean_distance(train_doc_embeddings, topic_embeddings)

            exp_dist = torch.exp(-dist / self.theta_temp)
            exp_train_dist = torch.exp(-train_dist / self.theta_temp)

            theta = exp_dist / (exp_train_dist.sum(0))
            theta = theta / theta.sum(1, keepdim=True)

            return theta

    def forward(self, train_bow, doc_embeddings):
        loss_DT, transp_DT, cost_DT_mean = self.DT_ETP(
            doc_embeddings, self.topic_embeddings, return_cost_mean=True
        )
        loss_TW, transp_TW, cost_TW_mean = self.TW_ETP(
            self.topic_embeddings, self.word_embeddings, return_cost_mean=True
        )

        # Auto scale balancing: normalize by distance scales so DT/TW losses are comparable.
        # Use stop-gradient on the scales to avoid introducing incentives to inflate distances.
        dt_scale = torch.clamp(cost_DT_mean.detach(), min=self.epsilon)
        tw_scale = torch.clamp(cost_TW_mean.detach(), min=self.epsilon)
        loss_ETP = (loss_DT / dt_scale) + (loss_TW / tw_scale)
        
        # # Auto scale balancing — compute distance matrices to normalize losses
        # from ._model_utils import pairwise_euclidean_distance
        # M_DT = pairwise_euclidean_distance(doc_embeddings, self.topic_embeddings)
        # M_TW = pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        
        # # Normalize by distance scales so DT/TW losses are comparable
        # loss_DT_normalized = loss_DT / M_DT.mean()
        # loss_TW_normalized = loss_TW / M_TW.mean()
        # loss_ETP = loss_DT_normalized + loss_TW_normalized
        
        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]

        
        recon = torch.matmul(theta, beta)
        loss_DSR = -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()
        
        # Add structural alignment losses (Laplacian + CKA)
        loss_lap, loss_cka = self._compute_struct_alignment_losses()

        # Keep legacy GenePT contrastive alignment loss, but only compute it
        # when the weight is positive to avoid unnecessary (and heavy) work.
        if self.genept_loss_weight is not None and self.genept_loss_weight > 0:
            loss_genept_alignment = self._compute_genept_alignment_loss()
        else:
            loss_genept_alignment = torch.tensor(
                0.0, device=self.word_embeddings.device
            )

        loss = (
            loss_DSR
            + 1e-2 * loss_ETP
            + self.align_alpha * loss_lap
            + self.align_beta * loss_cka
            + self.genept_loss_weight * loss_genept_alignment
        )

        rst_dict = {
            'loss': loss,
            'loss_ETP': loss_ETP,
            'loss_DSR': loss_DSR,
            'loss_DT': loss_DT,
            'loss_TW': loss_TW,
            'loss_genept_alignment': loss_genept_alignment,
            'loss_lap': loss_lap,
            'loss_cka': loss_cka,
        }

        return rst_dict
    
    def _compute_genept_alignment_loss(self):
        """
        Compute GenePT alignment loss to align learned gene embeddings
        with pretrained GenePT embeddings.
        """
        # If no vocabulary information, return zero loss
        if not hasattr(self, '_vocab') or self._vocab is None:
            return torch.tensor(0.0, device=self.word_embeddings.device)
        
        # Lazy-load GenePT embeddings
        if not hasattr(self, '_genept_embeddings') or self._genept_embeddings is None:
            self._load_genept_embeddings()
        
        # If loading failed or there are no aligned genes, return zero loss
        if self._genept_embeddings is None or self._gene_alignment_mask is None:
            return torch.tensor(0.0, device=self.word_embeddings.device)
        
        aligned_word_embeddings = self.word_embeddings[self._gene_alignment_mask]
        device = aligned_word_embeddings.device

        if self.word_projector is None:
            self.word_projector = self._build_projector(aligned_word_embeddings.shape[1], self.genept_proj_dim).to(device)
        else:
            self.word_projector = self.word_projector.to(device)

        genept_embeddings = self._genept_embeddings.to(device)
        if self.genept_projector is None:
            self.genept_projector = self._build_projector(genept_embeddings.shape[1], self.genept_proj_dim).to(device)
        else:
            self.genept_projector = self.genept_projector.to(device)

        word_proj = self.word_projector(aligned_word_embeddings)
        gene_proj = self.genept_projector(genept_embeddings)

        word_proj = F.normalize(word_proj, dim=1)
        gene_proj = F.normalize(gene_proj, dim=1)

        logits = torch.matmul(word_proj, gene_proj.t()) / self.genept_temperature
        targets = torch.arange(logits.size(0), device=device)

        loss_i = F.cross_entropy(logits, targets)
        loss_j = F.cross_entropy(logits.t(), targets)
        alignment_loss = 0.5 * (loss_i + loss_j)

        return alignment_loss

    def _compute_struct_alignment_losses(self):
        """
        Compute Laplacian and CKA losses using reference structures from GenePT.
        Returns a tuple (loss_lap, loss_cka).
        """
        if not self.align_enable or self._align_ref is None:
            dev = self.word_embeddings.device
            z = torch.tensor(0.0, device=dev)
            return z, z
        try:
            loss_lap = self._align_ref.laplacian_loss(self.word_embeddings)
        except Exception as e:
            print(f"⚠️ Failed to compute Laplacian alignment loss: {e}")
            loss_lap = torch.tensor(0.0, device=self.word_embeddings.device)
        try:
            loss_cka = self._align_ref.cka_loss(self.word_embeddings)
        except Exception as e:
            print(f"⚠️ Failed to compute CKA alignment loss: {e}")
            loss_cka = torch.tensor(0.0, device=self.word_embeddings.device)
        return loss_lap, loss_cka
    
    def _load_genept_embeddings(self):
        """
        Load GenePT gene embeddings and create an alignment mask.
        """
        try:
            genept_path = 'GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle'
            
            with open(genept_path, 'rb') as f:
                genept_dict = pickle.load(f)
            
            # Build aligned list and mask
            aligned_embeddings = []
            alignment_mask = []
            
            for i, gene_name in enumerate(self._vocab):
                if gene_name in genept_dict:
                    genept_emb = torch.tensor(genept_dict[gene_name], dtype=torch.float32)
                    aligned_embeddings.append(genept_emb)
                    alignment_mask.append(i)
            
            if len(aligned_embeddings) > 0:
                self._genept_embeddings = torch.stack(aligned_embeddings)
                self._gene_alignment_mask = torch.tensor(alignment_mask, dtype=torch.long)
                print(f"✅ GenePT alignment: {len(aligned_embeddings)}/{len(self._vocab)} genes matched")
                if self.genept_projector is None:
                    in_dim = self._genept_embeddings.shape[1]
                    self.genept_projector = self._build_projector(in_dim, self.genept_proj_dim)
            else:
                self._genept_embeddings = None
                self._gene_alignment_mask = None
                print("⚠️ No genes could be aligned with GenePT")
                
        except Exception as e:
            print(f"❌ Failed to load GenePT embeddings: {e}")
            self._genept_embeddings = None
            self._gene_alignment_mask = None

    def _build_projector(self, in_dim: int, out_dim: int) -> nn.Module:
        hidden_dim = self.genept_proj_hidden if self.genept_proj_hidden and self.genept_proj_hidden > 0 else None
        if hidden_dim and hidden_dim != in_dim and hidden_dim != out_dim:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            return nn.Linear(in_dim, out_dim)
