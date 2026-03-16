"""
ranking/predictor.py
---------------------
Inference wrapper for the trained NeuMF / DeepFM ranking model.

Loads the saved checkpoint at startup, exposes a single method:
  score_candidates(user_idx, candidate_item_idxs) → np.ndarray of scores

Used by api/recommender.py during online inference.
"""

import os
import json
import numpy as np
import torch

SAVED_DIR  = os.path.join("models", "saved")
META_PATH  = os.path.join("data", "processed", "feature_meta.json")


class RankingPredictor:
    """
    Parameters
    ----------
    model_path : str    Path to saved state dict (.pt file)
    meta_path  : str    Path to feature_meta.json
    device     : str    'cpu' or 'cuda'
    """

    def __init__(
        self,
        model_path: str = os.path.join(SAVED_DIR, "ranking_model.pt"),
        meta_path: str = META_PATH,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # Load feature metadata
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"feature_meta.json not found at {meta_path}. "
                "Run preprocessing/feature_engineering.py first."
            )
        with open(meta_path) as f:
            meta = json.load(f)

        self.n_users      = meta["n_users"]
        self.n_items      = meta["n_items"]
        self.user_feat_dim = meta.get("user_feat_dim", 0)
        self.item_feat_dim = meta.get("item_feat_dim", 0)
        self.model_type    = meta.get("model_type", "neumf")

        # Build model skeleton
        if self.model_type == "neumf":
            from models.neumf_model import build_neumf
            self.model = build_neumf(
                self.n_users, self.n_items,
                self.user_feat_dim, self.item_feat_dim,
            )
        else:
            from models.deepfm_model import build_deepfm
            self.model = build_deepfm(
                self.n_users, self.n_items,
                dense_dim=self.user_feat_dim + self.item_feat_dim,
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Ranking model checkpoint not found at {model_path}. "
                "Run ranking/train_ranking.py first."
            )

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[RankingPredictor] Loaded {self.model_type} from {model_path}")

        # Pre-load feature arrays
        data_dir = os.path.dirname(meta_path)
        uf_path = os.path.join(data_dir, "user_features.npy")
        if_path = os.path.join(data_dir, "item_features.npy")
        self.user_features = (
            np.load(uf_path) if os.path.exists(uf_path) and self.user_feat_dim > 0
            else None
        )
        self.item_features = (
            np.load(if_path) if os.path.exists(if_path) and self.item_feat_dim > 0
            else None
        )
        if self.user_features is not None:
            print(f"[RankingPredictor] Successfully loaded features from {data_dir}")
        else:
            print(f"[RankingPredictor] WARNING: Feature arrays not found in {data_dir}!")

    @torch.no_grad()
    def score_candidates(
        self,
        user_idx: int,
        candidate_item_idxs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute relevance scores for a set of candidate items for one user.

        Parameters
        ----------
        user_idx             : int scalar
        candidate_item_idxs  : 1-D int array of item indices

        Returns
        -------
        scores : 1-D float32 array, same length as candidate_item_idxs
        """
        n = len(candidate_item_idxs)
        u_tensor = torch.tensor([user_idx] * n, dtype=torch.long, device=self.device)
        i_tensor = torch.tensor(candidate_item_idxs, dtype=torch.long, device=self.device)

        uf = itf = None
        if self.user_features is not None:
            uf = torch.tensor(
                self.user_features[[user_idx] * n], dtype=torch.float32,
                device=self.device,
            )
        if self.item_features is not None:
            itf = torch.tensor(
                self.item_features[candidate_item_idxs], dtype=torch.float32,
                device=self.device,
            )

        if self.model_type == "neumf":
            scores = self.model(u_tensor, i_tensor, uf, itf)
        else:
            sparse_in = torch.stack([u_tensor, i_tensor], dim=1)
            dense_in  = None
            if uf is not None and itf is not None:
                dense_in = torch.cat([uf, itf], dim=-1)
            elif uf is not None:
                dense_in = uf
            elif itf is not None:
                dense_in = itf
            scores = self.model(sparse_in, dense_in)

        return scores.cpu().numpy()
