import os
import torch
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

class DualFeatureExtractor:
    """Extracts both global (CLS token) and local (patch) features using DINOv2"""
    
    def __init__(self, model_name='dinov2_vits14'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim

    def extract(self, image_path: str) -> dict:
        """Extract features from a single image"""
        with Image.open(image_path) as img:
            return self._process_image(img)

    def extract_batch(self, image_dir: str) -> dict:
        """Extract features from all images in a directory"""
        if not os.path.isdir(image_dir):
            raise ValueError(f"Invalid directory: {image_dir}")
            
        features = {}
        for fname in tqdm(os.listdir(image_dir), desc="Processing images"):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    with Image.open(os.path.join(image_dir, fname)) as img:
                        features[fname] = self._process_image(img)
                except Exception as e:
                    print(f"Error processing {fname}: {str(e)}")
        return features

    def _process_image(self, img: Image.Image) -> dict:
        """Enhanced preprocessing with dimension validation"""
        img = img.convert('RGB')
        original_w, original_h = img.size
        
        # Resize to maintain aspect ratio with shortest side 252
        t_h, t_w = (252, int(252*(original_h/original_w))) if original_w > original_h \
                 else (int(252*(original_w/original_h)), 252)
        
        img = transforms.functional.resize(img, (t_h, t_w))
        
        # Calculate padding needs
        def _pad_spec(x): 
            pad = (14 - (x % 14)) % 14
            return (pad//2, pad - pad//2)
        
        w_pad = _pad_spec(img.width)
        h_pad = _pad_spec(img.height)
        
        # Apply symmetric padding
        img = transforms.functional.pad(img, (w_pad[0], h_pad[0], w_pad[1], h_pad[1]), fill=0)
        
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.forward_features(tensor)
        
        return {
            'global': features['x_norm_clstoken'].cpu().numpy().flatten(),
            'local': features['x_norm_patchtokens'].cpu().numpy().squeeze()
        }

class DualVLADBuFF:
    """Handles both global and local feature aggregation with memory efficiency"""
    
    def __init__(self, n_clusters=64, batch_size=1024):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.global_kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                           batch_size=batch_size,
                                           n_init=3)
        self.local_kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                          batch_size=batch_size,
                                          n_init=3)

    def fit(self, features: dict):
        """Train both global and local vocabularies"""
        global_features = np.array([v['global'] for v in features.values()])
        self.global_kmeans.fit(global_features)
        
        local_features = []
        for v in tqdm(features.values(), desc="Processing local features"):
            local_features.append(v['local'])
            if len(local_features) >= self.batch_size:
                self.local_kmeans.partial_fit(np.vstack(local_features))
                local_features = []
        if local_features:
            self.local_kmeans.partial_fit(np.vstack(local_features))
            
        return self

    def transform(self, features: dict) -> dict:
        """Create combined VLAD vectors"""
        vlad_vectors = {}
        for name, vecs in tqdm(features.items(), desc="Creating VLAD vectors"):
            global_vlad = self._compute_vlad(vecs['global'], self.global_kmeans)
            local_vlad = self._compute_vlad(vecs['local'], self.local_kmeans)
            
            combined = np.concatenate([global_vlad, local_vlad])
            combined /= np.linalg.norm(combined) + 1e-12
            vlad_vectors[name] = combined
            
        return vlad_vectors

    def _compute_vlad(self, features: np.ndarray, kmeans) -> np.ndarray:
        """Compute VLAD vector for a single feature set"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        cluster_ids = kmeans.predict(features)
        residuals = features - kmeans.cluster_centers_[cluster_ids]
        
        unique, counts = np.unique(cluster_ids, return_counts=True)
        weights = np.log(1 + counts[np.searchsorted(unique, cluster_ids)])
        
        vlad = np.zeros((self.n_clusters, features.shape[1]))
        np.add.at(vlad, cluster_ids, weights[:, None] * residuals)
        
        vlad = vlad.flatten()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        return vlad / (np.linalg.norm(vlad) + 1e-12)

    def save(self, path: str):
        """Save trained model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'n_clusters': self.n_clusters,
                'global_kmeans': self.global_kmeans,
                'local_kmeans': self.local_kmeans
            }, f)

    @classmethod
    def load(cls, path: str):
        """Load trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vlad = cls(n_clusters=data['n_clusters'])
        vlad.global_kmeans = data['global_kmeans']
        vlad.local_kmeans = data['local_kmeans']
        return vlad

class EnhancedMatchFinder:
    """Efficient similarity search with combined features"""
    
    def __init__(self, vlad_vectors: dict):
        self.names = np.array(list(vlad_vectors.keys()))
        self.vectors = np.array(list(vlad_vectors.values()))

    def query(self, query_vec: np.ndarray, top_k=10) -> list:
        """Find top matches with score thresholding"""
        similarities = cosine_similarity([query_vec], self.vectors)[0]
        sorted_indices = np.argsort(-similarities)
        
        results = []
        for idx in sorted_indices:
            results.append((self.names[idx], float(similarities[idx])))
            if len(results) >= top_k:
                break
                
        return results

def save_features(features: dict, path: str):
    """Save features with compression"""
    with open(path, 'wb') as f:
        pickle.dump(features, f, protocol=4)

def load_features(path: str) -> dict:
    """Load saved features"""
    with open(path, 'rb') as f:
        return pickle.load(f)