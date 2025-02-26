import utils
import numpy as np

extractor = utils.DualFeatureExtractor()
features = extractor.extract_batch("./image_data/")

vlad = utils.DualVLADBuFF(n_clusters=128)
vlad.fit(features)
vlad_vectors = vlad.transform(features)

vlad.save("dual_vlad.pkl")
utils.save_features(vlad_vectors, "dual_features.pkl")


query_features = extractor.extract("./target_image/team3.jpg")
query_global = vlad._compute_vlad(query_features['global'], vlad.global_kmeans)
query_local = vlad._compute_vlad(query_features['local'], vlad.local_kmeans)
query_vec = np.concatenate([query_global, query_local])

finder = utils.EnhancedMatchFinder(vlad_vectors)
matches = finder.query(query_vec, top_k=5)

print("Top matches:")
for i, (name, score) in enumerate(matches, 1):
    print(f"{i}. {name} ({score:.3f})")
