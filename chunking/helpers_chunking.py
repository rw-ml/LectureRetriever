import numpy as np

class RunningCentroid:
    """Simple running centroid for one cluster."""
    def __init__(self):
        self.centroid = None
        self.count = 0

    def similarity(self, embedding):
        if self.centroid is None:
            return 1.0
        return float(self.centroid @ embedding)

    def add(self, embedding):
        if self.centroid is None:
            self.centroid = embedding
            self.count = 1
        else:
            new_centroid = (self.centroid * self.count + embedding) / (self.count + 1)
            self.centroid = new_centroid / np.linalg.norm(new_centroid)
            self.count += 1




class MultiCentroidManager:
    """
    Manage multiple running centroids for dynamic topic assignment.
    """
    def __init__(self, similarity_threshold=0.65):
        self.similarity_threshold = similarity_threshold
        self.centroids = []        # list of RunningCentroid
        self.chunk_ids = []        # chunk_id per centroid
        self.chunks = {}           # chunk_id -> list of page dicts
        self.next_chunk_id = 0

    def add_page(self, embedding, page_info):
        best_sim = -1
        best_idx = None

        for idx, c in enumerate(self.centroids):
            sim = c.similarity(embedding)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_sim >= self.similarity_threshold:
            centroid = self.centroids[best_idx]
            chunk_id = self.chunk_ids[best_idx]
            centroid.add(embedding)
        else:
            centroid = RunningCentroid()
            centroid.add(embedding)
            self.centroids.append(centroid)
            chunk_id = self.next_chunk_id
            self.chunk_ids.append(chunk_id)
            self.next_chunk_id += 1

        if chunk_id not in self.chunks:
            self.chunks[chunk_id] = []

        self.chunks[chunk_id].append(page_info)

    def get_chunks(self):
        results = []
        for chunk_id, pages in self.chunks.items():
            results.append({
                'chunk_id': chunk_id,
                'source': pages[0]['source'],
                'pages': [p['page'] for p in pages],
                'text': " ".join([p['text'] for p in pages])
            })
        return results