# Dataset curation

Tools used to build the released collections (described in the supplementary
methods of the manuscript), to be added to this folder:

1. **Duplicate screening.** Exact duplicates are consolidated. Near-duplicate
   images are detected by matching ORB keypoints with RANSAC geometric
   verification and grouped with union-find; perceptual hashing provides a
   fast pre-filter.
2. **Visual-pattern clusters.** EfficientNet-B0 embeddings are clustered at
   the 99th percentile of the pairwise cosine similarity; small clusters are
   reduced to one representative and larger retained clusters are assigned to
   a single partition, so that shared content never crosses partitions.
3. **Annotation.** Instances are annotated with SAM (ViT-B) on 2 x 2 patches
   and reviewed manually on the full image.
4. **Partitioning and statistics.** Stratified partitions (instance density and
   resolution group) and descriptive statistics of the collections.
