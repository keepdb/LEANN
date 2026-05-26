extern unsigned char __heap_base;

static unsigned char *heap_cursor = &__heap_base;

void *malloc(unsigned long size) {
  unsigned long aligned_size = (size + 7UL) & ~7UL;
  void *ptr = heap_cursor;
  heap_cursor += aligned_size;
  return ptr;
}

void free(void *ptr) {
  (void)ptr;
}

static float sqrt_approx(float value) {
  if (value <= 0.0f) {
    return 0.0f;
  }

  float x = value > 1.0f ? value : 1.0f;
  for (int i = 0; i < 12; i += 1) {
    x = 0.5f * (x + value / x);
  }
  return x;
}

int leann_wasm_version(void) {
  return 1;
}

float leann_wasm_score_dot(const float *a, const float *b, int dimension) {
  if (a == 0 || b == 0 || dimension <= 0) {
    return 0.0f;
  }

  float total = 0.0f;
  for (int i = 0; i < dimension; i += 1) {
    total += a[i] * b[i];
  }
  return total;
}

static float score_l2(const float *a, const float *b, int dimension) {
  float total = 0.0f;
  for (int i = 0; i < dimension; i += 1) {
    const float diff = a[i] - b[i];
    total += diff * diff;
  }
  return -sqrt_approx(total);
}

static float score_cosine(const float *a, const float *b, int dimension) {
  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;

  for (int i = 0; i < dimension; i += 1) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  if (norm_a <= 0.0f || norm_b <= 0.0f) {
    return 0.0f;
  }

  return dot / (sqrt_approx(norm_a) * sqrt_approx(norm_b));
}

static float score_vector(const float *a, const float *b, int dimension, int metric) {
  if (metric == 1) {
    return score_cosine(a, b, dimension);
  }
  if (metric == 2) {
    return score_l2(a, b, dimension);
  }
  return leann_wasm_score_dot(a, b, dimension);
}

static void insert_desc(int label, float score, int *labels, float *scores, int limit) {
  int insert_at = -1;
  for (int rank = 0; rank < limit; rank += 1) {
    if (score > scores[rank]) {
      insert_at = rank;
      break;
    }
  }
  if (insert_at < 0) {
    return;
  }
  for (int rank = limit - 1; rank > insert_at; rank -= 1) {
    scores[rank] = scores[rank - 1];
    labels[rank] = labels[rank - 1];
  }
  scores[insert_at] = score;
  labels[insert_at] = label;
}

static float score_label(
    const float *vectors,
    int label,
    int dimension,
    const float *query,
    int metric) {
  return score_vector(query, vectors + ((long)label * dimension), dimension, metric);
}

static int hnsw_neighbor_begin(
    const unsigned long long *offsets,
    const int *cum_neighbors,
    int label,
    int level) {
  return (int)(offsets[label] + (unsigned long long)cum_neighbors[level]);
}

static int hnsw_neighbor_end(
    const unsigned long long *offsets,
    const int *cum_neighbors,
    int label,
    int level) {
  return (int)(offsets[label] + (unsigned long long)cum_neighbors[level + 1]);
}

int leann_wasm_flat_search(
    const float *vectors,
    int vector_count,
    int dimension,
    const float *query,
    int top_k,
    int metric,
    int *out_labels,
    float *out_scores) {
  if (vectors == 0 || query == 0 || out_labels == 0 || out_scores == 0) {
    return -1;
  }
  if (vector_count <= 0 || dimension <= 0 || top_k <= 0) {
    return -2;
  }

  for (int i = 0; i < top_k; i += 1) {
    out_labels[i] = -1;
    out_scores[i] = -3.4028234663852886e38f;
  }

  for (int label = 0; label < vector_count; label += 1) {
    const float *candidate = vectors + ((long)label * dimension);
    const float candidate_score = score_vector(query, candidate, dimension, metric);

    int insert_at = -1;
    for (int rank = 0; rank < top_k; rank += 1) {
      if (candidate_score > out_scores[rank]) {
        insert_at = rank;
        break;
      }
    }

    if (insert_at < 0) {
      continue;
    }

    for (int rank = top_k - 1; rank > insert_at; rank -= 1) {
      out_scores[rank] = out_scores[rank - 1];
      out_labels[rank] = out_labels[rank - 1];
    }

    out_scores[insert_at] = candidate_score;
    out_labels[insert_at] = label;
  }

  return 0;
}

int leann_wasm_hnsw_search(
    const float *vectors,
    int vector_count,
    int dimension,
    const unsigned long long *offsets,
    const int *neighbors,
    const int *levels,
    const int *cum_neighbors,
    int cum_neighbor_count,
    int entry_point,
    int max_level,
    const float *query,
    int top_k,
    int ef_search,
    int metric,
    int *out_labels,
    float *out_scores) {
  if (vectors == 0 || offsets == 0 || neighbors == 0 || levels == 0 ||
      cum_neighbors == 0 || query == 0 || out_labels == 0 || out_scores == 0) {
    return -1;
  }
  if (vector_count <= 0 || dimension <= 0 || top_k <= 0 || entry_point < 0 ||
      entry_point >= vector_count || cum_neighbor_count < 2) {
    return -2;
  }

  int ef = ef_search > top_k ? ef_search : top_k;
  if (ef > vector_count) {
    ef = vector_count;
  }
  if (ef <= 0) {
    return -2;
  }

  int *visited = (int *)malloc((unsigned long)vector_count * sizeof(int));
  int *candidate_labels = (int *)malloc((unsigned long)ef * sizeof(int));
  int *expanded = (int *)malloc((unsigned long)vector_count * sizeof(int));
  float *candidate_scores = (float *)malloc((unsigned long)ef * sizeof(float));
  if (visited == 0 || candidate_labels == 0 || expanded == 0 || candidate_scores == 0) {
    return -3;
  }

  for (int i = 0; i < vector_count; i += 1) {
    visited[i] = 0;
    expanded[i] = 0;
  }
  for (int i = 0; i < top_k; i += 1) {
    out_labels[i] = -1;
    out_scores[i] = -3.4028234663852886e38f;
  }

  int nearest = entry_point;
  float nearest_score = score_label(vectors, nearest, dimension, query, metric);

  for (int level = max_level; level >= 1; level -= 1) {
    int changed = 1;
    while (changed) {
      changed = 0;
      if (nearest < 0 || nearest >= vector_count || levels[nearest] <= level) {
        break;
      }
      const int begin = hnsw_neighbor_begin(offsets, cum_neighbors, nearest, level);
      const int end = hnsw_neighbor_end(offsets, cum_neighbors, nearest, level);
      for (int pos = begin; pos < end; pos += 1) {
        const int neighbor = neighbors[pos];
        if (neighbor < 0) {
          break;
        }
        if (neighbor >= vector_count) {
          continue;
        }
        const float score = score_label(vectors, neighbor, dimension, query, metric);
        if (score > nearest_score) {
          nearest = neighbor;
          nearest_score = score;
          changed = 1;
        }
      }
    }
  }

  for (int i = 0; i < ef; i += 1) {
    candidate_labels[i] = -1;
    candidate_scores[i] = -3.4028234663852886e38f;
  }

  visited[nearest] = 1;
  insert_desc(nearest, nearest_score, candidate_labels, candidate_scores, ef);

  while (1) {
    int best_rank = -1;
    for (int rank = 0; rank < ef; rank += 1) {
      const int label = candidate_labels[rank];
      if (label >= 0 && label < vector_count && !expanded[label]) {
        best_rank = rank;
        break;
      }
    }
    if (best_rank < 0) {
      break;
    }

    const int current = candidate_labels[best_rank];
    if (current < 0 || current >= vector_count) {
      continue;
    }
    expanded[current] = 1;

    const int begin = hnsw_neighbor_begin(offsets, cum_neighbors, current, 0);
    const int end = hnsw_neighbor_end(offsets, cum_neighbors, current, 0);
    for (int pos = begin; pos < end; pos += 1) {
      const int neighbor = neighbors[pos];
      if (neighbor < 0) {
        break;
      }
      if (neighbor >= vector_count || visited[neighbor]) {
        continue;
      }
      visited[neighbor] = 1;
      const float score = score_label(vectors, neighbor, dimension, query, metric);
      insert_desc(neighbor, score, candidate_labels, candidate_scores, ef);
    }
  }

  for (int rank = 0; rank < top_k; rank += 1) {
    out_labels[rank] = candidate_labels[rank];
    out_scores[rank] = candidate_scores[rank];
  }

  free(visited);
  free(candidate_labels);
  free(expanded);
  free(candidate_scores);
  return 0;
}
