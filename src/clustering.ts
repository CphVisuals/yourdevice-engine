/**
 * Pure agglomerative clustering of speaker embeddings — no models, no DOM.
 * Average-linkage, cosine distance, stop when the closest cluster pair
 * exceeds a distance threshold (so the number of speakers N is discovered,
 * not given). Framework-free per CLAUDE.md rule 2; unit-tested per rule 3.
 */

export type Vector = ArrayLike<number>;

/** Cosine distance in [0, 2]; 0 = identical direction, 1 = orthogonal. */
export function cosineDistance(a: Vector, b: Vector): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] ?? 0;
    const bi = b[i] ?? 0;
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 1 : 1 - dot / denom;
}

/** Largest pairwise cosine distance in the set (0 for < 2 vectors). */
export function maxPairwiseDistance(embeddings: readonly Vector[]): number {
  let max = 0;
  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      const d = cosineDistance(embeddings[i]!, embeddings[j]!);
      if (d > max) max = d;
    }
  }
  return max;
}

/**
 * Agglomerative clustering with average linkage.
 * @param embeddings one vector per item to cluster
 * @param threshold merge while the closest cluster pair is below this distance
 * @returns cluster label per input in [0, N), labels assigned by the order
 *   clusters' lowest member index appears (stable, input-order deterministic)
 */
export function agglomerative(embeddings: readonly Vector[], threshold: number): number[] {
  const n = embeddings.length;
  if (n === 0) return [];
  if (n === 1) return [0];

  // Pairwise distance matrix over the original points.
  const dist: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    const di = dist[i]!;
    const dj = (j: number) => dist[j]!;
    for (let j = i + 1; j < n; j++) {
      const d = cosineDistance(embeddings[i]!, embeddings[j]!);
      di[j] = d;
      dj(j)[i] = d;
    }
  }

  // Each cluster is a list of member indices; average linkage = mean of all
  // cross-member pair distances.
  const clusters: number[][] = Array.from({ length: n }, (_, i) => [i]);

  const linkage = (a: number[], b: number[]): number => {
    let sum = 0;
    for (const x of a) for (const y of b) sum += dist[x]![y]!;
    return sum / (a.length * b.length);
  };

  while (clusters.length > 1) {
    let best = Infinity;
    let bi = -1;
    let bj = -1;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const d = linkage(clusters[i]!, clusters[j]!);
        if (d < best) {
          best = d;
          bi = i;
          bj = j;
        }
      }
    }
    if (best > threshold) break; // closest pair too far apart → stop merging
    clusters[bi] = clusters[bi]!.concat(clusters[bj]!);
    clusters.splice(bj, 1);
  }

  // Label clusters by the order of their smallest member index, so labels are
  // deterministic and roughly follow first appearance.
  clusters.sort((a, b) => Math.min(...a) - Math.min(...b));
  const labels = new Array<number>(n).fill(0);
  clusters.forEach((members, label) => {
    for (const idx of members) labels[idx] = label;
  });
  return labels;
}
