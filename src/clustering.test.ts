import { describe, expect, it } from 'vitest';
import { agglomerative, cosineDistance, maxPairwiseDistance } from './clustering.js';

describe('cosineDistance', () => {
  it('is 0 for identical direction, 1 for orthogonal, 2 for opposite', () => {
    expect(cosineDistance([1, 0], [2, 0])).toBeCloseTo(0, 6);
    expect(cosineDistance([1, 0], [0, 1])).toBeCloseTo(1, 6);
    expect(cosineDistance([1, 0], [-1, 0])).toBeCloseTo(2, 6);
  });

  it('treats a zero vector as maximally dissimilar (no NaN)', () => {
    expect(cosineDistance([0, 0], [1, 1])).toBe(1);
  });
});

describe('maxPairwiseDistance', () => {
  it('is 0 for fewer than two vectors', () => {
    expect(maxPairwiseDistance([])).toBe(0);
    expect(maxPairwiseDistance([[1, 0]])).toBe(0);
  });

  it('returns the largest pair distance', () => {
    expect(
      maxPairwiseDistance([
        [1, 0],
        [1, 0],
        [0, 1],
      ]),
    ).toBeCloseTo(1, 6);
  });
});

describe('agglomerative', () => {
  it('handles empty and singleton inputs', () => {
    expect(agglomerative([], 0.5)).toEqual([]);
    expect(agglomerative([[1, 0]], 0.5)).toEqual([0]);
  });

  it('merges near-identical vectors into one cluster', () => {
    const labels = agglomerative(
      [
        [1, 0],
        [0.99, 0.01],
        [0.98, 0.02],
      ],
      0.5,
    );
    expect(new Set(labels).size).toBe(1);
  });

  it('separates two well-separated groups', () => {
    const labels = agglomerative(
      [
        [1, 0],
        [0.98, 0.02],
        [0, 1],
        [0.02, 0.98],
      ],
      0.5,
    );
    expect(new Set(labels).size).toBe(2);
    expect(labels[0]).toBe(labels[1]);
    expect(labels[2]).toBe(labels[3]);
    expect(labels[0]).not.toBe(labels[2]);
  });

  it('keeps everything separate below the merge threshold', () => {
    const labels = agglomerative(
      [
        [1, 0],
        [0, 1],
        [-1, 0],
      ],
      0.5,
    );
    expect(new Set(labels).size).toBe(3);
  });

  it('labels clusters by first-appearance order (deterministic)', () => {
    const labels = agglomerative(
      [
        [0, 1], // group X
        [1, 0], // group Y
        [0.02, 0.98], // group X
      ],
      0.5,
    );
    // The cluster containing index 0 must be label 0.
    expect(labels[0]).toBe(0);
    expect(labels[2]).toBe(0);
    expect(labels[1]).toBe(1);
  });
});
