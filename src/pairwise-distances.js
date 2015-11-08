import ndarray from 'ndarray';
import cwise from 'cwise';

// Euclidean distance
let euclidean = cwise({
  args: ['array', 'array'],
  pre: function(a, b) {
    this.sum = 0.0;
  },
  body: function(a, b) {
    var d = a - b;
    this.sum += d * d;
  },
  post: function(a, b) {
    return Math.sqrt(this.sum);
  }
});

// Manhattan distance
let manhattan = cwise({
  args: ['array', 'array'],
  pre: function(a, b) {
    this.sum = 0.0;
  },
  body: function(a, b) {
    this.sum += Math.abs(a - b);
  },
  post: function(a, b) {
    return this.sum;
  }
});

// Jaccard dissimilarity
let jaccard = cwise({
  args: ['array', 'array'],
  pre: function(a, b) {
    this.tf = 0.0;
    this.tt = 0.0;
  },
  body: function(a, b) {
    var a_bool = Math.round(a);
    var b_bool = Math.round(b);
    this.tf += +(a_bool != b_bool);
    this.tt += +(a_bool == 1 && b_bool == 1);
  },
  post: function(a, b) {
    if (this.tf + this.tt === 0) return 1.0;
    return this.tf / (this.tf + this.tt);
  }
});

// Dice dissimilarity
let dice = cwise({
  args: ['array', 'array'],
  pre: function(a, b) {
    this.tf = 0.0;
    this.tt = 0.0;
  },
  body: function(a, b) {
    var a_bool = Math.round(a);
    var b_bool = Math.round(b);
    this.tf += +(a_bool != b_bool);
    this.tt += +(a_bool == 1 && b_bool == 1);
  },
  post: function(a, b) {
    if (this.tf + this.tt === 0) return 1.0;
    return this.tf / (this.tf + 2 * this.tt);
  }
});

export default function(data, metric) {
  let nSamples = data.shape[0];
  let distance = ndarray(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);

  switch (metric) {
    case 'euclidean':
      for (let i = 0; i < nSamples; i++) {
        for (let j = i + 1; j < nSamples; j++) {
          let d = euclidean(data.pick(i, null), data.pick(j, null));
          distance.set(i, j, d);
          distance.set(j, i, d);
        }
      }
      break;
    case 'manhattan':
      for (let i = 0; i < nSamples; i++) {
        for (let j = i + 1; j < nSamples; j++) {
          let d = manhattan(data.pick(i, null), data.pick(j, null));
          distance.set(i, j, d);
          distance.set(j, i, d);
        }
      }
      break;
    case 'jaccard':
      for (let i = 0; i < nSamples; i++) {
        for (let j = i + 1; j < nSamples; j++) {
          let d = jaccard(data.pick(i, null), data.pick(j, null));
          distance.set(i, j, d);
          distance.set(j, i, d);
        }
      }
      break;
    case 'dice':
      for (let i = 0; i < nSamples; i++) {
        for (let j = i + 1; j < nSamples; j++) {
          let d = dice(data.pick(i, null), data.pick(j, null));
          distance.set(i, j, d);
          distance.set(j, i, d);
        }
      }
      break;
    default:
  }

  return distance;
}
