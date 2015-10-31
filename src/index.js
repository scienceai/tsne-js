import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from 'ndarray-pack';
import unpack from 'ndarray-unpack';
import cwise from 'cwise';
import randn from './lib/randn';
import pairwiseDistances from './lib/pairwise-distances';
import jointProbabilities from './lib/joint-probabilities';
import divergenceKL from './lib/kl-divergence';

export default class TSNE {
  constructor(config) {
    config = config || {};

    this.dim = config.dim || 2;
    this.perplexity = config.perplexity || 30.0;
    this.earlyExaggeration = config.earlyExaggeration || 4.0;
    this.learningRate = config.learningRate || 1000.0;
    this.tolerance = config.tolerance || 1e-5;
    this.nIter = config.nIter || 1000;
    this.metric = config.metric || 'euclidean';

    this.inputData = null;
    this.outputEmbedding = null;
  }

  init(inputData) {
    // input data as ndarray
    this.inputData = pack(inputData);
    // random initialization of output embedding
    this.outputEmbedding = randn(this.inputData.shape[0], this.dim);
  }

  run() {
    // calculate pairwise distances
    this.distances = pairwiseDistances(this.inputData, this.metric);

    this.alpha = Math.max(this.dim - 1, 1);
    this.P = jointProbabilities(this.distances, this.perplexity, this.tolerance);

    let error = Number.MAX_VALUE;
    let iter = 0;

    // early exaggeration
    ops.mulseq(this.P, this.earlyExaggeration);
    [error, iter] = this._gradDesc(iter, 50, 0.5, 0.0, 0.0);
    [error, iter] = this._gradDesc(iter + 1, 100, 0.8, 0.0, 0.0);

    // final optimization
    ops.divseq(this.P, this.earlyExaggeration);
    [error, iter] = this._gradDesc(iter + 1, this.nIter, 0.8, 1e-6, 1e-6);

    return [error, iter];
  }

  getOutput() {
    return unpack(this.outputEmbedding);
  }

  _gradDesc(iter, nIter, momentum, minGradNorm=1e-6, minErrorDiff=1e-6) {
    let nIterWithoutProg = 30;

    let update = ndarray(new Float64Array(this.outputEmbedding.size), this.outputEmbedding.shape);
    let gains = ndarray(new Float64Array(this.outputEmbedding.size).fill(1.0), this.outputEmbedding.shape);

    let error = Number.MAX_VALUE;
    let bestError = Number.MAX_VALUE;
    let bestIter = 0;

    let norm = cwise({
      args: ['array'],
      pre: function(a) { this.sum = 0.0; },
      body: function(a) { this.sum += a * a; },
      post: function(a) { return Math.sqrt(this.sum); }
    });

    let gainsUpdate = cwise({
      args: ['array', 'array', 'array'],
      body: function(c_gains, c_update, c_grad) {
        if (c_update * c_grad >= 0) { c_gains += 0.05; }
        else { c_gains *= 0.95; }
        // set mininum gain 0.01
        c_gains = Math.max(c_gains, 0.01);
      }
    });

    let i;
    for (i = iter; i < nIter; i++) {
      let [cost, grad] = divergenceKL(this.outputEmbedding, this.P, this.alpha);
      let errorDiff = Math.abs(cost - error);
      error = cost;
      let gradNorm = norm(grad);

      if (error < bestError) {
        bestError = error;
        bestIter = i;
      } else if ((i - bestIter) > nIterWithoutProg) {
        break;
      }

      if (minGradNorm >= gradNorm) break;
      if (minErrorDiff >= errorDiff) break;

      gainsUpdate(gains, update, grad);
      ops.muleq(grad, gains);

      let temp = ndarray(new Float64Array(grad.size), grad.shape);
      ops.muls(temp, grad, this.learningRate);
      ops.mulseq(update, momentum);
      ops.subeq(update, temp);
      ops.addeq(this.outputEmbedding, update);
    }

    return [error, i];
  }
}
