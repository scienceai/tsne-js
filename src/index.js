import { EventEmitter } from 'events';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from 'ndarray-pack';
import unpack from 'ndarray-unpack';
import cwise from 'cwise';
import randn from './randn';
import pairwiseDistances from './pairwise-distances';
import jointProbabilities from './joint-probabilities';
import divergenceKL from './kl-divergence';

class TSNE extends EventEmitter {
  constructor(config) {
    super();
    config = config || {};

    this.dim = config.dim || 2;
    this.perplexity = config.perplexity || 30.0;
    this.earlyExaggeration = config.earlyExaggeration || 4.0;
    this.learningRate = config.learningRate || 1000.0;
    this.nIter = config.nIter || 1000;
    this.metric = config.metric || 'euclidean';

    this.inputData = null;
    this.outputEmbedding = null;
  }

  init(opts) {
    opts = opts || {};

    let inputData = opts.data || [];
    let type = opts.type || 'dense';

    // format input data as ndarray
    if (type === 'dense') {

      this.inputData = pack(inputData);

    } else if (type === 'sparse') {

      let shape = [];
      let size = 1;
      for (let d = 0; d < inputData[0].length; d++) {
        let dimShape = Math.max.apply(null, inputData.map(coord => coord[d])) + 1;
        shape.push(dimShape);
        size *= dimShape;
      }
      this.inputData = ndarray(new Float64Array(size), shape);
      for (let coord of inputData) {
        this.inputData.set(...coord, 1);
      }

    } else {
      throw new Error('input data type must be dense or sparse');
    }

    // random initialization of output embedding
    this.outputEmbedding = randn(this.inputData.shape[0], this.dim);
  }

  run() {
    // calculate pairwise distances
    this.emit('progressStatus', 'Calculating pairwise distances');
    this.distances = pairwiseDistances(this.inputData, this.metric);

    this.emit('progressStatus', 'Calculating joint probabilities');
    this.alpha = Math.max(this.dim - 1, 1);
    this.P = jointProbabilities(this.distances, this.perplexity);

    let error = Number.MAX_VALUE;
    let iter = 0;

    // early exaggeration
    this.emit('progressStatus', 'Early exaggeration with momentum 0.5');
    ops.mulseq(this.P, this.earlyExaggeration);
    [error, iter] = this._gradDesc(iter, 50, 0.5, 0.0, 0.0);
    this.emit('progressStatus', 'Early exaggeration with momentum 0.8');
    [error, iter] = this._gradDesc(iter + 1, 100, 0.8, 0.0, 0.0);

    // final optimization
    this.emit('progressStatus', 'Final optimization with momentum 0.8');
    ops.divseq(this.P, this.earlyExaggeration);
    [error, iter] = this._gradDesc(iter + 1, this.nIter, 0.8, 1e-6, 1e-6);

    this.emit('progressStatus', 'Optimization end');
    return [error, iter];
  }

  rerun() {
    // random re-initialization of output embedding
    this.outputEmbedding = randn(this.inputData.shape[0], this.dim);

    // re-run with gradient descent
    let [error, iter] = this.run();

    return [error, iter];
  }

  getOutput() {
    return unpack(this.outputEmbedding);
  }

  getOutputScaled() {
    // scale output embedding to [-1, 1]
    let outputEmbeddingScaled = ndarray(new Float64Array(this.outputEmbedding.size), this.outputEmbedding.shape);
    let temp = ndarray(new Float64Array(this.outputEmbedding.shape[0]), [this.outputEmbedding.shape[0]]);

    for (let d = 0; d < this.outputEmbedding.shape[1]; d++) {
      let maxVal = ops.sup(ops.abs(temp, this.outputEmbedding.pick(null, d)));
      ops.divs(outputEmbeddingScaled.pick(null, d), this.outputEmbedding.pick(null, d), maxVal);
    }

    return unpack(outputEmbeddingScaled);
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

      this.emit('progressIter', [i, error, gradNorm]);

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

      this.emit('progressData', this.getOutputScaled());
    }

    return [error, i];
  }
}

export default TSNE;
