import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import cwise from 'cwise';
import pairwiseDistances from './pairwise-distances';

const MACHINE_EPSILON = Number.EPSILON || 2.220446049250313e-16;

let square = cwise({
  args: ['array'],
  body: function(a) { a = a * a; }
});

export default function(embedding, P, alpha) {

  let nSamples = embedding.shape[0];
  let dim = embedding.shape[1];

  // Q: Student's t-distribution
  let Q = ndarray(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);
  let n = pairwiseDistances(embedding, 'euclidean');
  square(n);
  let beta = (alpha + 1.0) / (-2.0);
  ops.powseq(ops.divseq(ops.addseq(n, 1), alpha), beta);
  for (let i = 0; i < nSamples; i++) { n.set(i, i, 0); }
  let sum_n = Math.max(ops.sum(n), MACHINE_EPSILON);
  ops.divs(Q, n, sum_n);
  ops.maxseq(Q, MACHINE_EPSILON);

  // Kullback-Leibler divergence of P and Q
  let temp = ndarray(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);
  let temp2 = ndarray(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);
  ops.div(temp, P, Q);
  ops.logeq(temp);
  ops.assign(temp2, P);
  let kl_divergence = ops.sum(ops.muleq(temp, temp2));

  // calculate gradient
  let grad = ndarray(new Float64Array(embedding.size), embedding.shape);
  let PQd = ndarray(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);
  ops.sub(PQd, P, Q);
  ops.muleq(PQd, n);
  for (let i = 0; i < nSamples; i++) {
    for (let d = 0; d < dim; d++) {
      let temp = ndarray(new Float64Array(embedding.shape[0]), [embedding.shape[0]]);
      ops.assign(temp, embedding.pick(null, d));
      ops.addseq(ops.negeq(temp), embedding.get(i, d));
      ops.muleq(temp, PQd.pick(i, null));
      grad.set(i, d, ops.sum(temp));
    }
  }
  let c = 2.0 * (alpha + 1.0) / alpha;
  ops.mulseq(grad, c);

  return [kl_divergence, grad];
}
