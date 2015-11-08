import ndarray from 'ndarray';
import ops from 'ndarray-ops';

const EPSILON_DBL = 1e-7;
const MACHINE_EPSILON = Number.EPSILON || 2.220446049250313e-16;
const PERPLEXITY_TOLERANCE = 1e-5;

export default function(distances, perplexity) {
  let nSteps = 100;
  let nSamples = distances.shape[0];
  let P_cond = ndarray(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);
  let P = ndarray(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);

  let beta, betaMin, betaMax = Infinity;
  let betaSum = 0.0;

  let desired_entropy = Math.log(perplexity);
  let entropyDiff, entropy;
  let sum_Pi, sum_disti_Pi;

  for (let i = 0; i < nSamples; i++) {
    beta = 1.0;
    betaMin = -Infinity;
    betaMax = Infinity;

    for (let step = 0; step < nSteps; step++) {

      for (let j = 0; j < nSamples; j++) {
        P_cond.set(i, j, Math.exp(-distances.get(i, j) * beta));
      }

      P_cond.set(i, i, 0.0);
      sum_Pi = 0.0;
      for (let j = 0; j < nSamples; j++) {
        sum_Pi += P_cond.get(i, j);
      }
      if (sum_Pi == 0.0) sum_Pi = EPSILON_DBL;

      sum_disti_Pi = 0.0;
      for (let j = 0; j < nSamples; j++) {
        P_cond.set(i, j, P_cond.get(i, j) / sum_Pi);
        sum_disti_Pi += distances.get(i, j) * P_cond.get(i, j);
      }

      entropy = Math.log(sum_Pi) + beta * sum_disti_Pi;
      entropyDiff = entropy - desired_entropy;
      if (Math.abs(entropyDiff) <= PERPLEXITY_TOLERANCE) break;

      if (entropyDiff > 0.0) {
        betaMin = beta;
        if (betaMax == Infinity) { beta = beta * 2.0; }
        else { beta = (beta + betaMax) / 2.0; }
      } else {
        betaMax = beta;
        if (betaMin == -Infinity) { beta = beta / 2.0; }
        else { beta = (beta + betaMin) / 2.0; }
      }
    }

    betaSum += beta;
  }

  ops.add(P, P_cond, P_cond.transpose(1,0));
  let sum_P = Math.max(ops.sum(P), MACHINE_EPSILON);
  ops.divseq(P, sum_P);
  ops.maxseq(P, MACHINE_EPSILON);
  return P;
}
