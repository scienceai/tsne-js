import ndarray from 'ndarray';
import ops from 'ndarray-ops';

const EPSILON_DBL = 1e-7;
const MACHINE_EPSILON = Number.EPSILON || 2.220446049250313e-16;

export default function(distances, perplexity, tolerance) {
  let n_steps = 100;
  let n_samples = distances.shape[0];
  let P_cond = ndarray(new Float64Array(n_samples * n_samples), [n_samples, n_samples]);
  let P = ndarray(new Float64Array(n_samples * n_samples), [n_samples, n_samples]);

  let beta, beta_min, beta_max = Infinity;
  let beta_sum = 0.0;

  let desired_entropy = Math.log(perplexity);
  let entropy_diff, entropy;
  let sum_Pi, sum_disti_Pi;

  for (let i = 0; i < n_samples; i++) {
    beta = 1.0;
    beta_min = -Infinity;
    beta_max = Infinity;

    for (let step = 0; step < n_steps; step++) {

      for (let j = 0; j < n_samples; j++) {
        P_cond.set(i, j, Math.exp(-distances.get(i, j) * beta));
      }

      P_cond.set(i, i, 0.0);
      sum_Pi = 0.0;
      for (let j = 0; j < n_samples; j++) {
        sum_Pi += P_cond.get(i, j);
      }
      if (sum_Pi == 0.0) sum_Pi = EPSILON_DBL;

      sum_disti_Pi = 0.0;
      for (let j = 0; j < n_samples; j++) {
        P_cond.set(i, j, P_cond.get(i, j) / sum_Pi);
        sum_disti_Pi += distances.get(i, j) * P_cond.get(i, j);
      }

      entropy = Math.log(sum_Pi) + beta * sum_disti_Pi;
      entropy_diff = entropy - desired_entropy;
      if (Math.abs(entropy_diff) <= tolerance) break;

      if (entropy_diff > 0.0) {
        beta_min = beta;
        if (beta_max == Infinity) { beta = beta * 2.0; }
        else { beta = (beta + beta_max) / 2.0; }
      } else {
        beta_max = beta;
        if (beta_min == -Infinity) { beta = beta / 2.0; }
        else { beta = (beta + beta_min) / 2.0; }
      }
    }

    beta_sum += beta;
  }

  ops.add(P, P_cond, P_cond.transpose(1,0));
  let sum_P = Math.max(ops.sum(P), MACHINE_EPSILON);
  ops.divseq(P, sum_P);
  ops.maxseq(P, MACHINE_EPSILON);
  return P;
}
