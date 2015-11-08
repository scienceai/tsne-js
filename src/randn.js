import ndarray from 'ndarray';

// random Gaussian distribution based on Box-Muller transform
function gaussRandom() {
  let u = 2 * Math.random() - 1;
  let v = 2 * Math.random() - 1;
  let r = u * u + v * v;
  if (r == 0 || r > 1) return gaussRandom();
  return u * Math.sqrt(-2 * Math.log(r) / r);
}

export default function(samples, dim) {

  let randArray = new Float64Array(samples * dim);

  for (let i = 0; i < randArray.length; i++) {
    randArray[i] = gaussRandom() * 1e-4;
  }

  return ndarray(randArray, [samples, dim]);

}
